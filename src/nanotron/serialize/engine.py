from abc import ABC, abstractmethod
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional, Type, Dict

import ctypes
import json
import torch
from safetensors.torch import safe_open, save_file

from nanotron import logging
from nanotron.config import CheckpointingEngineType
from nanotron.serialize.metadata import TensorMetadata

logger = logging.get_logger(__name__)

SIZE_UINT64 = ctypes.sizeof(ctypes.c_uint64)


class CheckpointEngine(ABC):

    @abstractmethod
    def save_unsafe(self, state_dict: Dict[str, Any], path: str) -> None:
        pass

    @abstractmethod
    def save(self, state_dict: Dict[str, Any], path: str, metadata: dict) -> None:
        pass

    @abstractmethod
    def load_unsafe(self, path: str, map_location: Optional[str] = None) -> None:
        pass

    @abstractmethod
    def load(self, path: str, framework="pt", device="cpu") -> None:
        pass

    @abstractmethod
    def wait(self) -> None:
        pass


class TorchCheckpointEngine(CheckpointEngine):
    TENSOR_SUFFIX = "safetensors"

    def __init__(self, config: Optional[dict] = None):
        self.config = config

    def save_unsafe(self, state_dict: Dict[str, Any], path: str) -> None:
        torch.save(state_dict, path)

    def save(self, state_dict: Dict[str, Any], path: str, metadata: dict) -> None:
        logger.debug(f"Saving checkpoint {path}...")

        if "data" in state_dict:
            save_file(tensors=state_dict, filename=path, metadata=metadata)
        else:
            raise Exception(f"The state_dict given to checkpoint must contain a data field.")

    def load_unsafe(self, path: str, map_location: Optional[str] = None) -> None:
        logger.debug(f"Loading checkpoint from {path}...")
        return torch.load(path, map_location=map_location)

    def load(self, path: str, framework="pt", device: str = "cpu") -> None:
        logger.debug(f"Loading checkpoint from {path}...")
        with safe_open(path, framework, device=device) as fi:
            return fi.get_tensor("data")

    def wait(self) -> None:
        pass


class DataStatesCheckpointEngine(CheckpointEngine):
    TENSOR_SUFFIX = "datastates"

    engine = None

    def __init__(self, config: Optional[dict] = None):
        self.config = config

        try:
            from datastates import CkptEngine
        except ImportError as e:
            raise e

        if not torch.cuda.is_available():
            raise RuntimeError(f"[DataStates] CUDA is not available. Make sure CUDA drivers are installed and GPU is accessible.")

        default_config = {
            "host_cache_size": 16, # in GB
            "parser_threads": 2,
            "pin_host_cache": True
        }
        merged_config = default_config.copy() # Start with the default config
        if config is not None:
            merged_config.update(config)
        host_cache_size = int(merged_config["host_cache_size"] * (1 << 30)) # from GB to Bytes

        self.executor = ThreadPoolExecutor(max_workers=merged_config["parser_threads"])
        self.last_ckpt_version = -1

        try:
            # TODO tbouvier: the last parameter should be the rank
            self.engine = CkptEngine(host_cache_size, int(torch.cuda.current_device()), 0)
        except Exception as e:
            raise Exception(f"[DataStates] Got an exception during DataStates init: {e}")

    def save_unsafe(self, state_dict: Dict[str, Any], path: str) -> None:
        torch.save(state_dict, path)

    def save(self, state_dict: Dict[str, Any], path: str, metadata: dict) -> None:
        logger.debug(f"[DataStates] Saving checkpoint {path}...")

        if not isinstance(state_dict, (dict, OrderedDict)):
            raise Exception(f"""
                [DataStates] state_dict given to checkpoint must be a dictionary.
                Passed {type(state_dict)} instead for {path}.
            """)

        if "data" in state_dict:
            if metadata is None:
                raise Exception(f"[DataStates] saving a state_dict containing a tensor requires metadata to be defined.")

            header = json.dumps(metadata).encode("utf-8")
            header_size = len(header).to_bytes(SIZE_UINT64, 'little') # force the header size to take 8 bytes
            metadata_size = len(header_size) + len(header)

            # Launch asynchronous copies
            async_ckpt_list = [(metadata["version"], state_dict["data"], metadata_size, str(path))]
            self.engine.async_save(async_ckpt_list)

            # Writing the metadata at the beginning of the file, without waiting
            with open(path, 'wb') as f:
                f.seek(0)
                f.write(header_size)
                f.write(header)
        
        else:
            raise Exception(f"[DataStates] The state_dict given to checkpoint must contain a data field.")

    def load_unsafe(self, path: str, map_location: Optional[str] = None) -> None:
        raise NotImplementedError

    def load(self, path: str, framework="pt", device: str = "cpu") -> None:
        raise NotImplementedError

    def wait(self) -> None:
        return self.engine.wait()

    def __del__(self) -> None:
        if self.engine is not None:
            return self.engine.shutdown()


# Mapping enum values to their corresponding classes
_CONFIG_TO_CHECKPOINT_ENGINE_CLASS: Dict[CheckpointingEngineType, Type[CheckpointEngine]] = {
    CheckpointingEngineType.TORCH: TorchCheckpointEngine,
    CheckpointingEngineType.DATASTATES: DataStatesCheckpointEngine,
}

def create_checkpoint_engine_class(engine_type: CheckpointingEngineType) -> Type[CheckpointEngine]:
    if engine_type not in _CONFIG_TO_CHECKPOINT_ENGINE_CLASS:
        raise ValueError(f"Unknown checkpoint engine: {engine_type}")
    return _CONFIG_TO_CHECKPOINT_ENGINE_CLASS[engine_type]

def get_checkpoint_engine_type_from_instance(engine: CheckpointEngine) -> CheckpointingEngineType:
    engine_class = type(engine)
    for engine_type, cls in _CONFIG_TO_CHECKPOINT_ENGINE_CLASS.items():
        if cls == engine_class:
            return engine_type
    raise ValueError(f"Unknown checkpoint engine class: {engine_class}")
