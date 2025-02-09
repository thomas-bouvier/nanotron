from abc import ABC, abstractmethod
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from packaging.version import Version
from pathlib import Path
from typing import Any, Optional, Type, Dict

import ctypes
import json
import numpy as np
import pickle
import torch
from safetensors.torch import safe_open, save_file

from nanotron import logging
from nanotron.config import CheckpointingEngineType
from nanotron.serialize.metadata import TensorMetadata

logger = logging.get_logger(__name__)

SIZE_UINT64 = ctypes.sizeof(ctypes.c_uint64)
KEY_SEPARATOR = "|"


class CheckpointEngine(ABC):
    CHECKPOINT_VERSION: Version

    def __init__(self, config: Optional[dict] = None):
        self.config = config

    @abstractmethod
    def save(self, state_dict: Dict[str, Any], path: Path, metadata: Optional[TensorMetadata] = None) -> None:
        pass

    @abstractmethod
    def load_unsafe(self, path: Path, map_location: Optional[str] = None) -> None:
        pass

    @abstractmethod
    def load(self, path: Path, framework="pt", device="cpu") -> None:
        pass

    @abstractmethod
    def wait(self) -> None:
        pass


class TorchCheckpointEngine(CheckpointEngine):
    CHECKPOINT_VERSION = Version("1.4")
    TENSOR_SUFFIX = "safetensors"

    def __init__(self, config: Optional[Dict] = None):
        self.config = config

    def _save_unsafe(self, state_dict: Dict[str, Any], path: Path) -> None:
        torch.save(state_dict, path)

    def save(self, state_dict: Dict[str, Any], path: Path, metadata: Optional[TensorMetadata] = None) -> None:
        logger.debug(f"Saving checkpoint {path}...")

        if metadata is not None:
            if "data" in state_dict:
                save_file(tensors=state_dict, filename=path, metadata=metadata.to_str_dict())
            else:
                raise Exception(f"The state_dict given to checkpoint must contain a data field as tensor metadata was passed")
        else:
            self._save_unsafe(state_dict, path)

    def load_unsafe(self, path: Path, map_location: Optional[str] = None) -> None:
        logger.debug(f"Loading checkpoint from {path}...")
        return torch.load(path, map_location=map_location)

    def load(self, path: Path, framework="pt", device: str = "cpu") -> None:
        logger.debug(f"Loading checkpoint from {path}...")
        with safe_open(path, framework, device=device) as fi:
            return fi.get_tensor("data")

    def wait(self) -> None:
        pass


class DataStatesCheckpointEngine(CheckpointEngine):
    CHECKPOINT_VERSION = Version("1.0")
    TENSOR_SUFFIX = "datastates"

    engine = None

    def __init__(self, config: Optional[Dict] = None):
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

    def save(self, state_dict: Dict[str, Any], path: Path, metadata: Optional[TensorMetadata] = None) -> None:
        logger.debug(f"[DataStates] Saving checkpoint {path}...")

        #if not isinstance(state_dict, (dict, OrderedDict, RandomStates)):
        #    raise Exception(f"[DataStates] state_dict given to checkpoint must be a dictionary. Passed {type(state_dict)} instead for {path}.")

        if metadata is not None:
            if "data" not in state_dict:
                raise Exception(f"The state_dict given to checkpoint must contain a data field as tensor metadata was passed")

        header = {}
        async_copies = {}
        _start_tensor_offset = 0
        _end_tensor_offset = 0

        def _parse_state(key, data):
            nonlocal _start_tensor_offset, _end_tensor_offset
            try:
                if torch.is_tensor(data):
                    tensor_size = data.numel() * data.element_size()
                    _end_tensor_offset += tensor_size

                    header[key] = {
                        "dtype": str(data.dtype),
                        "shape": tuple(data.shape),
                        "data_offsets": [_start_tensor_offset, _end_tensor_offset],
                    }
                    if key == "data":
                        header[key].update(**(metadata.to_str_dict() if metadata is not None else {}))

                    data = data.contiguous()
                    async_copies[key] = {
                        "tensor": data,
                        "file_offset": _start_tensor_offset
                    }
                    _start_tensor_offset = _end_tensor_offset
                    snapshot = f"TENSOR{KEY_SEPARATOR}{key}"

                elif isinstance(data, list):
                    snapshot = [None] * len(data)
                    for (idx, ele) in enumerate(data):
                        new_key = f"{key}{KEY_SEPARATOR}{idx}" if len(key) else f"{idx}"
                        snapshot[idx] = _parse_state(new_key, ele)

                elif isinstance(data, (dict, OrderedDict)):
                    snapshot = {}
                    for (k, v) in data.items():
                        new_key = f"{key}{KEY_SEPARATOR}{k}" if len(key) else f"{k}"
                        snapshot[k] = _parse_state(new_key, v)

                else:
                    snapshot = data

                return snapshot

            except Exception as exc:
                raise Exception(f"[DataStates] Cannot parse {key}, exception: {exc}, data is {data}")

        lean_state_dict = pickle.dumps(_parse_state("", state_dict), protocol=pickle.HIGHEST_PROTOCOL)
        _end_tensor_offset += len(lean_state_dict)

        header.update({
            "datastates_metadata": {
                "data_offsets": [_start_tensor_offset, _end_tensor_offset],
            }
        })
        enc_header = json.dumps(header).encode("utf-8")
        enc_header_size = len(enc_header).to_bytes(SIZE_UINT64, 'little') # force the header size to take 8 bytes
        metadata_size = len(enc_header_size) + len(enc_header)

        # Launch asynchronous copies
        async_ckpt_list = []
        for _, v in async_copies.items():
            # We offset file_offsets by metadata_size
            v["file_offset"] += metadata_size
            async_ckpt_list.append((str(self.CHECKPOINT_VERSION), v["tensor"], v["file_offset"], str(path)))

        self.engine.async_save(async_ckpt_list)

        with open(path, 'wb') as f:
            f.seek(0)
            f.write(enc_header_size)
            f.write(enc_header)
            # Write the lean state dict towards the end of the file
            f.seek(_start_tensor_offset + metadata_size)
            f.write(lean_state_dict)

    def load_unsafe(self, path: Path, map_location: Optional[str] = None) -> None:
        logger.debug(f"Loading checkpoint from {path}...")
        return torch.load(path, map_location=map_location)

    def load(self, path: Path, framework="pt", device: str = "cpu") -> None:
        try:
            version = 0
            f = open(path, 'rb')
            f.seek(0)

            header_size_bytes = f.read(SIZE_UINT64)
            header_size = int.from_bytes(header_size_bytes, 'little')
            metadata_size = header_size + SIZE_UINT64
            header = json.loads(f.read(header_size))

            [start_offset, end_offset] = np.add(header["datastates_metadata"]["data_offsets"], metadata_size)
            del(header["datastates_metadata"])

            f.seek(start_offset)
            data = pickle.loads(f.read(end_offset - start_offset))

            try:
                restore_list = []

                for k, v in header.items():
                    split_k = deque(k.split(KEY_SEPARATOR))
                    dtype = v["dtype"]
                    if dtype.startswith("torch"):
                        dtype = dtype.replace('torch.', '')
                    shape = v["shape"]
                    [start_offset, end_offset] = np.add(v["data_offsets"], metadata_size)

                    pre_dest = data
                    dest = data
                    while len(split_k):
                        sub_k = split_k.popleft()
                        if sub_k.isdigit():
                            sub_k = int(sub_k)
                        pre_dest = dest
                        dest = dest[sub_k]

                    if dest != f"TENSOR{KEY_SEPARATOR}{k}":
                        raise Exception(f"[DataStates] The key in header {k} does not match key at location {dest}")

                    tensor_restored = torch.zeros(size=tuple(shape), dtype=getattr(torch, dtype))
                    restore_list.append((version, tensor_restored, start_offset, str(path)))

                    f.seek(start_offset)
                    buffer = f.read(end_offset - start_offset)
                    tensor_restored = torch.frombuffer(buffer, dtype=getattr(torch, dtype)).reshape(tuple(shape))
                    pre_dest[sub_k] = tensor_restored

                self.engine.load(restore_list)

            except Exception as exc:
                raise Exception(f"[DataStates] Got error with tensor loading {dtype}, {shape}, {exc}")
                self.logger.info(f"[DataStates] Loaded checkpoint from {path}.")
                return data

        except Exception as exc:
            logger.error(f"[DataStates][ERROR] Could not load {path}, exception: {exc}")

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
