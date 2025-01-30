from enum import StrEnum, auto


class CheckpointingEngineType(StrEnum):
    TORCH = auto()
    DATASTATES = auto()


def is_valid_checkpointing_engine_type(engine_type: str) -> bool:
    try:
        CheckpointingEngineType(engine_type.lower())
        return True
    except ValueError:
        return False
