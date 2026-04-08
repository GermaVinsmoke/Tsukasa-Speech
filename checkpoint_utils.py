import pickle
import warnings
from pathlib import Path

import torch


def _looks_like_git_lfs_pointer(path):
    file_path = Path(path)
    if not file_path.is_file() or file_path.stat().st_size > 1024:
        return False

    with file_path.open("r", encoding="utf-8", errors="ignore") as handle:
        header = handle.read(128)

    return header.startswith("version https://git-lfs.github.com/spec/v1")


def load_torch_checkpoint(path, map_location="cpu", trusted=True, **kwargs):
    """Load local checkpoints across PyTorch versions.

    PyTorch 2.6 changed torch.load(..., weights_only=...) to default to True.
    Older project checkpoints can require pickle-based loading, so for trusted
    local files we retry with weights_only=False when that specific failure
    occurs.
    """
    if _looks_like_git_lfs_pointer(path):
        raise RuntimeError(
            f"{path} is a Git LFS pointer, not the actual checkpoint file. "
            "Fetch the model weights with Git LFS before running the app."
        )

    try:
        return torch.load(path, map_location=map_location, weights_only=True, **kwargs)
    except TypeError:
        # Older PyTorch versions do not support the weights_only argument.
        return torch.load(path, map_location=map_location, **kwargs)
    except pickle.UnpicklingError as exc:
        if not trusted or "Weights only load failed" not in str(exc):
            raise

        warnings.warn(
            f"Retrying trusted checkpoint load with weights_only=False: {path}",
            RuntimeWarning,
        )
        return torch.load(path, map_location=map_location, weights_only=False, **kwargs)
