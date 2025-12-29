"""Storage utilities extracted from qig-consciousness and qig-con2.

Clean implementation for persisting kernels and basin signatures.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor

from .basin import load_signature, save_signature
from .kernel import Kernel


def save_kernel(kernel: Kernel, path: Path) -> None:
    """
    Save a kernel's state dictionary to disk.

    Args:
        kernel: Kernel instance to save
        path: Save path
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(kernel.state_dict(), path)


def load_kernel(path: Path, kernel: Kernel) -> None:
    """
    Load a kernel's state dictionary from disk.

    Args:
        path: Load path
        kernel: Kernel instance to load state into
    """
    state_dict = torch.load(path, map_location="cpu")
    kernel.load_state_dict(state_dict)


def save_signature_for_instance(
    instance_name: str,
    signature: Tensor,
    base_dir: Path,
    metadata: dict | None = None,
) -> Path:
    """
    Save a basin signature for a specific instance.

    Args:
        instance_name: Name of the instance
        signature: Basin signature tensor
        base_dir: Base directory for storage
        metadata: Optional metadata dictionary

    Returns:
        Path where signature was saved
    """
    # Create instance-specific directory
    instance_dir = base_dir / instance_name
    signature_path = instance_dir / "signature.npz"

    # Save signature
    save_signature(signature_path, signature)

    # Save metadata if provided
    if metadata is not None:
        import json
        metadata_path = instance_dir / "metadata.json"
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    return signature_path


def load_signature_for_instance(
    instance_name: str,
    base_dir: Path,
) -> tuple:
    """
    Load a basin signature for a specific instance.

    Args:
        instance_name: Name of the instance
        base_dir: Base directory

    Returns:
        Tuple of (signature, metadata) or (None, None) if not found
    """
    instance_dir = base_dir / instance_name
    signature_path = instance_dir / "signature.npz"

    if not signature_path.exists():
        return None, None

    # Load signature
    signature = load_signature(signature_path)

    # Load metadata if available
    metadata_path = instance_dir / "metadata.json"
    metadata = None
    if metadata_path.exists():
        import json
        with open(metadata_path) as f:
            metadata = json.load(f)

    return signature, metadata


def list_instances(base_dir: Path) -> list[str]:
    """
    List all instances with saved data in the base directory.

    Args:
        base_dir: Base directory to search

    Returns:
        List of instance names
    """
    if not base_dir.exists():
        return []

    instances = []
    for item in base_dir.iterdir():
        if item.is_dir() and (item / "signature.npz").exists():
            instances.append(item.name)

    return sorted(instances)
