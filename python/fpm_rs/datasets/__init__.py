"""Offline spectral loading and explicit dataset registry/cache operations."""

from fpm_rs._core import (
    Dataset,
    DatasetRegistry,
    DatasetRegistryEntry,
    SpectralDataset,
    load_spectral_dataset,
    open_dataset,
)

__all__ = [
    "Dataset",
    "DatasetRegistry",
    "DatasetRegistryEntry",
    "SpectralDataset",
    "load_spectral_dataset",
    "open_dataset",
]
