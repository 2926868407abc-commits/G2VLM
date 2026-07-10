try:
    from .edit_dataset import UnifiedEditIterableDataset
except ModuleNotFoundError as exc:
    if exc.name not in {"edit_dataset", "data.interleave_datasets.edit_dataset"}:
        raise
    UnifiedEditIterableDataset = None

