from pathlib import Path

from datasets import Dataset, concatenate_datasets, load_dataset


def load_imagenet_validation(num_samples: int, cache_dir: str | None = None):
    dataset = _load_prepared_arrow_validation(cache_dir)
    if dataset is None:
        print(" - Prepared ImageNet Arrow cache not found; falling back to Hugging Face download/cache.")
        dataset = load_dataset("ILSVRC/imagenet-1k", split="validation", cache_dir=cache_dir)

    if num_samples < len(dataset):
        dataset = dataset.select(range(num_samples))
    return dataset


def _load_prepared_arrow_validation(cache_dir: str | None):
    for root in _candidate_dataset_roots(cache_dir):
        if not root.exists():
            continue

        groups = {}
        for arrow in root.glob("default/*/*/imagenet-1k-validation-*.arrow"):
            groups.setdefault(arrow.parent, []).append(arrow)

        if not groups:
            continue

        parent, files = max(groups.items(), key=lambda item: len(item[1]))
        files = sorted(files)
        if not files:
            continue

        print(f" - Using prepared ImageNet Arrow cache: {parent} ({len(files)} validation shards)")
        datasets = [Dataset.from_file(str(path)) for path in files]
        return datasets[0] if len(datasets) == 1 else concatenate_datasets(datasets)

    return None


def _candidate_dataset_roots(cache_dir: str | None):
    seen = set()
    bases = []
    if cache_dir:
        bases.append(Path(cache_dir).expanduser())

    for base in bases:
        for root in (
            base / "datasets" / "ILSVRC___imagenet-1k",
            base / "datasets" / "imagenet-1k",
        ):
            resolved = root.resolve() if root.exists() else root
            if resolved in seen:
                continue
            seen.add(resolved)
            yield root
