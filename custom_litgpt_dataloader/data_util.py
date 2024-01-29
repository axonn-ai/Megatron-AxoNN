from lit_gpt.packed_dataset import CombinedDataset, PackedDataset
from lit_gpt.utils import CycleIterator
from torch.utils.data import DataLoader
from pathlib import Path
from axonn import axonn as ax
import torch.distributed as dist
from typing import Tuple, Union, Optional

data_config = [
    ("train_slimpajama", 69.3584),
    ("train_starcoder", 30.6),
    # 0.693584, 0.306416)
    # ("c4", 15.0),
    # ("cc", 67.0),
    # ("github", 4.5),
    # ("stackexchange", 2.0),
    # ("wikipedia", 4.5),
]


def create_dataloader(
    batch_size: int, block_size: int, data_dir: Path, shuffle: bool = True, seed: int = 12345
) -> DataLoader:
    datasets = []
    for prefix, _ in data_config:
        filenames = list(data_dir.glob(f"{prefix}*"))
        if not filenames:
            raise FileNotFoundError(
                f"No files found at {str(data_dir)} with prefix {prefix}. Did you forget to run `prepare_redpajama.py`?"
            )
        dataset = PackedDataset(
            filenames,
            n_chunks=4,
            block_size=block_size,
            shuffle=shuffle,
            seed=seed,
            num_processes=ax.config.G_data,
            process_rank=ax.config.data_parallel_rank,
        )
        datasets.append(dataset)

    if not datasets:
        raise RuntimeError(
            f"No data found at {data_dir}. Make sure you ran prepare_redpajama.py to create the dataset."
        )

    weights = [weight for _, weight in data_config]
    sum_weights = sum(weights)
    weights = [el / sum_weights for el in weights]

    combined_dataset = CombinedDataset(datasets=datasets, seed=seed, weights=weights)

    return DataLoader(combined_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)


def create_dataloaders(
    batch_size: int,
    block_size: int,
    train_data_dir: Path = Path("/lustre/orion/csc569/proj-shared/language_datasets/spj_star_combined_sample_tinyllama_tokd"),
    val_data_dir: Optional[Path] = Path("/lustre/orion/csc569/proj-shared/language_datasets/spj_star_combined_sample_tinyllama_tokd"),
    seed: int = 12345,
) -> Tuple[DataLoader, DataLoader]:
    # Increase by one because we need the next word as well
    effective_block_size = block_size + 1
    train_dataloader = create_dataloader(
        batch_size=batch_size,
        block_size=effective_block_size,
        data_dir=train_data_dir,
        shuffle=True,
        seed=seed,
    )
    val_dataloader = (
        create_dataloader(
            batch_size=batch_size,
            block_size=effective_block_size,
            data_dir=val_data_dir,
            shuffle=False,
            seed=seed,
        )
        if val_data_dir
        else None
    )
    return CycleIterator(train_dataloader), CycleIterator(val_dataloader)

if __name__ == "__main__":
    ax.init(G_inter=1, G_data=1, G_intra_r=8)
    train_loader, val_loader = create_dataloaders(
        batch_size=32,
        block_size=1024, #seuqnce length?
    )
    data = next(train_loader)
    print(dist.get_rank(), ":", data.view(-1)[:5])
