from csv import DictReader
from io import TextIOWrapper
import json
import os
from pathlib import Path
from typing import Callable, Generator, Iterator, NamedTuple, TypeVar
from .neurosity_types import FIELDNAMES, NeurosityDatapoint, NeurosityDatasetMeta

import cattrs


class NeurosityCSVReader(DictReader):
    def __init__(self, file: TextIOWrapper) -> None:
        super().__init__(
            file,
            fieldnames=FIELDNAMES,
        )

    def __next__(self):
        next = super().__next__()
        return cattrs.structure(next, NeurosityDatapoint)


class NeurosityDataset(NamedTuple):
    name: str
    path: str
    meta: NeurosityDatasetMeta | None


def datasets_in_parent(root_path: Path) -> Generator[NeurosityDataset, None, None]:
    for dirstr, _, filenames in os.walk(root_path):
        meta = None
        dirpath = Path(dirstr)

        if "dataset-metadata.json" in filenames:
            with open(dirpath / "dataset-metadata.json") as meta_file:
                meta = cattrs.structure(json.load(meta_file), NeurosityDatasetMeta)
        if "dataset.csv" in filenames:
            yield NeurosityDataset(
                name=dirpath.name,
                path=dirpath / "dataset.csv",
                meta=meta,
            )


def process_datasets_in_parent(
    root_path: Path,
    callback: Callable[[NeurosityDataset, Iterator[NeurosityDatapoint]], None],
):
    for dataset in datasets_in_parent(root_path):
        if dataset.meta.valid:
            with open(dataset.path, "r") as f:
                callback(dataset, NeurosityCSVReader(f))


A = TypeVar("A")


# def get_chunks(data_folder: Path, count: int, size: int) -> list[list[NeurosityDatapoint]]:
#     files = [(data_folder / f) for f in os.listdir(data_folder) if (data_folder / f).is_file()]
#     num_files = len(files)
#     picks_per_file = [0] * num_files
#     chunks = []
#     for _ in range(count):
#         for i in range(num_files):
#             if picks_per_file[i] >= size:
#                 picks_per_file[i] = 0
#                 continue
#             with open(files[i], "r") as f:
#                 reader = NeurosityCSVReader(f)
#                 chunks.append(reader.read(size))
#                 picks_per_file[i] += 1
#     return chunks
