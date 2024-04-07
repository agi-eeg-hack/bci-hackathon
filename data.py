import csv
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Type


@dataclass
class NeurosityDatapoint:
    stddev: int
    channel_1: float
    channel_2: float
    channel_3: float
    channel_4: float
    channel_5: float
    channel_6: float
    channel_7: float
    channel_8: float
    unknown2: str
    timestamp_sec: float
    session_id: str
    left_fingers_up: int
    right_fingers_up: int
    doing: bool
    showing: bool

    @property
    def python_timestamp(self) -> datetime:
        return datetime.fromtimestamp(self.timestamp_sec / 1000)


@dataclass
class CombinedDatapoint:
    CP3: float
    C3: float
    F5: float
    PO3: float
    PO4: float
    F6: float
    C4: float
    CP4: float
    timestamp: float
    session_id: str

    @property
    def channel_1(self) -> float:
        return self.CP3

    @property
    def channel_2(self) -> float:
        return self.C3

    @property
    def channel_3(self) -> float:
        return self.F5

    @property
    def channel_4(self) -> float:
        return self.PO3

    @property
    def channel_5(self) -> float:
        return self.PO4

    @property
    def channel_6(self) -> float:
        return self.F6

    @property
    def channel_7(self) -> float:
        return self.C4

    @property
    def channel_8(self) -> float:
        return self.CP4


T = TypeVar("T")


class Dataset(Generic[T]):
    def __init__(self, data_path: Path, datapoint_class: Type[T]):
        self.datapoints: list[T] = []
        for csv_path in data_path.glob("*.csv"):
            session_id = csv_path.stem
            with open(csv_path, "r") as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                for row in reader:
                    if datapoint_class == NeurosityDatapoint:
                        datapoint = NeurosityDatapoint(
                            stddev=int(row[0]),
                            channel_1=float(row[1]),
                            channel_2=float(row[2]),
                            channel_3=float(row[3]),
                            channel_4=float(row[4]),
                            channel_5=float(row[5]),
                            channel_6=float(row[6]),
                            channel_7=float(row[7]),
                            channel_8=float(row[8]),
                            unknown2=row[9],
                            timestamp_sec=float(row[10]),
                            session_id=session_id,
                            left_fingers_up=int(row[11]) if row[11] else None,
                            right_fingers_up=int(row[12]) if row[12] else None,
                            doing=bool(row[13]),
                            showing=bool(row[14]),
                        )
                    elif datapoint_class == CombinedDatapoint:
                        datapoint = CombinedDatapoint(
                            CP3=float(row[0]),
                            C3=float(row[1]),
                            F5=float(row[2]),
                            PO3=float(row[3]),
                            PO4=float(row[4]),
                            F6=float(row[5]),
                            C4=float(row[6]),
                            CP4=float(row[7]),
                            timestamp=float(row[8]),
                            session_id=row[9],
                        )
                    else:
                        raise ValueError(
                            f"Unsupported datapoint class: {datapoint_class}"
                        )
                    self.datapoints.append(datapoint)

    def __len__(self):
        return len(self.datapoints)

    def __iter__(self):
        return iter(self.datapoints)

    def __getitem__(self, i):
        return self.datapoints[i]


# Example usage
data_ft_path = Path("data_ft")
neurosity_dataset = Dataset(data_ft_path, NeurosityDatapoint)

combined_data_path = Path("data_session")
combined_dataset = Dataset(combined_data_path, CombinedDatapoint)

# # Iterate over the datasets
# for datapoint in neurosity_dataset:
#     print(datapoint.python_timestamp, datapoint.session_id)

# for datapoint in combined_dataset:
#     print(datapoint.timestamp, datapoint.session_id)

print(f"Total Neurosity datapoints: {len(neurosity_dataset)}")
print(f"Total Combined datapoints: {len(combined_dataset)}")
print(f"First Neurosity datapoint: {neurosity_dataset[0]}")
print(f"First Combined datapoint: {combined_dataset[0]}")
