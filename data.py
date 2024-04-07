import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Literal


class NeurosityDatapoint:
    def __init__(self, row, session_id):
        self.stddev = int(row[0])
        self.channel_1 = float(row[1])
        self.channel_2 = float(row[2])
        self.channel_3 = float(row[3])
        self.channel_4 = float(row[4])
        self.channel_5 = float(row[5])
        self.channel_6 = float(row[6])
        self.channel_7 = float(row[7])
        self.channel_8 = float(row[8])
        self.unknown2 = row[9]
        self.timestamp_sec = float(row[10])
        self.left_fingers_up = int(row[11]) if row[11] else None
        self.right_fingers_up = int(row[12]) if row[12] else None
        self.doing = bool(row[13])
        self.showing = bool(row[14])
        self.session_id = session_id

    @property
    def python_timestamp(self) -> datetime:
        return datetime.fromtimestamp(self.timestamp_sec / 1000)


class Dataset:
    def __init__(self, data_ft_path: Path):
        self.datapoints: list[NeurosityDatapoint] = []
        for csv_path in data_ft_path.glob("*.csv"):
            session_id = csv_path.stem
            with open(csv_path, "r") as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                for row in reader:
                    datapoint = NeurosityDatapoint(row, session_id)
                    self.datapoints.append(datapoint)

    def __len__(self):
        return len(self.datapoints)

    def __iter__(self):
        return iter(self.datapoints)

    def __getitem__(self, i):
        return self.datapoints[i]


data_ft_path = Path("data_ft")
dataset = Dataset(data_ft_path)

# Example usage
for datapoint in dataset:
    print(datapoint.python_timestamp, datapoint.stddev, datapoint.session_id)

print(f"Total datapoints: {len(dataset)}")
print(f"First datapoint: {dataset[0]}")
