from typing import Any, Literal
from attr import define
from datetime import datetime

FIELDNAMES = [
    "stddev",
    "channel_1",
    "channel_2",
    "channel_3",
    "channel_4",
    "channel_5",
    "channel_6",
    "channel_7",
    "channel_8",
    "unknown2",
    "timestamp_sec",
    "left_fingers_up",
    "right_fingers_up",
    "doing",
    "showing",
]


@define
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
    unknown2: Any
    timestamp_sec: float
    left_fingers_up: Literal[0, 1, 2, 3, 4, 5] | None
    right_fingers_up: Literal[0, 1, 2, 3, 4, 5] | None
    doing: bool
    showing: bool

    @property
    def python_timestamp(self) -> datetime:
        return datetime.fromtimestamp(self.timestamp_sec / 1000)


@define
class NeurosityDatasetMeta:
    modelId: str
    id: str
    name: str
    timestamp: str
    label: str
    experimentId: str
    metric: str
    sessionId: str
    duration: int
    storagePath: str
    type: str
    anonymousSubjectId: str

    @property
    def python_timestamp(self) -> datetime:
        return datetime.fromtimestamp(self.timestamp / 1000)

    @property
    def valid(self) -> bool:
        # So we can easily filter out datasets that are not of interest
        return True
