from csv import DictWriter
from datetime import datetime, timedelta
from io import TextIOWrapper
from random import randint
import time
import cattrs
from neurosity import NeurositySDK
from pathlib import Path

from neurosity_types import FIELDNAMES, NeurosityDatapoint

DUMPS_DIR = Path("./dumps")


class NeurosityCSVWriter(DictWriter):
    def __init__(self, file: TextIOWrapper) -> None:
        super().__init__(
            file,
            fieldnames=FIELDNAMES,
        )


def init_neurosity() -> NeurositySDK:
    neurosity = NeurositySDK(
        {
            "device_id": "cf3de5eefaae268e02d689f07abf5cba",
        }
    )

    neurosity.login({"email": "mrthinger@gmail.com", "password": "agihack"})
    return neurosity


def main() -> None:
    neurosity = init_neurosity()
    filename = str(datetime.now())
    DUMPS_DIR.mkdir(exist_ok=True)
    dump_path = DUMPS_DIR / f"{filename}.csv"

    with open(dump_path, "w") as f:
        neurosity_writer = NeurosityCSVWriter(f)
        prompted = False
        doing = False
        left = None
        right = None
        stddev = 0

        def callback(data: dict[any, any]):
            ctime = datetime.fromtimestamp(int(data["info"]["startTime"]) / 1000)
            delta = timedelta(milliseconds=1000 / 256)
            sensor_data = data["data"]
            for i in range(16):
                neurosity_writer.writerow(
                    cattrs.unstructure(
                        NeurosityDatapoint(
                            stddev=0,
                            channel_1=sensor_data[0][i],
                            channel_2=sensor_data[1][i],
                            channel_3=sensor_data[2][i],
                            channel_4=sensor_data[3][i],
                            channel_5=sensor_data[4][i],
                            channel_6=sensor_data[5][i],
                            channel_7=sensor_data[6][i],
                            channel_8=sensor_data[7][i],
                            unknown2=None,
                            left_fingers_up=left,
                            right_fingers_up=right,
                            doing=doing,
                            showing=showing,
                            timestamp_sec=ctime.timestamp() * 1000,
                        )
                    )
                )
                ctime += delta

        unsubscribe = neurosity.brainwaves_raw(callback)

        def callback2(data):
            stddev = min([datum["status"] for datum in data])

        unsubscribe2 = neurosity.signal_quality(callback2)
        while True:
            left = randint(0, 5)
            right = randint(0, 5)
            print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
            print("Get ready to show...")
            print(f"Left: {left}\t\t\tRight: {right}")
            showing = True
            time.sleep(5)
            print("Show!")
            doing = True
            time.sleep(5)
            print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
            print("Reset your hands")
            showing = False
            doing = False
            left = None
            right = None
            time.sleep(2)


main()
