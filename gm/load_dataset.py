import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet


def read_csv_neurosity_dataset(file):
    data = pd.read_csv(file)

    max_encoder_length = 257
    max_prediction_length = 1
    N = int(len(data) * 0.85)

    # data = data.head(10)

    data["index"] = data.index

    data["left_hand"] = data["left_hand"].astype(str)
    data["right_hand"] = data["right_hand"].astype(str)
    data["showing"] = data["showing"].astype(str)
    data["doing"] = data["doing"].astype(str)

    data.sample(frac=1)

    training = TimeSeriesDataSet(
        data,
        time_idx="index",
        target=[
            "CP3",
            "C3",
            "F5",
            "PO3",
            "PO4",
            "F6",
            "C4",
            "CP4",
            "left_hand",
            "right_hand",
        ],
        group_ids=["session_id"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        time_varying_unknown_reals=["CP3", "C3", "F5", "PO3", "PO4", "F6", "C4", "CP4"],
        time_varying_unknown_categoricals=["left_hand", "right_hand"],
        # time_varying_known_reals=["timestamp"],
        time_varying_known_categoricals=["showing", "doing"],
    )

    # create validation and training dataset
    validation = TimeSeriesDataSet.from_dataset(
        training,
        data,
        min_prediction_idx=training.index.time.max() + 1,
        stop_randomization=True,
    )
    training = TimeSeriesDataSet.from_dataset(
        training, data.head(N), min_prediction_idx=0, stop_randomization=True
    )
    batch_size = 128
    train_dataloader = training.to_dataloader(
        train=True, batch_size=batch_size, num_workers=2
    )
    val_dataloader = validation.to_dataloader(
        train=False, batch_size=batch_size, num_workers=2
    )

    return train_dataloader, val_dataloader, training




if __name__ == "__main__":
    train_dataloader, val_dataloader, training = read_csv_neurosity_dataset(
        "/workspace/evan/hack/gm/combined_dataset_finetune.csv"
    )
    
    for x, y in train_dataloader:
        x['encoder_cont'] # b, ts, c
        # np.stack y 10 features take the 8 chans
        # y is a python list y[0] is the list of 

        # yt=np.concatenate(y[0][:8], axis=1) # b, c


