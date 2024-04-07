import torch
from dataclasses import dataclass
from gm.load_dataset import read_csv_neurosity_dataset
from mistral.model import Transformer
from mistral.tokenizer import Tokenizer
from pathlib import Path
import torch.nn.functional as F
from tqdm import tqdm

@dataclass
class TrainConfig:
    model_path: str = "./models/mistral-tar"
    data_path: str = "./gm/combined_dataset_finetune.csv"
    epochs: int = 10
    batch_size: int = 128
    learning_rate: float = 1e-5


config = TrainConfig()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float16

train_dataloader, val_dataloader, training = read_csv_neurosity_dataset(
    config.data_path, batch_size=config.batch_size
)

def train(config: TrainConfig):
    # Load the pretrained model and tokenizer
    model = Transformer.from_folder(
        Path(config.model_path), max_batch_size=config.batch_size
    )

    # model.to(DEVICE)
    # tokenizer = Tokenizer(str(Path(config.model_path) / "tokenizer.model"))

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # Training loop
    for epoch in range(config.epochs):
        model.train()

        for x, _ in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{config.epochs}"):
            databatch: torch.Tensor = x["encoder_cont"].to(device=DEVICE, dtype=DTYPE)  # dont ask...

            x = databatch[:, :-1, :]
            y = databatch[:, 1:, :]

            b, t, c = x.shape
            y_hat = model(x, seqlens=[c] * b)

            loss = F.mse_loss(y_hat, y)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"l: {loss.item():.4f}, e [{epoch+1}/{config.epochs}]")

    # Save the fine-tuned model
    torch.save(model.state_dict(), "finetuned_model.pth")


if __name__ == "__main__":
    train(config)
