import torch
from torch.utils.data import DataLoader
from mistral.model import Transformer
from mistral.tokenizer import Tokenizer
from pathlib import Path
from data import NeurosityDatapoint, CombinedDatapoint, Dataset


def train(
    model_path: str, data_path: str, epochs: int, batch_size: int, learning_rate: float
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the pretrained model and tokenizer
    model = Transformer.from_folder(Path(model_path), max_batch_size=batch_size)
    model.to(device)
    tokenizer = Tokenizer(str(Path(model_path) / "tokenizer.model"))

    # Load the datasets
    neurosity_dataset = Dataset(Path(data_path) / "data_ft", NeurosityDatapoint)
    combined_dataset = Dataset(Path(data_path) / "data_session", CombinedDatapoint)

    # Create data loaders
    neurosity_loader = DataLoader(
        neurosity_dataset, batch_size=batch_size, shuffle=True
    )
    combined_loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)

    # Set up the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        model.train()

        for neurosity_batch, combined_batch in zip(neurosity_loader, combined_loader):
            # Prepare the input data
            neurosity_input = tokenizer.encode(str(neurosity_batch))
            combined_input = tokenizer.encode(str(combined_batch))

            # Move data to the device
            neurosity_input = torch.tensor(neurosity_input, device=device)
            combined_input = torch.tensor(combined_input, device=device)

            # Forward pass
            neurosity_output = model(neurosity_input, seqlens=[len(neurosity_input)])
            combined_output = model(combined_input, seqlens=[len(combined_input)])

            # Compute the loss
            loss = torch.mean(neurosity_output) + torch.mean(combined_output)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    # Save the fine-tuned model
    torch.save(model.state_dict(), "finetuned_model.pth")


if __name__ == "__main__":
    model_path = "./models/mistral-hf"
    data_path = "."
    epochs = 10
    batch_size = 8
    learning_rate = 1e-5

    train(model_path, data_path, epochs, batch_size, learning_rate)
