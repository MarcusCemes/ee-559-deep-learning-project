from pathlib import Path

from torch import save
from torch.nn import BCEWithLogitsLoss, Module
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm


def train(
    model: Module,
    training_loader: DataLoader,
    learning_rate: float,
    epochs: int,
    output_dir: Path,
    device: str,
):
    output_dir.mkdir(exist_ok=True)

    loss = BCEWithLogitsLoss()
    optimizer = Adam(params=model.parameters(), lr=learning_rate)

    model.train()

    for epoch in range(epochs):

        with tqdm(training_loader, unit="batch", total=len(training_loader)) as bar:
            for batch in bar:
                bar.set_description(f"Epoch {epoch}")

                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                token_type_ids = batch["token_type_ids"].to(device)
                targets = batch["targets"].to(device)

                outputs = model(input_ids, attention_mask, token_type_ids)
                optimizer.zero_grad()
                output = loss(outputs, targets)

                optimizer.zero_grad()
                output.backward()
                optimizer.step()

                bar.set_postfix(loss=output.item())

        save(model.state_dict(), output_dir / f"model_{epoch}.pth")

    save(model.state_dict(), output_dir / "model.pth")
