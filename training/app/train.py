from logging import info
from pathlib import Path

from sklearn import metrics
from torch import no_grad, save
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, Module
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm


def train(
    model: Module,
    training_loader: DataLoader,
    validation_loader: DataLoader,
    learning_rate: float,
    epochs: int,
    output_dir: Path,
    device: str,
):
    output_dir.mkdir(exist_ok=True)

    loss_binary = BCEWithLogitsLoss()
    loss_multi = CrossEntropyLoss()

    optimizer = Adam(params=model.parameters(), lr=learning_rate)

    model.train()

    for epoch in range(epochs):
        epoch = epoch + 1

        with tqdm(
            training_loader, unit="batch", total=len(training_loader), mininterval=60
        ) as bar:
            for batch in bar:
                bar.set_description(f"Epoch {epoch}")

                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                token_type_ids = batch["token_type_ids"].to(device)
                target = batch["target"].to(device)
                labels = batch["labels"].to(device)

                output_binary, output_multi = model(
                    input_ids, attention_mask, token_type_ids
                )

                optimizer.zero_grad()

                output_binary = loss_binary(output_binary, target.unsqueeze(1))
                output_multi = loss_multi(output_multi, labels)

                output_binary.backward(retain_graph=True)
                output_multi.backward()

                optimizer.step()

        save(model.state_dict(), output_dir / f"model_{epoch}.pth")

        validate(model, validation_loader, device)

    save(model.state_dict(), output_dir / "model.pth")


def validate(model: Module, loader: DataLoader, device: str):
    model.eval()

    val_target_binary = []
    val_targets_multi = []

    val_output_binary = []
    val_outputs_multi = []

    with no_grad():
        for batch in loader:

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            target = batch["target"].to(device)
            labels = batch["labels"].to(device)

            output_binary, output_multi = model(
                input_ids, attention_mask, token_type_ids
            )

            val_target_binary.extend(target.cpu().detach().numpy().tolist())
            val_output_binary.extend(
                output_binary.squeeze(1).cpu().detach().numpy().tolist()
            )

            val_targets_multi.extend(labels.cpu().detach().numpy().tolist())
            val_outputs_multi.extend(output_multi.cpu().detach().numpy().tolist())

    val_output_binary = threshold_list(val_output_binary)
    val_outputs_multi = list(map(threshold_list, val_outputs_multi))

    accuracy_binary = metrics.accuracy_score(val_target_binary, val_output_binary)
    accuracy_multi = metrics.accuracy_score(val_targets_multi, val_outputs_multi)

    f1_score_multi = metrics.f1_score(
        val_targets_multi, val_outputs_multi, average="weighted"
    )

    info(f"Binary accuracy Score = {accuracy_binary}")
    info(f"Multi accuracy Score = {accuracy_multi}")
    info(f"F1 Score (multi) = {f1_score_multi}")


def threshold_list(x: list) -> list:
    return [1 if i >= 0.5 else 0 for i in x]
