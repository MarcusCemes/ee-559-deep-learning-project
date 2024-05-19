import logging
from pathlib import Path
from torch.utils.data import DataLoader, random_split


from .dataset import (
    CLASSES,
    HateSpeechDataset,
    load_dataset,
    preprocess_dataset,
    prepare_labels,
)
from .model import load_bert, MultiLabelHateBert
from .train import train

# == Configurations == #

EPOCHS = 20
DEVICE = "cuda"

DATASET_PATH = "/scratch/izar/cemes/measuring-hate-speech.parquet"
HATEBERT_PATH = "/scratch/izar/cemes/hateBERT"
OUTPUT_DIR = "/scratch/izar/cemes/output"

BATCH_SIZE = 64
LEARNING_RATE = 1e-05

UPDATE_NLTK = False

# == Main == #

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

logging.info("Loading BERT")
(tokenizer, model) = load_bert(HATEBERT_PATH)

logging.info("Preparing model")
model = MultiLabelHateBert(model, len(CLASSES)).to(DEVICE)

logging.info("Preparing dataset")
dataset = load_dataset(DATASET_PATH)
dataset = preprocess_dataset(dataset, UPDATE_NLTK)
dataset = prepare_labels(dataset)
dataset = HateSpeechDataset(dataset, tokenizer, device=DEVICE)


training_set, validation_set = random_split(dataset, [0.8, 0.2])

train_dataloader = DataLoader(
    training_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
)

test_dataloader = DataLoader(
    validation_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
)

logging.info(f"Training dataset length: {len(training_set)}")
logging.info(f"Validation dataset length: {len(validation_set)}")

logging.info("Training model")

output_dir = Path(OUTPUT_DIR)
train(model, train_dataloader, LEARNING_RATE, EPOCHS, output_dir, DEVICE)
