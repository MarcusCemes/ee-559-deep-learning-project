from torch import Tensor, load, no_grad
from torch.nn import Dropout, Linear, Module, Softmax
from transformers import BertConfig, BertModel, BertTokenizer

DEVICE = "cuda"
HATEBERT_PATH = "./tmp/hateBERT"
WEIGHTS_PATH = "./tmp/checkpoint_2024-05-20"

COLUMNS = [
    "respect",
    "insult",
    "humiliate",
    "status",
    "dehumanize",
    "violence",
    "genocide",
    "attack_defend",
]


class Analyser:

    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained(HATEBERT_PATH)

        config = BertConfig.from_pretrained(f"{HATEBERT_PATH}/config.json")

        self.model = MultilabelHateBert(BertModel(config))
        self.model.load_state_dict(load(WEIGHTS_PATH))
        self.model.to(DEVICE)
        self.model.eval()

    def classify(self, text: str) -> dict[str, float]:
        input = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        input.to(DEVICE)

        with no_grad():
            output = self.model(**input)

            return {COLUMNS[i]: output[0][i].item() for i in range(len(COLUMNS))}


class MultilabelHateBert(Module):

    def __init__(self, model: BertModel):
        super(MultilabelHateBert, self).__init__()

        self.bert_model = model
        self.dropout = Dropout(0.3)
        self.linear = Linear(768, len(COLUMNS))
        self.softmax = Softmax(dim=1)

    def forward(self, input_ids, attention_mask, token_type_ids) -> Tensor:
        out = self.bert_model(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )
        out = self.dropout(out.pooler_output)
        out = self.linear(out)
        print(out)
        out = self.softmax(out)

        return out
