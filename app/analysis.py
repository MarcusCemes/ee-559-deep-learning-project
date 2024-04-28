from torch import load, no_grad
from torch.nn import Dropout, Linear, Module
from transformers import BertConfig, BertModel, BertTokenizer

TRANSFORM_COLUMNS = [
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
        self.tokenizer = BertTokenizer.from_pretrained("./tmp/hateBERT")

        config = BertConfig.from_pretrained("./tmp/hateBERT/config.json")

        self.model = MultilabelHateBert(BertModel(config))
        self.model.load_state_dict(load("./tmp/hateBERT_weights"))
        self.model.to("cuda")
        self.model.eval()

    def classify(self, text: str):
        input = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        input.to("cuda")

        with no_grad():
            return self.model(**input)


class MultilabelHateBert(Module):

    def __init__(self, model: BertModel):
        super(MultilabelHateBert, self).__init__()

        self.bertmodel = model
        self.dropout = Dropout(0.3)
        self.linear = Linear(768, len(TRANSFORM_COLUMNS))

    def forward(self, input_ids, attention_mask, token_type_ids):
        out = self.bertmodel(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )
        out = self.dropout(out.pooler_output)
        out = self.linear(out)

        return out
