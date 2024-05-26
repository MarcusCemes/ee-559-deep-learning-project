from torch import Tensor, argmax, load, no_grad
from torch.nn import Dropout, Linear, Module, Softmax
from transformers import BertConfig, BertModel, BertTokenizer

DEVICE = "cuda"
HATEBERT_PATH = "./tmp/hateBERT"
MULTI_WEIGHTS_PATH = "./tmp/checkpoint_2024-05-21"
BINARY_WEIGHTS_PATH = "./tmp/base_model_bert_two_epoch.pth"

CLASSES = [
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

        self.mmodel = MultilabelHateBert(BertModel(config), len(CLASSES))
        self.mmodel.load_state_dict(load(MULTI_WEIGHTS_PATH))
        self.mmodel.to(DEVICE)
        self.mmodel.eval()

        self.bmodel = BinaryHateBert(BertModel(config))
        self.bmodel.load_state_dict(load(BINARY_WEIGHTS_PATH))
        self.bmodel.to(DEVICE)
        self.bmodel.eval()

    def classify(self, text: str) -> tuple[bool, dict[str, float]]:
        input = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        input.to(DEVICE)

        with no_grad():
            _, output_multi = self.mmodel(**input)
            output_binary = self.bmodel(**input)

            sentiments = {c: output_multi[0][i].item() for i, c in enumerate(CLASSES)}
            is_hate_speech = bool(argmax(output_binary[0]))

            return is_hate_speech, sentiments


class BinaryHateBert(Module):
    def __init__(self, bert_model: BertModel):
        super(BinaryHateBert, self).__init__()

        self.bertmodel = bert_model
        self.dropout = Dropout(0.3)
        self.linear = Linear(768, 2)

    def forward(self, input_ids, attention_mask, token_type_ids) -> Tensor:
        out = self.bertmodel(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )
        out = self.dropout(out.pooler_output)
        out = self.linear(out)

        return out


class MultilabelHateBert(Module):

    def __init__(self, bert_model: BertModel, outputs: int):
        super(MultilabelHateBert, self).__init__()

        self.bert_model = bert_model
        self.dropout = Dropout(0.3)
        self.binary = Linear(768, 1)
        self.multi = Linear(768, outputs)
        self.softmax = Softmax(dim=1)

    def forward(
        self, input_ids, attention_mask, token_type_ids
    ) -> tuple[Tensor, Tensor]:
        out = self.bert_model(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )
        out = self.dropout(out.pooler_output)

        output_binary = self.binary(out)

        output_multi = self.multi(out)
        output_multi = self.softmax(output_multi)

        return output_binary, output_multi
