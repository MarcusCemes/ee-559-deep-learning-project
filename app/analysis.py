from torch import Tensor, argmax, load, no_grad
from torch.nn import Dropout, Linear, Module
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertConfig,
    BertModel,
)

DEVICE = "cuda"
BERT_MODEL = "distilbert-base-uncased"
HATEBERT_PATH = "./tmp/hateBERT"
MULTI_WEIGHTS_PATH = "./tmp/multilabel_bert_five_epochs.pth"
BINARY_WEIGHTS_PATH = "./tmp/base_model_bert_two_epoch.pth"


ID2LABEL = {
    0: "race",
    1: "religion",
    2: "origin",
    3: "gender",
    4: "sexuality",
    5: "age",
    6: "disability",
}

LABEL2ID = {
    "race": 0,
    "religion": 1,
    "origin": 2,
    "gender": 3,
    "sexuality": 4,
    "age": 5,
    "disability": 6,
}


class Analyser:

    def __init__(self):
        self.mtokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)

        self.mmodel = AutoModelForSequenceClassification.from_pretrained(
            BERT_MODEL,
            problem_type="multi_label_classification",
            num_labels=len(LABEL2ID),
            label2id=LABEL2ID,
            id2label=ID2LABEL,
        )

        self.mmodel.load_state_dict(load(MULTI_WEIGHTS_PATH))
        self.mmodel.to(DEVICE)
        self.mmodel.eval()

        self.btokenizer = AutoTokenizer.from_pretrained(HATEBERT_PATH)

        bconfig = BertConfig.from_pretrained(f"{HATEBERT_PATH}/config.json")
        self.bmodel = BinaryHateBert(BertModel(bconfig))
        self.bmodel.load_state_dict(load(BINARY_WEIGHTS_PATH))
        self.bmodel.to(DEVICE)
        self.bmodel.eval()

    def classify(self, text: str) -> tuple[bool, dict[str, float]]:
        binput = self.btokenizer(
            text, padding=True, truncation=True, return_tensors="pt"
        )
        minput = self.mtokenizer(
            text, padding=True, truncation=True, return_tensors="pt"
        )

        binput.to(DEVICE)
        minput.to(DEVICE)

        with no_grad():
            output_binary = self.bmodel(**binput)
            output_multi = self.mmodel(**minput)

            sentiments = {
                ID2LABEL[key]: output_multi[0].squeeze()[key].item() for key in ID2LABEL
            }

            is_hate_speech = bool(argmax(output_binary.squeeze()).item())

            return is_hate_speech, sentiments


class BinaryHateBert(Module):
    def __init__(self, bert_model: Module):
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

    def __init__(self, bert_model: Module, outputs: int):
        super(MultilabelHateBert, self).__init__()

        self.bert_model = bert_model
        self.dropout = Dropout(0.3)
        self.multi = Linear(768, outputs)

    def forward(self, input_ids, attention_mask, token_type_ids) -> Tensor:
        out = self.bert_model(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )

        out = self.dropout(out.pooler_output)
        out = self.multi(out)

        return out
