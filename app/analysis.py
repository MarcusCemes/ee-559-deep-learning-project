from lime.lime_text import LimeTextExplainer
from torch import Tensor, argmax, load, no_grad
from torch.nn import Dropout, Linear, Module
from torch.nn import functional as F
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertConfig,
    BertModel,
    pipeline,
)

DEVICE = "cuda"

BERT_MODEL = "distilbert-base-uncased"
HATEBERT_PATH = "./tmp/hateBERT"
MULTI_WEIGHTS_PATH = "./tmp/multilabel_bert_five_epochs.pth"
BINARY_WEIGHTS_PATH = "./tmp/base_model_bert_two_epoch.pth"

INTERPRET_SAMPLES = 50

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

    def __init__(self, interpret=True):
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

        self.interpreter = (
            Interpreter(self.mmodel, self.mtokenizer) if interpret else None
        )

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

    def interpret(self, text: str):
        if not self.interpreter:
            return {}

        return self.interpreter.interpret(text)


class Interpreter:

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

        class_names = [ID2LABEL[i] for i in range(len(ID2LABEL))]

        self.explainer = LimeTextExplainer(
            class_names=class_names, split_expression="\\s+", bow=False
        )

        self.pipeline = pipeline(
            task="text-classification",
            model=model,
            tokenizer=tokenizer,
            device="cuda",
            top_k=None,
        )

    def interpret(self, instance: str) -> dict[str, float]:
        attributions = {}

        exp = self.explainer.explain_instance(
            instance, self._predictor, num_features=200, num_samples=INTERPRET_SAMPLES
        )

        explanation_dict = dict(list(exp.as_map().values())[0])
        tokens = instance.split(" ")

        for i in range(len(tokens)):
            attributions[tokens[i]] = explanation_dict[i]

        return attributions

    def _predictor(self, texts):
        encodings = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=256,
            return_tensors="pt",
        ).to(DEVICE)

        logits = self.model(**encodings).logits
        probabilities = F.softmax(logits, dim=1)

        return probabilities.cpu().detach().numpy()


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
