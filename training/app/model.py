from logging import info, warn
from typing import Optional
from torch.nn import Dropout, Linear, Module
from transformers import BertModel, BertTokenizer


def load_bert(path: Optional[str]) -> tuple[BertTokenizer, BertModel]:
    if path:
        info("Loading BERT from local path")
        tokenizer = BertTokenizer.from_pretrained(path)
        model = BertModel.from_pretrained(path)
    else:
        info("Loading BERT from Hugging Face")
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = BertModel.from_pretrained("bert-base-uncased")

    return (tokenizer, model)  # type: ignore


class MultiLabelHateBert(Module):

    def __init__(self, bert_model: BertModel, outputs: int):
        super(MultiLabelHateBert, self).__init__()

        self.bert_model = bert_model
        self.dropout = Dropout(0.3)
        self.binary = Linear(768, 1)
        self.multi = Linear(768, outputs)

    def forward(self, ids, mask, token_type_ids, freeze_bert: bool = False):
        if freeze_bert:
            warn("Freezing BERT!")
            for param in self.bert_model.parameters():
                param.requires_grad = False

        output = self.bert_model(
            ids, attention_mask=mask, token_type_ids=token_type_ids
        )

        output = self.dropout(output.pooler_output)

        output_binary = self.binary(output)
        output_multi = self.multi(output)

        return output_binary, output_multi
