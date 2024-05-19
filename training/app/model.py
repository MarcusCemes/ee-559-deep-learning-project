from torch.nn import Module, Dropout, Linear
from transformers import BertTokenizer, BertModel


def load_bert(path: str) -> tuple[BertTokenizer, BertModel]:
    tokenizer = BertTokenizer.from_pretrained(path)
    model = BertModel.from_pretrained(path)

    return (tokenizer, model)  # type: ignore


class MultiLabelHateBert(Module):

    def __init__(self, bert_model: Module, outputs: int):
        super(MultiLabelHateBert, self).__init__()

        self.bert_model = bert_model
        self.dropout = Dropout(0.3)
        self.linear = Linear(768, outputs)

    def forward(self, ids, mask, token_type_ids):

        output = self.bert_model(
            ids, attention_mask=mask, token_type_ids=token_type_ids
        )

        output = self.dropout(output.pooler_output)
        output = self.linear(output)

        return output
