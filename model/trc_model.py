import torch
from torch import nn
from transformers import AutoConfig, AutoModel


class TRCModel(nn.Module):
    def __init__(self, output_size, tokenizer):
        super().__init__()
        config = AutoConfig.from_pretrained('onlplab/alephbert-base')
        self.model = AutoModel.from_pretrained('onlplab/alephbert-base', config=config)
        self.model.resize_token_embeddings(len(tokenizer))

        self.dropout = nn.Dropout()
        self.linear = nn.Linear(config.hidden_size * 2, output_size)

    def forward(self, text,
                entity_1, entity_2,
                entity_mark_1_s, entity_mark_1_e,
                entity_mark_2_s, entity_mark_2_e,
                masks, marks_only=False, entity_and_marks=False):
        outputs = self.model(text, attention_mask=masks, output_hidden_states=True)
        predictions = outputs.last_hidden_state

        if marks_only or entity_and_marks:
            e1_mark_tensors = outputs[torch.arange(outputs.size(0)), entity_mark_1_s]
            e2_mark_tensors = outputs[torch.arange(outputs.size(0)), entity_mark_2_s]

            entities_tensors = torch.cat((e1_mark_tensors, e2_mark_tensors), 1)

            if entity_and_marks:
                entity_1_tensor =

            predictions = self.linear(entities_tensors)
        return predictions

