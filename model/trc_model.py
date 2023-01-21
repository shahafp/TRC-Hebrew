import torch
from torch import nn
from transformers import AutoConfig, AutoModel
from transformers import BertForSequenceClassification


class TRCModel(nn.Module):
    def __init__(self, output_size, tokenizer, check_point, architecture):
        super().__init__()
        self.architecture = architecture
        config = AutoConfig.from_pretrained(check_point)
        self.lm = AutoModel.from_pretrained(check_point, config=config)
        self.lm.resize_token_embeddings(len(tokenizer))

        self.classification_layers = None
        if self.architecture == 'SEQ_CLS':
            self.classification_layers = nn.Sequential(
                # nn.Dropout(),
                nn.Linear(config.hidden_size, output_size)
            )
        if self.architecture == 'EMP':
            self.e_1_linear = nn.Linear(config.hidden_size * 2, config.hidden_size)
            self.e_2_linear = nn.Linear(config.hidden_size * 2, config.hidden_size)

        if self.architecture in ['ESS', 'EF', 'EMP']:
            self.classification_layers = nn.Sequential(
                # nn.Dropout(),
                nn.Linear(config.hidden_size * 2, config.hidden_size),
                # nn.ReLU(),
                nn.Linear(config.hidden_size, output_size)
            )

    def forward(self,
                input_ids,
                entity_1,
                entity_2,
                entity_mark_1_s,
                entity_mark_2_s,
                masks,
                marks_only=False,
                entity_and_marks=False):

        lm_outputs = self.lm(input_ids, attention_mask=masks, output_hidden_states=True)
        lm_outputs = lm_outputs.last_hidden_state

        predictions = None
        if self.architecture == 'SEQ_CLS':
            cls_token_feature_tensor = lm_outputs[:, 0, :]
            predictions = self.classification_layers(cls_token_feature_tensor)
        else:
            e1_start_mark_tensors = lm_outputs[torch.arange(lm_outputs.size(0)), entity_mark_1_s]
            e2_start_mark_tensors = lm_outputs[torch.arange(lm_outputs.size(0)), entity_mark_2_s]

            e1_tensor = lm_outputs[torch.arange(lm_outputs.size(0)), entity_1]
            e2_tensor = lm_outputs[torch.arange(lm_outputs.size(0)), entity_2]

            if self.architecture == 'ESS':
                e_start_markers_cat = torch.cat((e1_start_mark_tensors, e2_start_mark_tensors), 1)
                predictions = self.classification_layers(e_start_markers_cat)

            if self.architecture == 'EF':
                events_cat = torch.cat((e1_tensor, e2_tensor), 1)
                predictions = self.classification_layers(events_cat)

            if self.architecture == 'EMP':
                e1_and_start_mark = self.e_1_linear(torch.cat((e1_start_mark_tensors, e1_tensor), 1))
                e2_and_start_mark = self.e_2_linear(torch.cat((e2_start_mark_tensors, e2_tensor), 1))
                both_e_cat = torch.cat((e1_and_start_mark, e2_and_start_mark), 1)
                predictions = self.classification_layers(both_e_cat)
        return predictions

        # if marks_only or entity_and_marks:
        #     e1_mark_tensors = outputs[torch.arange(outputs.size(0)), entity_mark_1_s]
        #     e2_mark_tensors = outputs[torch.arange(outputs.size(0)), entity_mark_2_s]
        #
        #     if entity_and_marks:
        #         entity_1_tensor = outputs[torch.arange(outputs.size(0)), entity_1]
        #         entity_2_tensor = outputs[torch.arange(outputs.size(0)), entity_2]
        #
        #         e1_mark_tensors = torch.cat((e1_mark_tensors, entity_1_tensor), 1)
        #         e2_mark_tensors = torch.cat((e2_mark_tensors, entity_2_tensor), 1)
        #
        #     classification_tensors = torch.cat((e1_mark_tensors, e2_mark_tensors), 1)
        #
        # else:
        #     classification_tensors = outputs[:, 0, :]
        #
        # dropout_output = self.dropout(classification_tensors)
        # predictions = self.linear(dropout_output)
        # return predictions
