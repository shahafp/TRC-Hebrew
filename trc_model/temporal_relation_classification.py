from typing import Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import BertPreTrainedModel, BertModel
from transformers.modeling_outputs import SequenceClassifierOutput

from trc_model.temporal_relation_classification_config import TemporalRelationClassificationConfig


class TemporalRelationClassification(BertPreTrainedModel):
    config_class = TemporalRelationClassificationConfig

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.EMS1 = config.EMS1
        self.EMS2 = config.EMS2
        self.architecture = config.architecture
        self.config = config

        self.bert = BertModel(config)
        self.bert.resize_token_embeddings(config.token_embeddings_size)

        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        # self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.classification_layers = None
        if self.architecture == 'SEQ_CLS':
            self.classification_layers = nn.Sequential(
                nn.Linear(config.hidden_size, config.num_labels)
            )
        if self.architecture == 'EMP':
            self.e_1_linear = nn.Linear(config.hidden_size * 2, config.hidden_size)
            self.e_2_linear = nn.Linear(config.hidden_size * 2, config.hidden_size)

        if self.architecture in ['ESS', 'EF', 'EMP']:
            self.classification_layers = nn.Sequential(
                nn.Linear(config.hidden_size * 2, config.hidden_size),
                nn.Linear(config.hidden_size, config.num_labels)
            )

        # Initialize weights and apply final processing
        self.post_init()

    def _get_entities_and_start_markers_indices(self, input_ids):
        em1_s = torch.tensor([(ids == self.EMS1).nonzero().item() for ids in input_ids], device=self.device)
        entity_1 = em1_s + 1

        em2_s = torch.tensor([(ids == self.EMS2).nonzero().item() for ids in input_ids], device=self.device)
        entity_2 = em2_s + 1
        return em1_s, entity_1, em2_s, entity_2

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = outputs.last_hidden_state

        last_hidden_state = self.dropout(last_hidden_state)

        logits = None
        if self.architecture == 'SEQ_CLS':
            cls_token_feature_tensor = last_hidden_state[:, 0, :]
            logits = self.classification_layers(cls_token_feature_tensor)
        else:
            entity_mark_1_s, entity_1, entity_mark_2_s, entity_2 = self._get_entities_and_start_markers_indices(
                input_ids)

            e1_start_mark_tensors = last_hidden_state[torch.arange(last_hidden_state.size(0)), entity_mark_1_s]
            e2_start_mark_tensors = last_hidden_state[torch.arange(last_hidden_state.size(0)), entity_mark_2_s]

            e1_tensor = last_hidden_state[torch.arange(last_hidden_state.size(0)), entity_1]
            e2_tensor = last_hidden_state[torch.arange(last_hidden_state.size(0)), entity_2]

            if self.architecture == 'ESS':
                e_start_markers_cat = torch.cat((e1_start_mark_tensors, e2_start_mark_tensors), 1)
                logits = self.classification_layers(e_start_markers_cat)

            if self.architecture == 'EF':
                events_cat = torch.cat((e1_tensor, e2_tensor), 1)
                logits = self.classification_layers(events_cat)

            if self.architecture == 'EMP':
                e1_and_start_mark = self.e_1_linear(torch.cat((e1_start_mark_tensors, e1_tensor), 1))
                e2_and_start_mark = self.e_2_linear(torch.cat((e2_start_mark_tensors, e2_tensor), 1))
                both_e_cat = torch.cat((e1_and_start_mark, e2_and_start_mark), 1)
                logits = self.classification_layers(both_e_cat)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
