from typing import Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, MSELoss
from transformers import BertPreTrainedModel, BertModel, BertForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput

from trc_model.temporal_relation_classification_config import TemporalRelationClassificationConfig


class TokenPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, token_tensor: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        pooled_output = self.dense(token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class TemporalRelationClassification(BertForSequenceClassification):
    config_class = TemporalRelationClassificationConfig

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.special_markers = config.special_markers
        self.pool_tokens = config.pool_tokens
        self.ES_ID = config.ES_ID
        self.EMS1 = config.EMS1
        self.EMS2 = config.EMS2
        self.EME1 = config.EME1
        self.EME2 = config.EME2
        self.architecture = config.architecture
        self.config = config

        self.bert = BertModel.from_pretrained(config.base_lm)
        if self.bert.config.vocab_size != config.vocab_size:
            self.bert.resize_token_embeddings(config.vocab_size)

        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        if config.pool_tokens:
            self.ems_1_pooler = TokenPooler(config)
            self.ems_2_pooler = TokenPooler(config)
            self.e_1_pooler = TokenPooler(config)
            self.e_2_pooler = TokenPooler(config)

        self.dropout = nn.Dropout(classifier_dropout)

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
        # self.post_init()

    def _get_entities_and_start_markers_indices(self, input_ids):
        em1_s = torch.tensor([(ids == self.EMS1).nonzero().item() for ids in input_ids], device=self.device)
        em1_e = torch.tensor([(ids == self.EME1).nonzero().item() for ids in input_ids], device=self.device)

        em2_s = torch.tensor([(ids == self.EMS2).nonzero().item() for ids in input_ids], device=self.device)
        em2_e = torch.tensor([(ids == self.EME2).nonzero().item() for ids in input_ids], device=self.device)

        return em1_s, em1_e, em2_s, em2_e

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

        logits = None
        if self.architecture == 'SEQ_CLS':
            pooled_output = outputs[1]

            pooled_output = self.dropout(pooled_output)
            logits = self.classification_layers(pooled_output)
        else:
            sequence_output = outputs[0]

            sequence_output = self.dropout(sequence_output)

            entity_mark_1_s, entity_mark_1_e, entity_mark_2_s, entity_mark_2_e = self._get_entities_and_start_markers_indices(input_ids)

            e1_start_mark_tensors = sequence_output[torch.arange(sequence_output.size(0)), entity_mark_1_s]
            e2_start_mark_tensors = sequence_output[torch.arange(sequence_output.size(0)), entity_mark_2_s]

            e1_tensor = torch.stack(
                [torch.max(sentence[entity_mark_1_s[i]: entity_mark_1_e[i]], dim=0, keepdim=True)[0] for i, sentence in
                 enumerate(sequence_output)]).reshape(sequence_output.shape[0], -1).to(self.device)
            e2_tensor = torch.stack(
                [torch.max(sentence[entity_mark_2_s[i]: entity_mark_2_e[i]], dim=0, keepdim=True)[0] for i, sentence in
                 enumerate(sequence_output)]).reshape(sequence_output.shape[0], -1).to(self.device)

            if self.pool_tokens:
                e1_start_mark_tensors = self.ems_1_pooler(e1_start_mark_tensors)
                e2_start_mark_tensors = self.ems_2_pooler(e2_start_mark_tensors)

                e1_tensor = self.e_1_pooler(e1_tensor)
                e2_tensor = self.e_2_pooler(e2_tensor)

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
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
