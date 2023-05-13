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
        self.config = config
        self.num_labels = config.num_labels
        self.architecture = config.architecture
        self.class_weights = config.class_weights
        self.EMS1 = config.EMS1
        self.EMS2 = config.EMS2
        self.EME1 = config.EME1
        self.EME2 = config.EME2

        self.bert: BertModel = BertModel.from_pretrained(config.base_lm)
        self.bert.resize_token_embeddings(config.vocab_size)

        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        if self.architecture == 'SEQ_CLS':
            self.post_transformer = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.Dropout(classifier_dropout)
            )
            self.classification_layer = nn.Linear(config.hidden_size, config.num_labels)

        if self.architecture in ['EMP', 'ESS']:
            self.post_transformer_1 = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.Dropout(classifier_dropout)
            )

            self.post_transformer_2 = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.Dropout(classifier_dropout)
            )
            self.relation_representation = nn.Sequential(
                nn.Linear(config.hidden_size * 2, config.hidden_size * 2),
                nn.Dropout(classifier_dropout)
            )
            self.classification_layer = nn.Linear(config.hidden_size * 2, config.num_labels)

    def _get_entities_and_start_markers_indices(self, input_ids):
        em1_s = torch.tensor([(ids == self.EMS1).nonzero().item() for ids in input_ids], device=self.device)
        em1_e = torch.tensor([(ids == self.EME1).nonzero().item() for ids in input_ids], device=self.device)

        em2_s = torch.tensor([(ids == self.EMS2).nonzero().item() for ids in input_ids], device=self.device)
        em2_e = torch.tensor([(ids == self.EME2).nonzero().item() for ids in input_ids], device=self.device)

        return em1_s, em1_e, em2_s, em2_e

    def _max_pool_entity(self, mark_start, mark_end, sequence_output):
        return torch.stack(
            [torch.max(sentence[mark_start[i] + 1: mark_end[i]], dim=0, keepdim=True)[0] for i, sentence
             in
             enumerate(sequence_output)]).reshape(sequence_output.shape[0], -1).to(self.device)

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

        if self.architecture == 'SEQ_CLS':
            pooled_output = outputs[1]
            relation_representation = self.post_transformer(pooled_output)


        else:
            sequence_output = outputs[0]

            entity_mark_1_s, entity_mark_1_e, entity_mark_2_s, entity_mark_2_e = self._get_entities_and_start_markers_indices(
                input_ids)

            if self.architecture == 'EMP':
                entity_1_max_pool = self._max_pool_entity(entity_mark_1_s, entity_mark_1_e, sequence_output)
                entity_1_norm = self.post_transformer_1(entity_1_max_pool)

                entity_2_max_pool = self._max_pool_entity(entity_mark_2_s, entity_mark_2_e, sequence_output)
                entity_2_norm = self.post_transformer_2(entity_2_max_pool)

                relation_representation = self.relation_representation(torch.cat((entity_1_norm, entity_2_norm), 1))


            else:  # self.architecture == 'ESS'
                e1_start_mark_tensors = sequence_output[torch.arange(sequence_output.size(0)), entity_mark_1_s]
                e1_start_mark_norm = self.post_transformer_1(e1_start_mark_tensors)

                e2_start_mark_tensors = sequence_output[torch.arange(sequence_output.size(0)), entity_mark_2_s]
                e2_start_mark_norm = self.post_transformer_2(e2_start_mark_tensors)

                relation_representation = self.relation_representation(torch.cat((e1_start_mark_norm, e2_start_mark_norm), 1))

        logits = self.classification_layer(relation_representation)
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(weight=torch.tensor(self.class_weights, device=self.device))
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
