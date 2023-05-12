from transformers import PretrainedConfig, BertConfig


class TemporalRelationClassificationConfig(BertConfig):
    model_type = "TemporalRelationClassification"

    def __init__(self, special_markers=False, pool_tokens=False, ES_ID=0, EMS1=0, EMS2=0, architecture=0, **kwargs):
        super().__init__(**kwargs)
        self.base_lm = self.name_or_path
        self.pool_tokens = pool_tokens
        self.special_markers = special_markers
        self.architecture = architecture
        self.EMS1 = EMS1
        self.EMS2 = EMS2
        self.ES_ID = ES_ID
