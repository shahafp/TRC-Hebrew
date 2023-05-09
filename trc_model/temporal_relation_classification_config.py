from transformers import PretrainedConfig, BertConfig


class TemporalRelationClassificationConfig(BertConfig):
    model_type = "TemporalRelationClassification"

    def __init__(self, ES_ID=0, EMS1=0, EMS2=0, architecture=0, **kwargs):
        super().__init__(**kwargs)
        self.architecture = architecture
        self.EMS1 = EMS1
        self.EMS2 = EMS2
        self.ES_ID = ES_ID
