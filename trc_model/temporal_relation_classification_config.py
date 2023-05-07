from transformers import PretrainedConfig, BertConfig


class TemporalRelationClassificationConfig(BertConfig):
    model_type = "TemporalRelationClassification"

    def __init__(self, EMS1=0, EMS2=0, architecture=0, token_embeddings_size=0, **kwargs):
        super().__init__(**kwargs)
        self.architecture = architecture
        self.token_embeddings_size = token_embeddings_size
        self.EMS1 = EMS1
        self.EMS2 = EMS2
