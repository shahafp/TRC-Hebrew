from transformers import TextClassificationPipeline


class TemporalRelationClassificationPipeline(TextClassificationPipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
