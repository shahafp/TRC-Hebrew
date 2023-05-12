from transformers import TextClassificationPipeline


class TemporalRelationClassificationPipeline(TextClassificationPipeline):

    def check_model_type(self, supported_models):
        pass


