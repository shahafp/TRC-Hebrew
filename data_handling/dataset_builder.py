import pandas as pd
from datasets import Dataset


class DatasetBuilder:
    tokenizer = None

    @staticmethod
    def build(paths_mapping: dict, label2id: dict):
        all_datasets = {}
        for split, path in paths_mapping.items():
            data_df = pd.read_csv(path)
            data_df['named_label'] = data_df['label']
            data_df['label'] = data_df['label'].apply(lambda l: label2id[l])
            clean_data_df = data_df[['text', 'label', 'named_label']]
            clean_data_df.to_csv(f'data_handling/clean_data/{split}.csv', index=False)
            all_datasets[split] = Dataset.from_pandas(data_df[['text', 'label']])
        return all_datasets

    @classmethod
    def preprocess_function(cls, examples):
        return cls.tokenizer(examples["text"], truncation=True)

    @classmethod
    def tokenize_dataset(cls, raw_datasets, tokenizer):
        cls.tokenizer = tokenizer
        tokenized_datasets = {}
        for split, dataset in raw_datasets.items():
            tokenized_datasets[split] = dataset.map(cls.preprocess_function, batched=True)
        return tokenized_datasets
