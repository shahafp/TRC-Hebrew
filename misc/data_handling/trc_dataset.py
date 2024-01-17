import pandas as pd
from torch.utils.data import Dataset
from transformers import BertTokenizerFast


class TRCDataset(Dataset):
    def __init__(self, data_path):
        # self.e2_idx = e2_idx
        # self.e1_idx = e1_idx
        self.df = pd.read_csv(data_path)
        # self.tokenizer = tokenizer
        # self.label_2_id = {label: i for i, label in enumerate(set(self.df['label'].values))}
        # self.id_2_label = {i: label for label, i in self.label_2_id.items()}
        # self.data = []
        # self.prepare_data()


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {'text': self.df['text'][idx],
                'label': self.df['label'][idx]}


if __name__ == '__main__':
    alephbert_tokenizer = BertTokenizerFast.from_pretrained('onlplab/alephbert-base')
    alephbert_tokenizer.add_tokens(['[א1]', '[/א1]', '[א2]', '[/א2]'])
    E1_start = alephbert_tokenizer.convert_tokens_to_ids('[א1]')
    E2_start = alephbert_tokenizer.convert_tokens_to_ids('[א2]')
    data = TRCDataset('split_data/train.csv', alephbert_tokenizer, E1_start, E2_start)
    print()
