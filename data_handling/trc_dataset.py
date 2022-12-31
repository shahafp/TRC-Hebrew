import pandas as pd
from torch.utils.data import Dataset
from transformers import BertTokenizerFast


class TRCDataset(Dataset):
    def __init__(self, data_path, tokenizer, e1_idx, e2_idx):
        self.e2_idx = e2_idx
        self.e1_idx = e1_idx
        self.df = pd.read_csv(data_path)
        self.tokenizer = tokenizer
        self.label_2_id = {label: i for i, label in enumerate(set(self.df['label'].values))}
        self.data = []
        self.prepare_data()

    def prepare_data(self):
        for row in self.df.itertuples():
            text = row.text
            label = row.label
            tokenized_data = self.tokenizer(text, padding=True)
            input_ids = tokenized_data['input_ids']
            attention_masks = tokenized_data['attention_mask']
            em1_s = input_ids.index(self.e1_idx)
            em2_s = input_ids.index(self.e2_idx)

            row_dict = {'text': text,
                        'input_ids': input_ids,
                        'entity_1': em1_s + 1,
                        'entity_2': em2_s + 1,
                        'entity_mark_1_s': em1_s,
                        'entity_mark_2_s': em2_s,
                        'attention_masks': attention_masks,
                        'label': label}
            self.data.append(row_dict)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


if __name__ == '__main__':
    alephbert_tokenizer = BertTokenizerFast.from_pretrained('onlplab/alephbert-base')
    alephbert_tokenizer.add_tokens(['[א1]', '[/א1]', '[א2]', '[/א2]'])
    E1_start = alephbert_tokenizer.convert_tokens_to_ids('[א1]')
    E2_start = alephbert_tokenizer.convert_tokens_to_ids('[א2]')
    data = TRCDataset('split_data/train.csv', alephbert_tokenizer, E1_start, E2_start)
    print()
