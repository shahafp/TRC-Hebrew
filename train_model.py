import torch
from torch.utils.data import DataLoader
from torch import optim, nn
from transformers import BertTokenizerFast

from data_handling.trc_dataset import TRCDataset
from model.trc_model import TRCModel
from trainer.trainer import Trainer
from trainer.training_utils import get_parameters

if torch.backends.cuda.is_built():
    device_name = 'cuda'

else:
    device_name = 'cpu'

device = torch.device(device_name)
print('device:', device)

BATCH_SIZE = 4
MODEL_CHECKPOINT = 'onlplab/alephbert-base'
TRAINING_LAYERS = 52
LABELS = ['BEFORE', 'AFTER', 'EQUAL', 'VAGUE']

data_paths = {
    'train': 'data_handling/split_data/train.csv',
    'test': 'data_handling/split_data/test.csv'}

tokenizer = BertTokenizerFast.from_pretrained(MODEL_CHECKPOINT)
tokenizer.add_tokens(['[א1]', '[/א1]', '[א2]', '[/א2]'])
E1_start = tokenizer.convert_tokens_to_ids('[א1]')
E2_start = tokenizer.convert_tokens_to_ids('[א2]')

train_set = TRCDataset(data_path=data_paths['train'])
test_set = TRCDataset(data_path=data_paths['test'])

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)

print(f'train: {len(train_set)}\ntest: {len(test_set)}')

model = TRCModel(output_size=len(LABELS), tokenizer=tokenizer, check_point=MODEL_CHECKPOINT, architecture='ESS')

trainer = Trainer(model, tokenizer=tokenizer,
                  optimizer=optim.Adam(get_parameters(model.named_parameters(), TRAINING_LAYERS), lr=1e-5),
                  criterion=nn.CrossEntropyLoss(),
                  entity_markers=(E1_start, E2_start),
                  labels=LABELS,
                  device=device)

trainer.train(train_loader=train_loader,
              valid_loader=test_loader,
              max_epochs=10)
