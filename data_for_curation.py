import shutil
import pandas as pd

data = pd.read_csv('multi_round_1_2.csv')

disagreement = data[data['shir'] != data['hadar']]

for id in disagreement['id'].tolist():
    doc, win, pair = id.split('.')
    file_name = f'doc_{doc}_win_{win}_{pair}.txt'
    file_path = f'inception_round_2/source/{file_name}'
    shutil.copy(file_path, 'data_for_curation')
