import shutil

import pandas as pd

data = pd.read_csv('round2_dis.csv')

disagreement = data[
    (~data['shir'].isna()) & (~data['hadar'].isna()) & (data['shir'] != data['hadar']) & (data['shir'] != 'NONE') & (
            data['hadar'] != 'NONE')]

sec_round_df  = disagreement
sec_round_ids = sec_round_df['id'].tolist()

for id in sec_round_ids:
    doc, win, pair = id.split('.')
    file_name = f'doc_{doc}_win_{win}_{pair}.txt'
    file_path = f'source/{file_name}'
    shutil.copy(file_path,'curation_data')
