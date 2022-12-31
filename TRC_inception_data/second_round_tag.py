import pandas as pd
import shutil

data = pd.read_csv('all_data.csv')

data = data[['text','shir']]
data = data.rename({'shir':'label'},axis='columns')
data = data[(data['label']!='NONE') & ~(data['label'].isna())]
data.to_csv('hw_trc_dataset.csv')
print()

# disagreement = data[
#     (~data['shir'].isna()) & (~data['hadar'].isna()) & (data['shir'] != data['hadar']) & (data['shir'] != 'NONE') & (
#             data['hadar'] != 'NONE')]
#
# agreements = data[data['shir'] == data['hadar']]
#
# sec_round_df  = pd.concat([disagreement,agreements.sample(frac=0.12)])
# sec_round_ids = sec_round_df['id'].tolist()
#
# for id in sec_round_ids:
#     doc, win, pair = id.split('.')
#     file_name = f'doc_{doc}_win_{win}_{pair}.txt'
#     file_path = f'source/{file_name}'
#     shutil.copy(file_path,'second_round')
