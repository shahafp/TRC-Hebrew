import pandas as pd


def get_label(row):
    shir = row['shir']
    hadar = row['hadar']
    curation = row['curated']
    if shir == hadar and not curation:
        return shir
    elif shir and not curation:
        return shir
    elif hadar and not curation:
        return hadar
    else:
        return curation


def add_curation_to_row(row, curated_data):
    match_row = curated_data[curated_data['id'] == row['id']]
    if match_row.empty:
        return None
    return match_row['label'].values[0]


round_1 = pd.read_csv('/Users/guy.yanko/Master/TRC-Hebrew/INCEPTION/inception_round_1/round1_data.csv')
round_2 = pd.read_csv('/Users/guy.yanko/Master/TRC-Hebrew/INCEPTION/inception_round_1/round1_data.csv')
round_3 = pd.read_csv('/Users/guy.yanko/Master/TRC-Hebrew/INCEPTION/inception_round_3/round3_data.csv')

round_2['curated'] = round_2.apply(lambda row: add_curation_to_row(row, round_3), axis=1)
round_2['shir'] = round_2['shir'].apply(lambda label: label if label in ['BEFORE', 'AFTER', 'EQUAL', 'VAGUE'] else None)
round_2['hadar'] = round_2['hadar'].apply(
    lambda label: label if label in ['BEFORE', 'AFTER', 'EQUAL', 'VAGUE'] else None)

round_2['label'] = round_2.apply(lambda row: get_label(row), axis=1)
# round_2.drop(index=round_2['label'].tolist().index(None), inplace=True)
data = round_2[~round_2['label'].isnull()]
data.to_csv('TRC_dataset.csv')
print()
