import pandas as pd


def update_taggers_entry(entry: pd.Series, id_2_taggers: dict):
    taggers = id_2_taggers.get(entry.id)
    if taggers:
        entry['shir'] = taggers[0]
        entry['hadar'] = taggers[1]
    return entry


round_1 = pd.read_csv('inception_round_1/multi_tagger_round_1.csv')
round_2 = pd.read_csv('inception_round_2/multi_tagger_round_2.csv')

round_2_dict = {r['id']: (r['shir'], r['hadar']) for _, r in round_2.iterrows()}
round_1_dict = {r['id']: (r['shir'], r['hadar']) for _, r in round_1.iterrows()}

round_1_dict.update(round_2_dict)
merge_df = round_1.apply(lambda entry: update_taggers_entry(entry, round_1_dict), axis=1)
merge_df.to_csv('multi_round_1_2.csv', index=False)
