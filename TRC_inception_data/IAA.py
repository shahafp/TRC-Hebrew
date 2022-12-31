import numpy as np
import pandas as pd

data = pd.read_csv('all_data.csv')
# data = pd.read_csv('/Users/guy.yanko/Master/TRC-Hebrew/merge1.csv')


disagreement = data[
    (~data['shir'].isna()) & (~data['hadar'].isna()) & (data['shir'] != data['hadar']) & (data['shir'] != 'NONE') & (
            data['hadar'] != 'NONE')]
agreements = data[data['shir'] == data['hadar']]

multiple_taggers = data[
    (~data['shir'].isna()) & (~data['hadar'].isna()) & (data['shir'] != 'NONE') & (data['hadar'] != 'NONE')]

multiple_taggers_no_vague = multiple_taggers[~((multiple_taggers['shir'] != multiple_taggers['hadar']) & (
            (multiple_taggers['shir'] == 'VAGUE') | (multiple_taggers['hadar'] == 'VAGUE')))]
import sklearn.metrics

res = sklearn.metrics.cohen_kappa_score(multiple_taggers['shir'].tolist(), multiple_taggers['hadar'].tolist())
res_no_vague = sklearn.metrics.cohen_kappa_score(multiple_taggers_no_vague['shir'].tolist(), multiple_taggers_no_vague['hadar'].tolist())

shir_labels = [l for l in data['shir'].tolist() if l not in ['NONE',np.nan]]
from collections import Counter

counts = Counter(shir_labels)
total_counts = sum(list(counts.values()))
counts_prec = {k:v/total_counts for k,v in counts.items()}
print()
