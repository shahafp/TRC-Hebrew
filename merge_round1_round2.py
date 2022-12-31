import pandas as pd

dis = pd.read_csv('dis.csv')
round_1 = pd.read_csv('/Users/guy.yanko/Master/TRC-Hebrew/TRC_inception_data/all_data.csv')
round_2 = pd.read_csv('/inception_round_2/round2_data.csv')
replace_ids = round_2['id'].tolist()
round_2=round_2.loc[round_2.id.isin(dis.id)]
# round_2.to_csv('round2_dis.csv')
round_2 = {r['id']: (r['shir'],r['hadar']) for i,r in round_2.iterrows()}
round_1 = {r['id']: (r['shir'],r['hadar']) for i,r in round_1.iterrows()}

all_d=round_1.update(round_2)
# round_1.update(round_2)
# shir = round_1['shir'].tolist()
# hadar = round_1['hadar'].tolist()

t_1=[]
t_2 = []
for s,h in round_1.values():
    if s in ['VAGUE','BEFORE','AFTER','EQUAL'] and h in ['VAGUE','BEFORE','AFTER','EQUAL']:
        t_1.append(s)
        t_2.append(h)

import sklearn.metrics

res = sklearn.metrics.cohen_kappa_score(t_1, t_2)


t_1=[]
t_2 = []
for s,h in round_1.values():
    if s in ['VAGUE','BEFORE','AFTER','EQUAL'] and h in ['VAGUE','BEFORE','AFTER','EQUAL']:
        if s=='VAGUE' and h!=s:
            s=h
        elif h=='VAGUE' and s!=h:
            h=s
        t_1.append(s)
        t_2.append(h)


res_no_vague = sklearn.metrics.cohen_kappa_score(t_1, t_2)

print()


# for id in replace_ids:
#     round_1.iloc[round_1.id==id, 'shir']
#     round_1.at[round_1['id']==id] = round_2[round_2['id']==id]
#     print()
#
# round_1.loc[round_1.id.isin(round_2.id), ['shir', 'hadar']] = round_2[['hadar','shir']]

print()