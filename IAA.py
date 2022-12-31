import pandas as pd
from sklearn.metrics import cohen_kappa_score


def clac_IAA(data_path, ignore_vague=False):
    data = pd.read_csv(data_path)
    id_2_labels = {r['id']: (r['shir'], r['hadar']) for i, r in data.iterrows()}

    t_1 = []
    t_2 = []
    for s, h in id_2_labels.values():
        if s in ['VAGUE', 'BEFORE', 'AFTER', 'EQUAL'] and h in ['VAGUE', 'BEFORE', 'AFTER', 'EQUAL']:
            if ignore_vague:
                if s == 'VAGUE' and h != s:
                    s = h
                elif h == 'VAGUE' and s != h:
                    h = s
            t_1.append(s)
            t_2.append(h)

    iaa = cohen_kappa_score(t_1, t_2)
    return iaa


if __name__ == '__main__':
    round_1_kappa = clac_IAA('INCEPTION/inception_round_1/multi_tagger_round_1.csv', ignore_vague=False)
    round_1_kappa_ignore_vague = clac_IAA('INCEPTION/inception_round_1/multi_tagger_round_1.csv', ignore_vague=True)

    round_2_kappa = clac_IAA('INCEPTION/inception_round_2/multi_tagger_round_2.csv', ignore_vague=False)
    round_2_kappa_ignore_vague = clac_IAA('INCEPTION/inception_round_2/multi_tagger_round_2.csv', ignore_vague=True)

    round_1_2_kappa = clac_IAA('multi_round_1_2.csv', ignore_vague=False)
    round_1_2_kappa_ignore_vague = clac_IAA('multi_round_1_2.csv', ignore_vague=True)


    print()
