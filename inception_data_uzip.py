import os
import json
import glob
from collections import defaultdict
import pandas as pd
import numpy as np


def create_doc_id(doc_name):
    doc_name = doc_name.replace('.txt', '')
    doc_name_parts = doc_name.split('_')
    doc_number, window, pair = doc_name_parts[1], doc_name_parts[3], doc_name_parts[4]
    return f'{doc_number}.{window}.{pair}'


def handle_json_file(doc, doc_id, dataset_dict, annotator_name):
    with open(doc, 'r') as f:
        data = json.load(f)

    text = None
    label = None
    for element in data['%FEATURE_STRUCTURES']:
        if not text:
            text = element.get('sofaString')
        if not label:
            label = element.get('Label')
    doc_num, window_num, pair_num = doc_id.split('.')
    dataset_dict['id'].append(doc_id)
    dataset_dict['document_number'].append(doc_num)
    dataset_dict['window_number'].append(window_num)
    dataset_dict['pair_number'].append(pair_num)
    dataset_dict['text'].append(text)
    dataset_dict[annotator_name].append(label)
    return dataset_dict


def json_to_df(*docs):
    dataset_dict = defaultdict(list)
    for doc in docs:
        # Get doc name
        annotator_name = doc.split('/')[-1].replace('.json', '')
        doc_name = doc.split('/')[-2]
        doc_id = create_doc_id(doc_name)
        current = handle_json_file(doc, doc_id, defaultdict(list), annotator_name)
        dataset_dict.update(current)

    return pd.DataFrame(dataset_dict)


def inception_to_csv(csv_file_path, inception_annotations_dir):
    unzip_files = inception_annotations_dir
    all_zip_files_for_all_docs = []

    for txt in os.listdir(unzip_files):
        for f in os.listdir(unzip_files + '/' + txt):
            all_zip_files_for_all_docs.append(unzip_files + '/' + txt + '/' + f)

    all_unzip_file_dir = set([os.path.dirname(file_) for file_ in all_zip_files_for_all_docs])
    all_data = []
    for i, file_ in enumerate(all_unzip_file_dir, 1):
        print(f'doc {i}/{len(all_unzip_file_dir)}')
        files_list = glob.glob(os.path.join(file_, '*.json'))
        annotation_df = json_to_df(*files_list)
        all_data.append(annotation_df)

    all_data_df = pd.concat(all_data)

    all_data_df = all_data_df.apply(lambda win: win if (win['shir'], win['hadar']) not in [('NONE', 'NONE'),
                                                                                           ('NONE', np.nan),
                                                                                           (np.nan, 'NONE'),
                                                                                           (np.nan, np.nan)] else None,
                                    axis=1).dropna(how='all')

    all_data_df.to_csv(csv_file_path, index=False)


def split_to_multi_single_taggers(data_path, multi_path, single_path):
    data = pd.read_csv(data_path)

    single_tagger = data[
        ((~data['shir'].isna()) & (data['hadar'].isna())) | ((data['shir'].isna()) & (~data['hadar'].isna()))]
    single_tagger = single_tagger.drop('hadar', axis=1).rename(columns={'shir': 'label'})
    single_tagger = single_tagger[single_tagger['label'] != 'NONE']
    single_tagger.to_csv(single_path, index=False)

    multiple_taggers = data[(~data['shir'].isna()) & (~data['hadar'].isna())]
    multiple_taggers = multiple_taggers[(multiple_taggers['shir']!='NONE') & (multiple_taggers['hadar']!='NONE')]
    multiple_taggers.to_csv(multi_path, index=False)


if __name__ == '__main__':
    # inception_to_csv('inception_round_2/round2_data.csv', 'inception_round_2/annotation')

    split_to_multi_single_taggers('INCEPTION/inception_round_2/round2_data.csv',
                                  'INCEPTION/inception_round_2/multi_tagger_round_2.csv',
                                  'INCEPTION/inception_round_2/single_tagger_round_2.csv')

