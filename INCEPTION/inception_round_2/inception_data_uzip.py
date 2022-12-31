import os

unzip_files = 'annotation'
all_zip_files_for_all_docs = []
all_doc_zip_files = [
    [all_zip_files_for_all_docs.append(unzip_files + '/' + txt + '/' + f) for f in os.listdir(unzip_files + '/' + txt)]
    for txt in os.listdir(unzip_files)]

# DEFINITIONS
import json
import glob
from collections import defaultdict
import pandas as pd
import numpy as np

kappa_annotations = pd.DataFrame(columns=['id', 'window_number', 'pair_number', 'text', 'label', 'document_number'])
trc_dataset = pd.DataFrame(columns=['id', 'window_number', 'pair_number', 'text', 'label', 'document_number'])


def create_doc_id(doc_name):
    doc_name = doc_name.replace('.txt', '')
    doc_name_parts = doc_name.split('_')
    doc_number, window, pair = doc_name_parts[1], doc_name_parts[3], doc_name_parts[4]
    return f'{doc_number}.{window}.{pair}'


def handle_json_file(doc, doc_id, doc_name, dataset_dict, annotator_name):
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

    # if 'Event' in data['_views']['_InitialView']:
    #     event = data['_views']['_InitialView']['Event']
    #     if 'Label' in event[0]:
    #         label = event[0]['Label']
    #         #   if label == 'NONE':
    #         #     return dataset_dict
    #         doc_num, window_num, pair_num = doc_id.split('.')
    #         dataset_dict['id'].append(doc_id)
    #         dataset_dict['document_number'].append(doc_num)
    #         dataset_dict['window_number'].append(window_num)
    #         dataset_dict['pair_number'].append(pair_num)
    #         dataset_dict['text'].append(data['_referenced_fss']['1']['sofaString'])
    #         dataset_dict[annotator_name].append(label)


def json_to_df(*docs):
    dataset_dict = defaultdict(list)
    for doc in docs:
        # Get doc name
        annotator_name = doc.split('/')[-1].replace('.json', '')
        doc_name = doc.split('/')[-2]
        doc_id = create_doc_id(doc_name)
        current = handle_json_file(doc, doc_id, doc_name, defaultdict(list), annotator_name)
        dataset_dict.update(current)

    return pd.DataFrame(dataset_dict)


all_unzip_file_dir = set([os.path.dirname(file_) for file_ in all_zip_files_for_all_docs])
# all_unzip_file_dir = all_zip_files_for_all_docs
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

all_data_df.to_csv('round2_data.csv', index=False)
print()
