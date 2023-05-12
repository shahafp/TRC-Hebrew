import pandas as pd
import re

['[א1]', '[/א1]', '[א2]', '[/א2]']


def replace_tokens(text):
    new_text = text
    new_text = re.sub(r'\[א2\]|\[א1\]', '<', new_text)
    new_text = re.sub(r'\[/א2\]|\[/א1\]', '>', new_text)
    return new_text


def create_new_markers_data(file_name, dir):
    data = pd.read_csv(f"{dir}/{file_name}")
    data['text'] = data['text'].apply(lambda t:replace_tokens(t))
    data.to_csv(f'new_markers_data/{file_name}')

if __name__ == '__main__':
    create_new_markers_data('train.csv','clean_data')
    create_new_markers_data('test.csv','clean_data')

# all_data = pd.concat([train_data,test_data],axis=0)
# all_texts = all_data['text'].tolist()
# for text in all_texts:
#     if '<' in text or '>' in text:
#         print('token')
#
#
