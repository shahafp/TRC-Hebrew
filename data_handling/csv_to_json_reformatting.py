import json
from collections import defaultdict

import pandas as pd
import re


def validate_reformatting(reformatted_data):
    failed_docs = defaultdict(list)
    for document in reformatted_data['documents']:
        for event in document['events']:
            try:
                assert event['text'] == document['text'][event['start']:event['end'] + 1]
            except AssertionError:
                failed_docs[document['document_id']].append((event, document['text'][event['start']:event['end'] + 1]))

    return failed_docs, sum([len(doc['temporal_relations']) for doc in reformatted_data['documents']])


def get_event_id_from_relation_event(rel_event, document_events):
    for event in document_events:
        if event['start'] == rel_event['start'] and event['end'] == rel_event['end'] and event['text'] == rel_event[
            'text']:
            return event['event_id']
    raise Exception('no events match')


def printLCSSubStr(X: str, Y: str,
                   m: int, n: int):
    # Create a table to store lengths of
    # longest common suffixes of substrings.
    # Note that LCSuff[i][j] contains length
    # of longest common suffix of X[0..i-1] and
    # Y[0..j-1]. The first row and first
    # column entries have no logical meaning,
    # they are used only for simplicity of program
    LCSuff = [[0 for i in range(n + 1)]
              for j in range(m + 1)]

    # To store length of the
    # longest common substring
    length = 0

    # To store the index of the cell
    # which contains the maximum value.
    # This cell's index helps in building
    # up the longest common substring
    # from right to left.
    row, col = 0, 0

    # Following steps build LCSuff[m+1][n+1]
    # in bottom up fashion.
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                LCSuff[i][j] = 0
            elif X[i - 1] == Y[j - 1]:
                LCSuff[i][j] = LCSuff[i - 1][j - 1] + 1
                if length < LCSuff[i][j]:
                    length = LCSuff[i][j]
                    row = i
                    col = j
            else:
                LCSuff[i][j] = 0

    # if true, then no common substring exists
    if length == 0:
        return ''

    # allocate space for the longest
    # common substring
    resultStr = ['0'] * length

    # traverse up diagonally form the
    # (row, col) cell until LCSuff[row][col] != 0
    while LCSuff[row][col] != 0:
        length -= 1
        resultStr[length] = X[row - 1]  # or Y[col-1]

        # move diagonally up to previous cell
        row -= 1
        col -= 1

    # required longest common substring
    return ''.join(resultStr)


def clean_markers(text):
    text = re.sub(r'\[א1\] | \[/א1\]', '', text)
    text = re.sub(r'\[א2\] | \[/א2\]', '', text)
    return text


def create_pair_dict(text):
    match_1 = re.search(r'(\[א1\] )(.*)( \[/א1\])', text)
    start_1, raw_end_1 = match_1.span()
    end_1 = raw_end_1 - 12
    event_1 = match_1.groups()[1]
    text = re.sub(r'\[א1\] | \[/א1\]', '', text)
    e_1 = {"start": start_1,
           "end": end_1,
           "text": event_1}

    match_2 = re.search(r'(\[א2\] )(.*)( \[/א2\])', text)
    start_2, raw_end_2 = match_2.span()
    end_2 = raw_end_2 - 12
    event_2 = match_2.groups()[1]
    text = re.sub(r'\[א2\] | \[/א2\]', '', text)
    e_2 = {"start": start_2,
           "end": end_2,
           "text": event_2}
    return e_1, e_2


#
# X = "OldSite:GeeksforGeeks.org"
# Y = "NewSite:GeeksQuiz.com"
# m = len(X)
# n = len(Y)
#
# printLCSSubStr(X, Y, m, n)

data = pd.read_csv('../TRC_dataset.csv')
# data = pd.read_csv('../hw_trc_dataset.csv')
# docs_df = [df for _, df in data.groupby('document_number')]
reformatted_data = {'documents': []}
count = 0
for _, doc_df in data.groupby('document_number'):
    doc_df.sort_values(['window_number', 'pair_number'], inplace=True)
    print(f'{count}')
    doc_text = ''
    document_events = []
    document_relations = []
    prev_window = ''
    for _, win_df in doc_df.groupby('window_number'):
        window_text = clean_markers(win_df.iloc[0]['text'])
        overlap = printLCSSubStr(prev_window, window_text, len(prev_window), len(window_text))
        if len(overlap) <= 17 or overlap == ' להתערב בקביעת גובה העמלות' or overlap == 'התנאים הניתנים לחוסכים ותיקים בלבד", ':
            # print(overlap)
            overlap = ''
        try:
            doc_text_no_overlap = re.sub(overlap, '', doc_text)
        except:
            doc_text_no_overlap = re.sub(re.escape(overlap), '', doc_text)

        for _, pair_row in win_df.iterrows():
            pair_full_text = doc_text_no_overlap + pair_row['text']
            e_1, e_2 = create_pair_dict(pair_full_text)
            document_relations.append({'event_id_1': e_1,
                                       'event_id_2': e_2,
                                       'label': pair_row['label']})
            if e_1 not in document_events:
                document_events.append(e_1)
            if e_2 not in document_events:
                document_events.append(e_2)
        prev_window = window_text
        doc_text = doc_text_no_overlap + window_text

    document_events.sort(key=lambda d: d['start'])
    for i, event in enumerate(document_events):
        event['event_id'] = i

    for relation in document_relations:
        e_id_1 = get_event_id_from_relation_event(relation['event_id_1'], document_events)
        e_id_2 = get_event_id_from_relation_event(relation['event_id_2'], document_events)
        relation['event_id_1'] = e_id_1
        relation['event_id_2'] = e_id_2

    document = {'document_id': pair_row['document_number'],
                'text': doc_text,
                'events': document_events,
                'temporal_relations': document_relations}
    reformatted_data['documents'].append(document)
    count += 1

failed_docs, relations_count = validate_reformatting(reformatted_data)
with open('TRC_data.json', 'w') as fp:
    json.dump(reformatted_data,fp)
print('done!')
