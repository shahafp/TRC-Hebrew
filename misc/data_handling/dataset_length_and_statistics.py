import json

with open('TRC_data.json', 'r') as file:
    trc = json.load(file)

total_words = 0

for doc in trc['documents']:
    text = doc['text']
    total_words += len(text.split())

print()
