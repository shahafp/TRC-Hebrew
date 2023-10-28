import re

from pydantic import BaseModel
from itertools import combinations

from collections import defaultdict


class Event(BaseModel):
    start: int
    end: int
    word: str
    row: int


def map_events_to_lines(events: list[Event]):
    event_to_line = defaultdict(list)
    for event in events:
        event_to_line[event.row].append(event)
    return event_to_line


def pad_events_in_line(lines: list[str], event_to_line: dict[str, list[Event]]) -> list[str]:
    padded_lines = []
    last_span = Event(start=0, end=0, word='', row=-1)
    for i, line in enumerate(lines):
        pad_line = ''
        for event in event_to_line[i]:
            pad_line += line[last_span.end:event.start] + '[EVENT] ' + line[event.start:event.end] + ' [/EVENT]'
            last_span = event
        if last_span.end < len(line):
            pad_line += line[last_span.end:]
        padded_lines.append(pad_line)
    return padded_lines


def create_ontology_data(lines: list[str], events: list[Event]) -> str:
    # Step 1: Map all the actions to their lines
    event_to_line = map_events_to_lines(events)
    # Step 2: Pad each event with special sign
    res = pad_events_in_line(lines, event_to_line)
    # Step 3: Concat all the line to one long text
    ontology_text = '\n'.join(res)
    return ontology_text


def split_to_windows(text: str) -> list[list]:
    # Split the text by periods to get a list of rows
    rows = text.split('.')

    # Initialize a list to store the windows
    windows = []

    # Create windows of size 2
    for i in range(len(rows) - 1):
        window = [rows[i], rows[i + 1]]
        if window[-1]:
            windows.append(window)

    return windows


def replace_event_tag_to_hebrew_tag(event_tuple, event_window) -> str:
    cur_window = '\n'.join(event_window)
    for i, event in enumerate(event_tuple):
        cur_window = cur_window.replace(f'[EVENT]{event.word}[/EVENT]', f'[e{i + 1}]{event.word}[/e{i + 1}]')
    # Replace the event tag with the Hebrew tag
    cur_window = cur_window.replace('[EVENT]', '').replace('[/EVENT]', '')
    cur_window = re.sub(r'\s+', ' ', cur_window)
    cur_window = cur_window.replace('[e1]', '[א1]').replace('[/e1]', '[/א1]')
    cur_window = cur_window.replace('[e2]', '[א2]').replace('[/e2]', '[/א2]')
    return cur_window


def get_all_events_in_window(event_windows):
    events_in_window = {}
    for i, window in enumerate(event_windows):
        events = re.finditer(r'\[EVENT\]([\s\S]*?)\[/EVENT\]', '\n'.join(window))
        events = [Event(start=event.start(), end=event.end(), word=event.group(1), row=i) for event in events]
        permutation_events = list(combinations(events, 2))
        events_in_window[i] = permutation_events
    return events_in_window


def create_windows_for_tagging(events_to_windows: dict, event_windows: list):
    windows_for_tagging = []
    for i, window in enumerate(event_windows):
        for event_tuple in events_to_windows[i]:
            windows_for_tagging.append(replace_event_tag_to_hebrew_tag(event_tuple, window))
    return windows_for_tagging


if __name__ == '__main__':
    short_text = """Once upon a time in a peaceful village, there lived a kind-hearted blacksmith.
He crafted exquisite pieces of jewelry and shared them with the villagers.
One day, a mysterious traveler arrived and requested a special pendant.
The blacksmith worked tirelessly to create the most beautiful pendant he had ever made.
When he presented it to the traveler, the traveler smiled with gratitude and thanked the blacksmith before departing into the unknown.
The village celebrated the blacksmith's talent and wished for his happiness.
And so, the blacksmith continued to bless the village with his creations for years to come."""
    lines = short_text.split('\n')

    # Define the events with their corresponding lines and spans
    event_data = [
        ("Once", 0, 0), ("lived", 0, 3), ("crafted", 1, 3),
        ("shared", 1, 9), ("arrived", 2, 4), ("worked", 3, 3),
        ("create", 3, 7), ("made", 3, 12), ("presented", 4, 3),
        ("smiled", 4, 6), ("thanked", 4, 12), ("departing", 4, 4),
        ("celebrated", 5, 4), ("wished", 5, 7), ("continued", 6, 5),
        ("bless", 6, 10)
    ]

    events = []

    for word, line_num, span_start in event_data:
        start = lines[line_num].find(word, span_start)
        end = start + len(word)
        event = Event(word=word, start=start, end=end, row=line_num)
        events.append(event)

    # Print the list of events with line spans
    for event in events:
        print(f"Word: {event.word}, Start: {event.start}, End: {event.end}, Line: {event.row}")

    ontology_text = create_ontology_data(lines, events)
    original_text = '\n'.join(lines)

    text_windows = split_to_windows(short_text)
    ontology_text_windows = split_to_windows(ontology_text)
    events_in_windows = get_all_events_in_window(ontology_text_windows)
    windows_for_taggers = create_windows_for_tagging(events_in_windows, ontology_text_windows)

    # Get unique values from a patient id column
    unique_values = df['id'].unique()
