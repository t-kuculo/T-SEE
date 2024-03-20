import json
import csv
from collections import Counter
import ast

def load_data(part="train", source="wikidata"):
    event_counter = Counter()
    data, t_samples, u_samples = [], [], []
    event_classes = set()

    part_map = {"dev": "val"}
    part = part_map.get(part, part)
    prefix = "wde" if source == "wikidata" else "dbpe"

    with open(f"../data/training/re/{prefix}_eq_{part}.json") as file1, open(f"../data/training/t2e/{prefix}_eq_{part}.json") as file2:
        for line1, line2 in zip(file1, file2):
            t_samples.append(json.loads(line1))
            u_samples.append(json.loads(line2))

    for t_sample, u_sample in zip(t_samples, u_samples):
        event_types = []
        for i, _ in enumerate(t_sample["sentences"]):
            for event in t_sample["events"][i]:
                _, _, event_type = event[0]
                event_counter.update({event_type: 1})
                event_classes.add(event_type)
                event_types.append(event_type)
        data.append({"label": event_types, "text": u_sample["text"]})

    return data, list(event_classes), event_counter.most_common(1000)

def read_schema(source="wikidata"):
    schema_file = "filtered_wikidata_event2.schema" if source == "wikidata" else "filtered_dbpedia_event2.schema"
    with open(f"../processing/{schema_file}") as file:
        lines = [line.rstrip() for line in file]
    return ast.literal_eval(lines[0]), ast.literal_eval(lines[1]), ast.literal_eval(lines[2])

def write_csv(part, data, event_classes, source="wikidata"):
    prefix = "wde" if source == "wikidata" else "dbpe"
    with open(f"../data/training/mlc_data/{prefix}_multilabel_{part}.csv", "w") as csvfile:
        fieldnames = ["id", "text"] + event_classes
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i, entry in enumerate(data):
            row = {"id": i, "text": entry["text"], **{event: int(event in entry["label"]) for event in event_classes}}
            writer.writerow(row)

def main(source="wikidata"):
    all_event_classes = []

    for part in ["train", "test", "dev"]:
        data, part_event_classes, _ = load_data(part, source)
        all_event_classes.append(part_event_classes)
        write_csv(part, data, part_event_classes, source)


if __name__ == "__main__":
    for source in ["wikidata", "dbpedia"]:
        main(source)

