import json
import copy
from collections import Counter
import os
all_keys = []
class_counter =  Counter()
data = []

def load_data(part = "train"):
    c = Counter()
    data = []
    t_samples = []
    u_samples = []
    event_classes = set()
    if part == "dev":
        t = "val"
    else:
        t = part
    with open("../data/training/re/wde_eq_"+part+".json", 'r') as json_file1, open("../data/training/t2e/wde_eq_"+t+".json", 'r') as json_file2:
        json_list1 = list(json_file1)
        json_list2 = list(json_file2)

    for json_str1, json_str2 in zip(json_list1, json_list2):
        t_samples.append(json.loads(json_str1))
        u_samples.append(json.loads(json_str2))
    
    xx = set(json_list1)
    yy = set(json_list2)
    zz = ([item for item, count in Counter(json_list2).items() if count > 1])
    for t_sample, u_sample in zip(t_samples, u_samples):
        event_types = []
        for i, sentence in enumerate(t_sample["sentences"]):
            for event in t_sample["events"][i]:
                token_start = event[0][0] 
                token_end = event[0][1] 
                event_type = event[0][2] 
                c.update({event_type:1})
                event_classes.update({event_type})
                event_types.append(event_type)
                class_counter.update({event_type})
        data.append({"label":event_types, "text": u_sample["text"]})

    return data, list(event_classes), class_counter.most_common(1000)

import pandas as pd   
import csv
import ast
import os
# read from schema all event types
with open("../data/training/t2e/wde_unlabelled_event.schema","r") as f:
    lines = [line.rstrip() for line in f]
event_classes = ast.literal_eval(lines[0])
event_properties = ast.literal_eval(lines[1])
event_class_properties = ast.literal_eval(lines[2])
a = []

#with open("all_class_labels.json") as f: #uncomment for wikidata
    #class_labels = json.load(f)
# comment for wikidata

for part in ["train","test", "dev"]:
    data, event_classes, cc = load_data(part)
    a.append(event_classes)
    with open("../data/training/mlc_data/wde_multilabel_"+part+".csv", "w") as csvfile:
        fieldnames = ["id","text"]+a[0]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i, entry in enumerate(data):
            x = dict()
            x["id"] = i
            x["text"] = entry["text"]
            for t in a[0]:
                if t not in entry["label"]:
                    x[t] = 0
                else:
                    x[t] = 1
            writer.writerow(x)

print(set(a[0])==set(a[1]))
print(set(a[0])==set(a[2]))
print(set(a[1])==set(a[2]))





















