import os
import ast
import os
from classes import * 
import ndjson
from collections import Counter
import shelve
import time
import json
import datetime
import datefinder
import shelve
import re
import copy
import requests
import unicodedata
from sklearn.model_selection import train_test_split
import jsonlines
from dateutil import parser
from SPARQLWrapper import SPARQLWrapper, JSON

rx = r"\.(?=\S)"


import spacy  # version 3.0.6'

print(0)
nlp = spacy.load("en_core_web_md")
nlp.add_pipe("entityLinker")

print(1)
def remove_non_ascii(string):
    return ''.join(char for char in string if ord(char) < 256)


dbpedia_ids_to_wikidata_ids = shelve.open('../data/shelves/dbpedia_to_wikidata_en')
wikidata_ids_to_dbpedia_ids = shelve.open('../data/shelves/wikidata_to_dbpedia_en')
types_dbo = shelve.open('../data/shelves/types_dbo2')
types_wd = shelve.open('../data/shelves/types_wd')
instance_types = shelve.open("../data/shelves/instance_types")


with open("event_subevents.json","r") as f:
    events = json.load(f)


def first_stage_eval_data():
    directory = "../data/raw_data/"
    dataset = {}
    dygiepp_data = []
    i = 0
    k = 0
    mistakes = 0
    event_wd_not_in_events = 0
    d = []
    all = 0
    e = time.time()
    context_window = 0
    for u, filename in enumerate(os.listdir(directory)):
        if ".gitignore" in filename:
            continue
        print("%d out of %d completed"%(u, len(os.listdir(directory))))
        with open(directory+filename, "r") as f:
            print(filename)
            json_list = ndjson.load(f)
        for g,js in enumerate(json_list):
            all+=1
            Article_obj = Article(js)
            parent_event_link = ("_".join(Article_obj.article_name.split(" ")))
            try:
                parent_event_wd_id = parent_event_link
            except (KeyError):
                mistakes+=1
                continue
            if parent_event_wd_id not in events:
                event_wd_not_in_events+=1

            for input_paragraph in Article_obj.input_paragraphs:
                last_sentence_subevents = []
                for sentence in input_paragraph.sentences:
                    all_sentence_link_types = [link["types"] for link in sentence.links]
                    all_sentence_link_types = [link for links in all_sentence_link_types for link in links]
                    for link in sentence.links:
                        if "Event" in all_sentence_link_types:
                            if "Event" in link["types"]:
                                subevent_link = link["target"]
                                try:
                                    subevent_wd_id = subevent_link
                                except (KeyError):
                                    d.append(subevent_wd_id)
                                    continue
                                # if subevent doesnt have partof event
                                i+=1
                                if i%10000==0:
                                    print(i)
                                last_sentence_subevents.append((subevent_wd_id, subevent_link))
                                if parent_event_wd_id not in dataset:
                                    dataset[parent_event_wd_id] = {}
                                if subevent_wd_id +"@" + subevent_link not in dataset[parent_event_wd_id]:
                                    dataset[parent_event_wd_id][subevent_wd_id +"@" + subevent_link] = {"samples":[], "groundtruth":{}}
                                dataset[parent_event_wd_id][subevent_wd_id +"@" + subevent_link]["samples"].append((link, sentence.text, sentence.links))
                                dygiepp_data.append(sentence.text)
                        elif context_window:
                            context_window = 0
                            for subevent in last_sentence_subevents:
                                subevent_wd_id, subevent_link = subevent
                                link, text, links = dataset[parent_event_wd_id][subevent_wd_id +"@" + subevent_link]["samples"][-1]
                                text += sentence.text
                                links += sentence.links
                                dataset[parent_event_wd_id][subevent_wd_id +"@" + subevent_link]["samples"][-1] = (link, text, links)
                                dygiepp_data.append(sentence.text)
                            last_sentence_subevents = [] 
    print("all articles:", all)
    print("mistakes:", mistakes)
    print("print not in events:",event_wd_not_in_events)
    print("missed_subevents:", len(d))
    with open('dbp_eval_data_with_context_window.txt', 'w') as f:
        json.dump(dataset,f)
    
first_stage_eval_data()



with open('dbp_eval_data_with_context_window.txt', 'r') as f:
    dataset = json.load(f)
    
done_already = []
with open("dbp_full_event_list2.txt","w") as f:
    for parent_event in dataset:
        subevents = dataset[parent_event]
        for key in subevents.keys():
            x = key.split("@")[0]+"\n"
            if x not in done_already:
                done_already.append(x)
                f.write(x) 
            else:
                continue   
print("First stage done")

subevents_properties = {}
with open("dbp_all_subevent_properties.csv", "r") as f:
    lines = f.readlines()[1:]
    for row in lines:
        row = row.rstrip().split("\t")
        event, property, value = row[0], row[1], row[2]
        if event not in subevents_properties:
            subevents_properties[event] = {}
            
        if property not in subevents_properties[event]:
            subevents_properties[event][property] = []
        subevents_properties[event][property].append(value)

d = []
for event_wd_id in dataset:
    for subevent in dataset[event_wd_id]:
        number_of_sentences = len(dataset[event_wd_id][subevent]["samples"]) 
        try:
            subevent_wd_id, subevent_link = subevent.split("@")
        except ValueError:
            print(subevent)
        if subevent_wd_id not in subevents_properties:
            d.append(subevent_wd_id)
            continue
        subevent_properties = subevents_properties[subevent_wd_id]
        dataset[event_wd_id][subevent]["groundtruth"] = subevent_properties
print(len(d))

with open('dbp_all_data.txt', 'w') as f:
    json.dump(dataset,f)


with open('dbp_all_data.txt', 'r') as f:
    dataset = json.load(f)


new_dataset = {}
tpe = 0
no_type = 0
total_subevents = []
subevents_missed_because_no_prop_or_no_instance_type = []
d2 = []
subevent_types_removed_because_they_have_no_properties = []
subevents_removed_because_they_have_no_types_remaining = []
d5 = []
number_of_samples = []
number_of_participant_properties = 0
number_of_time_properties = 0
missed_subevents = 0
ggg = 0
total_failure = 0
all_event_types = set()
for num, event_wd_id in enumerate(dataset):
    print("%d out %d"%(num, len(dataset)))
    for subevent in dataset[event_wd_id]:
        total_subevents.append(subevent)
        try:
            subevent_wd_id, subevent_link = subevent.split("@")
        except ValueError:
            total_failure+=1
            continue
        if "World_War_II" in subevent_wd_id:
            print()
        if subevent_wd_id in types_dbo:
            try:
                subevent_types = types_dbo[subevent_wd_id]
            except(KeyError):
                d2.append(subevent_link)
        else:
            full_url = "<http://dbpedia.org/resource/"+subevent_wd_id+">"
            if full_url in types_dbo:
                try:
                    subevent_types = types_dbo[full_url]
                except(KeyError):
                    d2.append(subevent_link)
            else:
                subevents_missed_because_no_prop_or_no_instance_type.append(subevent_wd_id)
                continue
        subevent_types = list(subevent_types)
        groundtruth_properties = dataset[event_wd_id][subevent]["groundtruth"]
        tmp = subevent_types
        #ignore events which are types as "event"
        all_event_types.update(subevent_types)
        if "SocietalEvent" in subevent_types: #event
            subevent_types.remove("SocietalEvent") #event
        if "Event" in subevent_types:
            subevent_types.remove("Event") 
        if not subevent_types:
            no_type+=1
            continue
        groundtruth_properties = {k:v for k, v in groundtruth_properties.items()}
        groundtruth_values = {wd:k for k,v in groundtruth_properties.items() for wd in v}

        if groundtruth_values == {}:
            tpe+=1
            continue

        completed = set()
        for sample in dataset[event_wd_id][subevent]["samples"]:
            new_groundtruth = {}
            sample_subevent = sample[0]
            if subevent_wd_id not in subevents_properties:
                missed_subevents+=1
                continue
            subevent_class_type = subevent_types
            sample_subevent["types"] = subevent_class_type
            
            
            sample_context = re.sub(rx, ". ", sample[1])
            sample_context = sample_context.replace("-"," - ")
            sample_context = sample_context.replace("–"," – ")
            sample_context = sample_context.replace("  ", " ")
            # avoid short bulletpoints
            if "*" in sample_context and len(sample_context)<100:
                continue

            if len(sample_context)<30:
                continue
            sample_links = sample[2]
            try: 
                subevent_anchor = nlp(sample_subevent["anchor_text"])   
                doc = nlp(sample_context)
            except:
                print("missed one")
                continue
            total = []
            first = False
            subevent_list_of_words = [t.text for t in subevent_anchor]

            if sample_subevent["anchor_text"] in doc.text:
                start = -1
                end = -1
                found = False
                start_char = doc.text.index(sample_subevent["anchor_text"])
                end_char = start_char + len(sample_subevent["anchor_text"])
                sample_subevent["character_range"] = (start_char,end_char)
                for t in doc:
                    if t.text == sample_subevent["anchor_text"]:
                        start = t.i
                        end = t.i
                        sample_subevent["token_range"] = (start,end)
                        found = True
                        break
                    elif not first and t.text in sample_subevent["anchor_text"]:
                        start = t.i
                        end = t.i
                        total.append(t.text)
                        first = True
                    elif total != subevent_anchor.text.split() and t.text not in sample_subevent["anchor_text"]:
                        first = False
                        total = []
                    elif t.text in sample_subevent["anchor_text"]:
                        end = t.i
                        total.append(t.text)
                    if total==subevent_list_of_words:
                        found = True
                        end = t.i
                        sample_subevent["token_range"] = (start,end)
                        break
            else:
                continue
            if not found:
                continue


            times = datefinder.find_dates(sample_context,base_date=datetime.datetime(9999, 1, 1, 0, 0), source="True")
            times = [(time_expression[0].isoformat(),time_expression[1])  for time_expression in times]
            new_times = []
            for tm in times:
                if  "." in tm[1]:
                    new_time = (tm[0],tm[1].split(".")[0])
                    new_times.append(new_time)
                else:
                    new_times.append(tm)
            if new_times:
                times = new_times
            if sample_subevent["anchor_text"] not in sample_context:
                    continue
            for ttime in times:
                if ttime[0] not in groundtruth_values:
                    continue
                else:
                    property = groundtruth_values[ttime[0]]
                    first = 1
                    # if the time is exactly accurate, store it and remove from groundtruth_values
                    for t in doc:
                        if first and t.text in ttime[1].split():
                            start = t.i
                            end = t.i
                            start_char = sample_context.index(t.text)
                            end_char = sample_context.index(t.text)+len(t.text)
                            first = 0
                        elif t.text in ttime[1].split(): # .split() so that a piece of word is not found in a bigger word
                            end = t.i
                            end_char = sample_context.index(t.text)+len(t.text)
                        elif not first:
                            break
                    if property not in new_groundtruth:
                        new_groundtruth[property] = []
                    new_groundtruth[property].append([start, end, start_char, end_char, property, ttime[1]]) # it used to be just ttime
                    del groundtruth_values[ttime[0]]
            
            updated_times =[]
            for t in times:
                if t:
                    tmp1 = t[0].split("-")
                    tmp2 = [i.split(":") for i in tmp1]
                    tmp3 = [i for j in tmp2 for i in j]
                    updated_times.append((tmp3,t[1]))
            # If a year,month../data correct sequence was found but the rest doesn't match, store the sequence as happening on the 1st
            ground_base = ["9999","01","01T00","00","00Z"]
            for property_value in groundtruth_values:
                if not updated_times:
                    break
                ground_time = []

                if isinstance(property_value, str) and len(property_value.split("-")) == 3 and property_value[-1]=="Z" and property_value[-3:-1].isdigit():
                    tmp1 = property_value.split("-")
                    tmp2 = [i.split(":") for i in tmp1]
                    arranged_property_value = [i for j in tmp2 for i in j]

                    found = False
                    for updated_time in updated_times:
                        start = -1
                        end = -1
                        # if the year is not correct just skip
                        if updated_time[0][0]!=arranged_property_value[0]:
                            continue
                        if found:
                            break
                        else:
                            ground_time.append(arranged_property_value[0])
                            if updated_time[0][1] == arranged_property_value[1]:
                                ground_time.append(arranged_property_value[1])
                            else:
                                ground_time.append(ground_base[1])
                            if updated_time[0][2] == arranged_property_value[2]:
                                ground_time.append(arranged_property_value[2])
                            else:
                                ground_time.append(ground_base[2])
                            if updated_time[0][3] == arranged_property_value[3]:
                                ground_time.append(arranged_property_value[3])
                            else:
                                ground_time.append(ground_base[3])
                            if updated_time[0][4] == arranged_property_value[4]:
                                ground_time.append(arranged_property_value[4])
                            else:
                                ground_time.append(ground_base[4])
                            if updated_time[1] not in sample_context:
                                ground_time = []
                                continue
                            start_char = sample_context.index(updated_time[1])
                            end_char = start_char+len(updated_time[1])
                            sc = 0
                            for t in doc:
                                if t.text in updated_time[1]:
                                    sc = sample_context.index(t.text, sc)
                                    if sc<=start_char and sc+len(t.text)>=start_char:
                                        start = t.i
                                    if sc<=end_char and sc+len(t.text)>=end_char:
                                        end = t.i
                                    if start>=0 and end>=0:
                                        break
                            if any(ground_time) and start>=0 and end>=0:
                                found = True
                                property = groundtruth_values[property_value]
                                if property not in new_groundtruth:
                                    new_groundtruth[property] = []
                                a,b,c,d,e = ground_time[0],ground_time[1],ground_time[2],ground_time[3],ground_time[4]
                                a,b,c,d,e = int(a),int(b), int(c[:2]), int(d), int(e[:-1])
                                ground_time = datetime.datetime(a,b,c,d,e)
                                if ground_time not in new_groundtruth[property]:
                                    new_groundtruth[property].append([start, end, start_char, end_char, property, updated_time[1]]) 
                                ground_time = []
                                
            for ent in doc._.linkedEntities:
                qid = "Q"+str(ent.identifier)
                try:
                    dbp_id = wikidata_ids_to_dbpedia_ids[qid]
                except:
                    continue
                if dbp_id in groundtruth_values:
                    property = groundtruth_values[dbp_id]
                    if property not in new_groundtruth:
                        new_groundtruth[property] = []
                    new_groundtruth[property].append([ent.span.start, ent.span.end-1, ent.span.start_char, ent.span.end_char, property, ent.span.text, dbp_id])

            # once we have gone through entities, we want to look for literals: quantities and time expressions. 
            count_sum = 0
            
            # skip if there are no integer values in property values
            start = -1
            if [val for val in list(groundtruth_values.keys()) if val.isdigit()]:
                # check each digit in the context if it fits any of the property values. Also add up numbers in case the property value is the sum of integers
                for i in doc:
                    txt = i.text
                    formatting_i = txt.replace(",","")
                    formatting_i = formatting_i.replace(".","")
                    if txt.isdigit():
                        if start == -1:
                            start = i.i
                        end = i.i
                        start_char = sample_context.index(txt)
                        end_char = sample_context.index(txt)+len(txt)
                        try:
                            count_sum+=int(txt)
                        except:
                            print(txt)
                    elif formatting_i.isdigit():
                        count_sum+=int(formatting_i)
                        if start == -1:
                            start = i.i
                        end = i.i
                        start_char = sample_context.index(txt)
                        end_char = sample_context.index(txt)+len(txt)
                    if  formatting_i in groundtruth_values:
                        property = groundtruth_values[formatting_i]
                        if property not in new_groundtruth:
                            new_groundtruth[property] = []
                        new_groundtruth[property].append([i.i, i.i, sample_context.index(txt), sample_context.index(txt)+len(txt), property, txt])
                if str(count_sum) in groundtruth_values:
                    if property not in new_groundtruth:
                        new_groundtruth[property] = []
                    new_groundtruth[property].append([start, end, start_char, end_char, property, doc.text[start_char:end_char]]) # count_sum
                

            if event_wd_id not in new_dataset:
                new_dataset[event_wd_id] = {}
            if subevent not in new_dataset[event_wd_id]:
                new_dataset[event_wd_id][subevent] = []
            if new_groundtruth:
                new_sample = {"sample_event_link":sample_subevent, "untokenized_context":sample_context,"tokenized_context":[t.text for t in doc], "sample_groundtruth": new_groundtruth, "ground_event_types": subevent_types}
                new_dataset[event_wd_id][subevent].append(new_sample)
    if num%100==0:
        with open("the_data/dbp_data"+str(num)+".json","w") as f:
            json.dump(new_dataset,f)
        new_dataset = {}

with open("the_data/dbp_data"+str(num)+".json","w") as f:
    json.dump(new_dataset,f)

dataset = {}

for subdir, dirs, files in os.walk("the_data"):
    for file in files:
        filepath = subdir + os.sep + file
        print(filepath)
        with open("the_data/"+file, "r") as f:
            dataset.update(json.load(f))

with open("the_data/dbp__full_data.json","w") as f:
    json.dump(dataset, f)


event_class_counter = Counter()


with open("the_data/dbp__full_data.json","r") as f:
    dataset = json.load(dataset)

tmp_dataset = copy.deepcopy(dataset)
for key in tmp_dataset:
    for subkey in tmp_dataset[key]:
        for j, sample in enumerate(tmp_dataset[key][subkey]):
            for i, prop in enumerate(sample["sample_groundtruth"].keys()):
                new_prop_label = prop.replace("http://dbpedia.org/ontology/","").replace(">","").replace("<","")
                dataset[key][subkey][j]["sample_groundtruth"][new_prop_label] = dataset[key][subkey][j]["sample_groundtruth"][prop]
                del dataset[key][subkey][j]["sample_groundtruth"][prop]
            for i, event_type in enumerate(sample["ground_event_types"]):
                    dataset[key][subkey][j]["ground_event_types"][i] = event_type.replace("<http://dbpedia.org/ontology/","").replace(">","")
            for i, event_type in enumerate(sample["sample_event_link"]["types"]):
                    dataset[key][subkey][j]["sample_event_link"]["types"][i] = event_type.replace("<http://dbpedia.org/ontology/","").replace(">","")

country_specific_event_classes = ['Q7864918', 'Q17496410', 'Q1128324', 'Q22333900', 'Q319496', 'Q849095', 'Q1141795',\
 'Q24333627', 'Q9102', 'Q24397514', 'Q56185179', 'Q17317594', 'Q22162827', 'Q123577', 'Q22284407', 'Q15283424', 'Q890055', 'Q25080094', 'Q47566',\
  'Q107540719', 'Q7870', 'Q248952', 'Q22696407', 'Q7961', 'Q9208', 'Q60874', 'Q8036']
bad_event_classes = ['Q1944346','Q2425489', 'Q2705481', 'Q148578', 'Q854845', 'Q15893266', 'Q928667', 'Q763288', 'Q1378139', 'Q26540', 'Q149918', 'Q7217761', 'Q27949697','Q27949697']
bad_properties = ['P4342', 'P9307', 'P9346', 'P935', 'P373', 'P2002']



clean_dataset = {}
property_counter = Counter()

for key in dataset:
    t = dataset[key]
    if not isinstance(dataset[key], dict):
        continue
    for subkey in dataset[key]:
        for sample in dataset[key][subkey]:
            if sample and "token_range" in sample["sample_event_link"]:
                if key not in clean_dataset:
                    clean_dataset[key] = {}
                if subkey not in clean_dataset[key]:
                    clean_dataset[key][subkey] = []
                clean_dataset[key][subkey].append(sample)

                property_counter.update(list(sample["sample_groundtruth"].keys()))
                event_class_counter.update(sample["ground_event_types"])

with open("dbpe__clean_data.json","w") as f:
    json.dump(clean_dataset, f)

with open("dbpe__clean_data.json","r") as f:
    clean_dataset = json.load(f)



def filter_clean_data(filter_size=50):
    filtered_dataset = {}
    for key in clean_dataset:
        for subkey in clean_dataset[key]:
            for sample in clean_dataset[key][subkey]:
                new_sample = copy.deepcopy(sample)
                for type in sample["sample_event_link"]["types"]:
                    if type in country_specific_event_classes + bad_event_classes:
                        new_sample["sample_event_link"]["types"].remove(type)
                if "type" in sample["sample_groundtruth"]:
                    del new_sample["sample_groundtruth"]["type"]
                for event_type in sample["ground_event_types"]:
                    if event_class_counter[event_type]<filter_size or event_type in country_specific_event_classes + bad_event_classes:
                        new_sample["ground_event_types"].remove(event_type)

                if not new_sample["ground_event_types"]:
                    continue

                for property in sample["sample_groundtruth"]:
                    if property_counter[property] < filter_size:
                        del new_sample["sample_groundtruth"][property]
                    elif property in bad_properties:
                        del new_sample["sample_groundtruth"][property]

                if not new_sample["sample_groundtruth"]:
                    continue

                if key not in filtered_dataset:
                    filtered_dataset[key] = {}
                if subkey not in filtered_dataset[key]:
                    filtered_dataset[key][subkey] = []
                filtered_dataset[key][subkey].append(new_sample)

    with open("dbpe__filtered_data"+str(filter_size)+".json","w") as f:
        json.dump(filtered_dataset, f)

def second_pass(data, class_filter, prop_filter, label ="2nd"):
    with open(data,"r") as f:
        d = json.load(f)
    c1 = Counter()
    c2 = Counter()
    filtered_dataset = {}


    for key in d:
        for subkey in d[key]:
            for sample in d[key][subkey]:
                s = set()
                s.update(sample["ground_event_types"])
                s.update(sample["sample_event_link"]["types"])
                s = list(s)
                c1.update(s)
                c2.update(list(sample["sample_groundtruth"].keys()))
    
    for key in d:
        for subkey in d[key]:
            for sample in d[key][subkey]:
                new_sample = copy.deepcopy(sample)
                for event_type in sample["ground_event_types"]:
                    if c1[event_type]<class_filter:
                        new_sample["ground_event_types"].remove(event_type)

                if not new_sample["ground_event_types"]:
                    continue

                for property in sample["sample_groundtruth"]:
                    if c2[property] < prop_filter:
                        del new_sample["sample_groundtruth"][property]

                if not new_sample["sample_groundtruth"]:
                    continue

                if key not in filtered_dataset:
                    filtered_dataset[key] = {}
                if subkey not in filtered_dataset[key]:
                    filtered_dataset[key][subkey] = []
                filtered_dataset[key][subkey].append(new_sample)

    with open("dbpe__"+label+"_pass_filtered_data.json","w") as f:
        json.dump(filtered_dataset, f)

    with open("dbpe_class_counter.json", "w") as f:
        json.dump(c1,f)
    
    with open("dbpe_property_counter.json", "w") as f:
        json.dump(c2,f)

filter_clean_data(100)

second_pass("dbpe__filtered_data100.json", class_filter = 100, prop_filter = 50, label="3rd")


print(2)
with open("dbpe_class_counter.json", "r") as f:
    C = json.load(f)

print(3)
with open("dbpe__3rd_pass_filtered_data.json","r") as f:
    dataset = json.load(f)
print(4)
n_events = 0
n_context = 0
n_links = 0
for key in dataset:
    context = ""
    for subkey in dataset[key]:
        n_events+=1
        for link in dataset[key][subkey]:
            if link["sample_groundtruth"]:
                n_links+=1
            if link["untokenized_context"] != context:
                context = link["untokenized_context"]
                n_context +=1


print(5)
all_samples = []
new_cl_counter = Counter()
new_prop_counter = Counter()

all_event_classes = set()
for event in dataset:
    for subevent in dataset[event]:
        for sample in dataset[event][subevent]:
            types = list(set(sample["sample_event_link"]["types"]+sample["ground_event_types"]))
            for c in types:
                all_event_classes.update({c})

with open("dbpe_all_event_classes.txt","w") as f:
    for line in list(all_event_classes):
        f.write(f"{line}\n")

#choose minority class
def select_minority_classes(dataset):
    all_samples = []
    for event in dataset:
        for subevent in dataset[event]:
            for sample in dataset[event][subevent]:
                types = list(set(sample["sample_event_link"]["types"]+sample["ground_event_types"]))
                min = 1000*1000
                for c in types:
                    if C[c] < min:
                        min = C[c]
                        min_cl = c
                del sample["sample_event_link"]["types"]
                sample["sample_event_link"]["event_class"] = min_cl
                new_cl_counter.update({min_cl:1})
                new_prop_counter.update(list(sample["sample_groundtruth"].keys()))
                all_samples.append(sample)
    return all_samples
print(6)
all_samples = select_minority_classes(dataset)
print(7)
n = len(all_samples)

def third_pass(all_samples, class_counter, property_counter, class_filter = 100, prop_filter=50):
    new_samples = []
    for sample in all_samples:
        new_sample = copy.deepcopy(sample)
        sample_class = sample["sample_event_link"]["event_class"] 
        if class_counter[sample_class]<class_filter:
            continue
        else:
            for property in sample["sample_groundtruth"]:
                if property_counter[property] < prop_filter:
                    del new_sample["sample_groundtruth"][property]

            if not new_sample["sample_groundtruth"]:
                continue
            
            new_samples.append(new_sample)

    return new_samples

all_samples = third_pass(all_samples, new_cl_counter, new_prop_counter)
print(8)
event_classes = set()
all_event_classes = set()
event_properties = set()
event_class_properties = {}
for sample in all_samples:
    sample_cl = sample["sample_event_link"]["event_class"]
    event_classes.update({sample_cl})
    properties = list(sample["sample_groundtruth"].keys())
    event_properties.update(properties)
    if sample_cl not in event_class_properties:
        event_class_properties[sample_cl] = set()
    event_class_properties[sample_cl].update(properties) 

event_classes = list(event_classes)
event_properties = list(event_properties)
all_event_classes = list(all_event_classes)
for cl in event_class_properties:
    event_class_properties[cl] = list(event_class_properties[cl])


with open("../data/training/t2e/dbpe_unlabelled_event.schema","w") as f:
    f.write(f"{list(event_classes)}\n")
    f.write(f"{list(event_properties)}\n")
    f.write(f"{event_class_properties}\n")

event_classes = [cl for cl in event_classes]
event_properties = [prop for prop in event_properties]
for cl in event_class_properties:
    event_class_properties[cl] = [prop for prop in event_class_properties[cl]]
event_class_properties = {cl:v for cl,v in event_class_properties.items()}

#with open("../data/training/t2e/dbpe_labelled_event.schema","w") as f:
    #f.write(f"{list(event_classes)}\n")
    #f.write(f"{list(event_properties)}\n")
    #f.write(f"{event_class_properties}\n")

#
print(9)
tmp = copy.deepcopy(all_samples)
tmp2 = {}
j = 0
for i, sample in enumerate(tmp):
    if "untokenized_context_with_links" in sample:
        print(j)
        j+=1
        context = sample["untokenized_context_with_links"]
    else:
        context = sample["untokenized_context"]
        tmp[i]["untokenized_context_with_links"] = context
    if context not in tmp2:
        tmp2[context] = [sample]
    else:
        if tmp2[context] == [sample]:
            continue
        else:
            event_classes = []
            for samp in tmp2[context]:
                event_classes.append(samp["sample_event_link"]["event_class"])
            if sample["sample_event_link"]["event_class"] in event_classes:
                continue
            else:
                tmp2[context].append(sample)
print(10)
all_samples = []
for i, context in enumerate(tmp2):
    if i%1000==0:
        print("%d / %d" %(i, len(tmp2)))
    new_sample = {}
    tmp_sample = tmp2[context]
    if len(tmp_sample)==1:
        cl = tmp_sample[0]["sample_event_link"]["event_class"]
        props = {prop:v for prop, v in tmp_sample[0]["sample_groundtruth"].items()}
        new_sample["untokenized_context"] = tmp_sample[0]["untokenized_context_with_links"]
        doc = nlp(new_sample["untokenized_context"])
        new_sample["tokenized_context"]= [t.text for t in doc]
        new_sample["groundtruth_events"] = [{cl : {"sample_event_link":tmp_sample[0]["sample_event_link"], "sample_groundtruth":props}}]
        all_samples.append(new_sample)
    else:
        for i, sample in enumerate(tmp2[context]):
            cl = sample["sample_event_link"]["event_class"]
            props = {prop:v for prop, v in sample["sample_groundtruth"].items()}
            new_sample["untokenized_context"] = sample["untokenized_context_with_links"]
            doc = nlp(new_sample["untokenized_context"])
            new_sample["tokenized_context"]= [t.text for t in doc]
            if "groundtruth_events" not in new_sample:
                new_sample["groundtruth_events"] = [{cl : {"sample_event_link":tmp_sample[0]["sample_event_link"], "sample_groundtruth":props}}]
            else:
                new_sample["groundtruth_events"].append({cl : {"sample_event_link":tmp_sample[0]["sample_event_link"], "sample_groundtruth":props}})
            all_samples.append(new_sample)


print(11)


with open("dbpedia_all_samples.json","w") as f:
    json.dump(all_samples, f)


def query_and_clean(event_dict):
    # For each event uri, query the dbpedia endpoint to get the properties and their values
    # Then match the properties and values to the ones in the groundtruth and remove the ones that don't match
    SPARQL_ENDPOINT = "https://dbpedia.org/sparql"
    
    # Headers for the HTTP request to specify that we expect JSON
    headers = {
        "Accept": "application/json"
    }

    # Iterate over each event URI in the event_dict
    for event_uri, properties in event_dict.items():
        # Construct the SPARQL query to retrieve properties and their values for the current event URI
        query = f"""
        SELECT ?property ?value WHERE {{
            <http://dbpedia.org/resource/{event_uri}> ?property ?value.
        }}
        """

        # Make the HTTP request to the SPARQL endpoint
        response = requests.get(SPARQL_ENDPOINT, params={'query': query}, headers=headers)
        
        if response.status_code == 200:
            # Parse the JSON response
            data = response.json()
            
            # Extract the bindings (the actual results) from the response
            results = data.get("results", {}).get("bindings", [])
            
            # Create a temporary dictionary to hold the properties and values from the query
            queried_properties = {}
            
            for result in results:
                # Extract the property and value URIs from the result
                property_uri = result["property"]["value"]
                value_uri = result["value"]["value"]
                
                # Add the property and value to the temporary dictionary
                if property_uri not in queried_properties:
                    queried_properties[property_uri] = set()
                queried_properties[property_uri].add(value_uri)

            # Now, match and clean the ground truth properties based on the queried properties
            for property, values in list(properties.items()):
                # Check if the property exists in the queried properties
                if property in queried_properties:
                    # Retain only the values that also exist in the queried results
                    event_dict[event_uri][property] = list(set([
                        value for value in values if value in queried_properties[property]
                    ]))
                else:
                    # If the property doesn't exist in the queried results, remove it from the ground truth
                    del event_dict[event_uri][property]

    return event_dict



def integrate_updated_properties(all_samples, updated_event_dict):
    """
    Updates the groundtruth properties in all_samples by removing values that are not present in updated_event_dict.
    Also removes properties without values and events without properties.

    Args:
    - all_samples (list): The original list of samples, each containing groundtruth events.
    - updated_event_dict (dict): A dictionary with event URIs as keys and updated properties and values as values.

    Returns:
    - list: The updated list of samples with properties cleaned according to updated_event_dict.
    """
    # Iterate over each sample in all_samples
    for sample in all_samples:
        groundtruth_events = sample["groundtruth_events"]
        
        events_to_remove = []  # List to keep track of events to remove

        # Iterate over each event in groundtruth_events
        for event in groundtruth_events:
            for event_class in event:
                event_link = event[event_class]["sample_event_link"]["target"]
                properties_to_remove = []  # List to keep track of properties to remove

                # Check if the current event_link is in updated_event_dict
                if event_link in updated_event_dict:
                    # Iterate over each property in the event's groundtruth
                    for property in event[event_class]["sample_groundtruth"]:
                        property_uri = 'http://dbpedia.org/ontology/' + property
                        
                        # Check if the property exists in updated_event_dict for the current event_link
                        if property_uri in updated_event_dict[event_link]:
                            new_values = []
                            for value in event[event_class]["sample_groundtruth"][property]:
                                standardized_value = None
                                if "date" in property.lower():
                                    date_obj = parser.parse(value[-1], fuzzy=True)
                                    standardized_value = date_obj.strftime('%Y-%m-%d')
                                else:
                                    standardized_value = 'http://dbpedia.org/resource/' + value[-1]
                                
                                if standardized_value in updated_event_dict[event_link][property_uri]:
                                    new_values.append(value)
                            
                            # Update the property with filtered values
                            event[event_class]["sample_groundtruth"][property] = new_values

                            # If no values are left for the property, mark it for removal
                            if not new_values:
                                properties_to_remove.append(property)
                    
                    # Remove properties marked for removal
                    for property in properties_to_remove:
                        del event[event_class]["sample_groundtruth"][property]

                    # If no properties are left for the event, mark the event for removal
                    if not event[event_class]["sample_groundtruth"]:
                        events_to_remove.append(event)
                
                # If the event_link is not in updated_event_dict, assume all properties are correct and do nothing

        # Remove events marked for removal
        for event in events_to_remove:
            groundtruth_events.remove(event)

    return all_samples


with open("dbpedia_all_samples.json","r") as f:
    all_samples = json.load(f)

event_dict_list = []
for sample in all_samples:
    event_dict = {}
    groundtruth_events = sample["groundtruth_events"]
    for event in groundtruth_events:
        for event_class in event:
            event_link = event[event_class]["sample_event_link"]["target"]
            for property in event[event_class]["sample_groundtruth"]:
                for property_value in event[event_class]["sample_groundtruth"][property]:
                    property_dbo_uri = property_value[4]
                    property_value_resource = property_value[-1]
                    if "date" in property.lower():
                        date_obj = parser.parse(property_value_resource, fuzzy=True)
                        standardized_date_string = date_obj.strftime('%Y-%m-%d')
                        property_value_resource = standardized_date_string
                    if event_link not in event_dict:
                        event_dict[event_link] = {}
                    if property_dbo_uri not in event_dict[event_link]:
                        event_dict[event_link][property_dbo_uri] = []
                    if not "date" in property.lower():   
                        event_dict[event_link][property_dbo_uri].append("http://dbpedia.org/resource/"+property_value_resource)
                    else:
                        event_dict[event_link][property_dbo_uri].append(property_value_resource)
    event_dict_list.append(event_dict)

updated_event_dict = query_and_clean(event_dict_list)

with open("dbpedia_updated_event_dict.json","w") as f:
    json.dump(updated_event_dict, f)

with open("dbpedia_updated_event_dict.json","r") as f:
    updated_event_dict = json.load(f)


all_samples = integrate_updated_properties(all_samples, updated_event_dict)




def convert_to_regular_chars(input_str):
    # Use a list comprehension to process each character in the input string
    converted_chars = [
        # Replace non-standard whitespace characters with a regular space
        ' ' if unicodedata.category(char).startswith('Z') else char
        for char in input_str
    ]

    # Join the processed characters back into a single string
    normalized_str = ''.join(converted_chars)

    # Further normalize the string to 'NFC' to combine any decomposable characters back to their single-character form
    return unicodedata.normalize('NFC', normalized_str)


def get_correct_indices_(tokenized_sentence, groundtruth_text, original_start_index, original_end_index):
    flag = False
    if tokenized_sentence[original_start_index:original_end_index + 1] == groundtruth_text.split():
        return original_start_index, original_end_index, flag

    start = -1
    end = -1

    # Copy the original tokenized_sentence to avoid modifying it
    tmp_tokenized_sentence = []
    
    # Create a mapping from the modified tokenized_sentence indices to the original indices
    index_mapping = []

    # Modify the tokenized_sentence and build the index mapping
    for i, token in enumerate(tokenized_sentence):
        cleaned_token = re.sub(r'\W+', '', token)
        if cleaned_token:  # Only consider non-empty tokens after cleaning
            tmp_tokenized_sentence.append(cleaned_token)
            index_mapping.append(i)
            
    # Clean the groundtruth_text
    tmp_groundtruth_text = re.sub(r'\W+', ' ', groundtruth_text).split()

    # Search for the groundtruth_text in the modified tokenized_sentence
    for i, token in enumerate(tmp_tokenized_sentence):
        if token in groundtruth_text:
            # Check if the sequence starting at this index matches the groundtruth_text
            if tmp_tokenized_sentence[i:i + len(tmp_groundtruth_text)] == tmp_groundtruth_text:
                start = index_mapping[i]  # Map back to the original index
                end = index_mapping[i + len(tmp_groundtruth_text) - 1]  # Map back to the original index
                break

    if start == -1:
        flag = True

    return start, end, flag

def create_baseline_training_data(all_samples, path = "../data/training"):
    t2e_training_samples = []
    relation_extraction_samples = []
    dygiepp_training_samples = []
    with_entities_samples = []
    for_deletion = set()
    for sample_num, sample in enumerate(all_samples):
        if sample_num % 100 == 0:
            print("%d / %d" %(sample_num, len(all_samples)))
        sentence_tokenized_context = []
        sentence_tokenized_events = []
        re_tokenized_events = []
        with_entities_events = []
        doc = nlp(sample["untokenized_context"])
        sync = dict()
        for sent in doc.sents:
            sentence_tokenized_context.append([])
            sentence_tokenized_events.append([])
            re_tokenized_events.append([])
            with_entities_events.append([])
            for t in sent:
                sentence_tokenized_context[-1].append(t.text)
                for event in sample["groundtruth_events"]:
                    for event_class in event:
                        if t.i == event[event_class]["sample_event_link"]["token_range"][0]:
                            sentence_tokenized_events[-1].append([[t.i, event_class]])
                            re_tokenized_events[-1].append([[t.i,event[event_class]["sample_event_link"]["token_range"][1],event_class]])
                            with_entities_events[-1].append([[t.i,event[event_class]["sample_event_link"]["token_range"][1],event_class]])
                            for t2 in sent:
                                for property in event[event_class]["sample_groundtruth"]:
                                    if isinstance(event[event_class]["sample_groundtruth"][property][0], list):
                                        if event[event_class]["sample_groundtruth"][property][0][0] == t2.i:
                                            if event_class not in sync:
                                                sync[event_class] = []
                                            sync[event_class].append(property)
                                            ground_text = event[event_class]["sample_groundtruth"][property][0][-2] if len(event[event_class]["sample_groundtruth"][property][0])>6 else event[event_class]["sample_groundtruth"][property][0][-1]
                                            original_start_index = event[event_class]["sample_groundtruth"][property][0][0]
                                            original_end_index = event[event_class]["sample_groundtruth"][property][0][1]
                                            start_index, end_index, flag = get_correct_indices_(sample["tokenized_context"], ground_text, original_start_index, original_end_index)
                                            if flag:
                                                for_deletion.add(sample_num)

                                            sentence_tokenized_events[-1][-1].append([start_index,end_index,property])
                                            re_tokenized_events[-1][-1].append([start_index,end_index,property])
                                            if len(event[event_class]["sample_groundtruth"][property][0])>6:
                                                entity_link = event[event_class]["sample_groundtruth"][property][0][-1]
                                                with_entities_events[-1][-1].append([start_index,end_index,property, ground_text, entity_link])
                                            elif "time" in property.lower() or "date" in property.lower():
                                                    date_obj = parser.parse(ground_text, fuzzy=True)
                                                    standardized_date_string = date_obj.strftime('%Y-%m-%d')
                                                    with_entities_events[-1][-1].append([start_index,end_index,property, ground_text, standardized_date_string])
                                            else:
                                                with_entities_events[-1][-1].append([start_index,end_index,property, ground_text])
                                    else:
                                        if event[event_class]["sample_groundtruth"][property][0] == t2.i:
                                            if event_class not in sync:
                                                sync[event_class] = []
                                            sync[event_class].append(property)
                                            ground_text = event[event_class]["sample_groundtruth"][property][0][-2] if len(event[event_class]["sample_groundtruth"][property][0])>6 else event[event_class]["sample_groundtruth"][property][0][-1]
                                            start_index, end_index, flag =  get_correct_indices_(sample["tokenized_context"], ground_text, original_start_index, original_end_index)
                                            if flag:
                                                for_deletion.add(sample_num)
                                            sentence_tokenized_events[-1][-1].append([start_index,end_index,property])
                                            re_tokenized_events[-1][-1].append([start_index,end_index,property])
                                            if len(event[event_class]["sample_groundtruth"][property][0])>6:
                                                entity_link = event[event_class]["sample_groundtruth"][property][0][-1]
                                                with_entities_events[-1][-1].append([start_index,end_index,property, ground_text, entity_link])
                                            elif "time" in property.lower() or "date" in property.lower():
                                                date_obj = parser.parse(ground_text, fuzzy=True)
                                                standardized_date_string = date_obj.strftime('%Y-%m-%d')
                                                with_entities_events[-1][-1].append([start_index,end_index,property, ground_text, standardized_date_string])
                                            else:
                                                with_entities_events[-1][-1].append([start_index,end_index,property, ground_text])
        X = "<extra_id_0>"
        for event in sample["groundtruth_events"]:
            for event_class in event :
                event_trigger = event[event_class]["sample_event_link"]["anchor_text"]
                X += "<extra_id_0>"+event_class+" "+ event_trigger
                groundtruth = event[event_class]["sample_groundtruth"]
                for property in groundtruth: 
                        if event_class not in sync or property not in sync[event_class]:
                            continue
                        X += "<extra_id_0>"
                        if isinstance(groundtruth[property][0], list):
                            X += property+" "+ str(groundtruth[property][0][5])
                        else:
                            X += property+" "+ groundtruth[property][5]
                        X += "<extra_id_1>" 
                X += "<extra_id_1>"
        X += "<extra_id_1>"
        t2e_training_sample = {}
        t2e_training_sample["text"] = sample["untokenized_context"]
        t2e_training_sample["event"] = X
        t2e_training_samples.append(t2e_training_sample)

        dygiepp_training_sample = {}
        dygiepp_training_sample["doc_key"] = sample_num
        dygiepp_training_sample["dataset"] = "event extraction"
        dygiepp_training_sample["sentences"] = sentence_tokenized_context
        dygiepp_training_sample["events"] =  sentence_tokenized_events
        dygiepp_training_samples.append(dygiepp_training_sample)

        re_sample = {}
        re_sample["doc_key"] = sample_num
        re_sample["dataset"] = "event extraction"
        re_sample["sentences"] = sentence_tokenized_context
        re_sample["events"] =  re_tokenized_events
        relation_extraction_samples.append(re_sample)

        with_entities_sample = {}
        with_entities_sample["doc_key"] = sample_num
        with_entities_sample["dataset"] = "event extraction"
        with_entities_sample["sentences"] = sentence_tokenized_context
        with_entities_sample["events"] =  with_entities_events
        with_entities_samples.append(with_entities_sample)

    for_deletion = list(for_deletion)
    for_deletion.sort(reverse=True)
    for i in for_deletion:
        del dygiepp_training_samples[i]
        del t2e_training_samples[i]
        del relation_extraction_samples[i]
        del with_entities_samples[i]

    indices = list(range(len(dygiepp_training_samples)))
    train_idx, test_idx = train_test_split(indices, test_size=0.3)
    dev_idx, test_idx = train_test_split(test_idx, test_size=0.5)

    train=[dygiepp_training_samples[i] for i in train_idx]
    test=[dygiepp_training_samples[i] for i in test_idx]
    dev=[dygiepp_training_samples[i] for i in dev_idx]

    with jsonlines.open(path+"/dygiepp/dbpe_eq_train.json","w") as f:
        f.write_all(train)

    with jsonlines.open(path+"/dygiepp/dbpe_eq_test.json","w") as f:
        f.write_all(test)

    with jsonlines.open(path+"/dygiepp/dbpe_eq_dev.json","w") as f:
        f.write_all(dev)

    train=[t2e_training_samples[i] for i in train_idx]
    test=[t2e_training_samples[i] for i in test_idx]
    dev=[t2e_training_samples[i] for i in dev_idx]

    with jsonlines.open(path+"/t2e/dbpe_eq_train.json","w") as f:
        f.write_all(train)

    with jsonlines.open(path+"/t2e/dbpe_eq_test.json","w") as f:
        f.write_all(test)

    with jsonlines.open(path+"/t2e/dbpe_eq_val.json","w") as f:
        f.write_all(dev)

    train=[relation_extraction_samples[i] for i in train_idx]
    test=[relation_extraction_samples[i] for i in test_idx]
    dev=[relation_extraction_samples[i] for i in dev_idx]

    with jsonlines.open(path+"/re/dbpe_eq_train.json","w") as f:
        f.write_all(train)

    with jsonlines.open(path+"/re/dbpe_eq_test.json","w") as f:
        f.write_all(test)

    with jsonlines.open(path+"/re/dbpe_eq_dev.json","w") as f:
        f.write_all(dev)

    train=[with_entities_samples[i] for i in train_idx]
    test=[with_entities_samples[i] for i in test_idx]
    dev=[with_entities_samples[i] for i in dev_idx]

    with jsonlines.open(path+"/with_entities/dbpe_eq_train.json","w") as f:
        f.write_all(train)

    with jsonlines.open(path+"/with_entities/dbpe_eq_test.json","w") as f:
        f.write_all(test)
    
    with jsonlines.open(path+"/with_entities/dbpe_eq_dev.json","w") as f:
        f.write_all(dev)



create_baseline_training_data(all_samples, path = "../data/training")

def create_sparse_full_data(tokenized_path, untokenized_path, output_path):
    data = []
    tokenized_samples = []
    untokenized_samples = []
    with open("filtered_dbpedia_event2.schema","r") as f:
        for o, line in enumerate(f):
            if o ==2:
                event2role =  ast.literal_eval(line)

    with open(tokenized_path, 'r') as json_file1, open(untokenized_path, "r") as json_file2:
        json_list1 = list(json_file1)
        json_list2 = list(json_file2)

    for json_str1, json_str2 in zip(json_list1, json_list2):
        tokenized_samples.append(json.loads(json_str1))
        untokenized_samples.append(json.loads(json_str2))

    k = 0

    for t, (t_sample, u_sample) in enumerate(zip(tokenized_samples, untokenized_samples)):
        if t % 1000==0:
            print("%d out of %d completed"%(t, len(tokenized_samples)))
        data.append({"title":str(t), "paragraphs":[]})
        u_sentence = u_sample["text"]
        for i, t_sentence in enumerate(t_sample["sentences"]):
            data[-1]["paragraphs"].append({"qas":[], "context":u_sentence})
            tmp = {}
            if len(t_sample["events"])<=i:
                continue
            for event in t_sample["events"][i]:
                token_start = event[0][0] 
                token_end = event[0][1]
                event_type = event[0][2] 
                trigger = " ".join(t_sentence[token_start:token_end+1]) 
                if event_type not in tmp:
                    tmp[event_type] = {"triggers":[], "arguments":{}}
                tmp[event_type]["triggers"].append(trigger)

                for event_part in event[1:]: 
                    token_start = event_part[0]
                    token_end = event_part[1]
                    role_type = event_part[2]
                    arg_string = event_part[3]

                    if role_type not in tmp[event_type]["arguments"]:
                        tmp[event_type]["arguments"][role_type] = []
  
                    tmp[event_type]["arguments"][role_type].append(arg_string)
                event_type = event_type
                for property in event2role[event_type]:      
                    k+=1
                    question = event_type + "," + property
                    if not event_type in tmp or not tmp[event_type]["arguments"]:
                        data[-1]["paragraphs"][-1]["qas"].append({"question":question, "id":str(k), "answers":[], "is_impossible":True})
                
                    elif event_type in tmp and tmp[event_type]["arguments"]:
                        if property not in tmp[event_type]["arguments"]:
                            data[-1]["paragraphs"][-1]["qas"].append({"question":question, "id":str(k), "answers":[], "is_impossible":True})
                        elif property in tmp[event_type]["arguments"]:
                            answ = {}
                            answrs = []
                            for argument_string in tmp[event_type]["arguments"][property]:
                                answ["text"] = argument_string
                                converted = convert_to_regular_chars(u_sentence)
                                answ["answer_start"] = converted.index(argument_string)
                                answrs.append(answ)
                            data[-1]["paragraphs"][-1]["qas"].append({"question":question, "id":str(k), "answers":answrs, "is_impossible":False})
    return data


def create_training_data(path):

    train = create_sparse_full_data(path+"with_entities/dbpe_eq_train.json", path+"t2e/dbpe_eq_train.json", path+"re/dbpe_re_train.json")
    train_dataset = {"version":"v2.0", "data":train}
    with open(path+'re/dbpe2_sparse_re_train.json', 'w') as f:
        json.dump(train_dataset, f)
    train_dataset = {}
    train = {}

    test = create_sparse_full_data(path+"with_entities/dbpe_eq_test.json", path+"t2e/dbpe_eq_test.json", path+"re/dbpe_re_test.json")
    test_dataset = {"version":"v2.0", "data":test}
    with open(path+'re/dbpe2_sparse_re_test.json', 'w') as f:
        json.dump(test_dataset, f)
    test_dataset={}
    test = {}

    dev = create_sparse_full_data(path+"with_entities/dbpe_eq_dev.json", path+"t2e/dbpe_eq_val.json", path+"re/dbpe_re_dev.json")
    dev_dataset = {"version":"v2.0", "data":dev}
    with open(path+'re/dbpe2_sparse_re_dev.json', 'w') as f:
        json.dump(dev_dataset, f)
    dev_dataset = {}
    dev = {}




def create_full_dataset():
    # The function merges with_entities train,test,dev files into one file
    with_entities_train = []
    with_entities_test = []
    with_entities_dev = []

    with jsonlines.open("../data/training/with_entities/dbpe_eq_train.json","r") as f:
        with_entities_train = list(f)
    with jsonlines.open("../data/training/with_entities/dbpe_eq_test.json","r") as f:
        with_entities_test = list(f)
    with jsonlines.open("../data/training/with_entities/dbpe_eq_dev.json","r") as f:
        with_entities_dev = list(f)

    with_entities_train.extend(with_entities_test)
    with_entities_train.extend(with_entities_dev)

    with jsonlines.open("../data/training/with_entities/dbpedia_full_dataset.json", "w") as f:
        f.write_all(with_entities_train)



# Function to check if a URI (class or property) is a dbo URI
def is_dbo_uri(uri, uri_type):
    sparql = SPARQLWrapper("http://dbpedia.org/sparql")
    type_dict = {'class': 'owl:Class', 'property': 'rdf:Property'}
    query = f"""
    ASK {{
        ?uri a {type_dict[uri_type]} .
        FILTER(str(?uri) = "http://dbpedia.org/ontology/{uri}")
    }}
    """
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    result = sparql.query().convert()

    return result['boolean']

# Function to check if a property is relevant to a class
def is_property_relevant_to_class(class_name, property_name):
    sparql = SPARQLWrapper("http://dbpedia.org/sparql")
    query = f"""
    ASK {{
        ?instance a dbo:{class_name} .
        ?instance dbo:{property_name} ?value .
    }}
    """
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    result = sparql.query().convert()

    return result['boolean']

def run_queries():
    # Read the input file
    with open('dbp_event.schema', 'r') as file:
        dbpedia_classes = ast.literal_eval(file.readline().strip())
        dbpedia_properties = ast.literal_eval(file.readline().strip())
        class_properties_dict = ast.literal_eval(file.readline().strip())

    # Filter classes and properties that are not dbo URIs
    filtered_classes = [cls for cls in dbpedia_classes if is_dbo_uri(cls, 'class')]
    filtered_properties = [prop for prop in dbpedia_properties if is_dbo_uri(prop, 'property')]
    filtered_class_properties_dict = {cls: [prop for prop in props if prop in filtered_properties] for cls, props in class_properties_dict.items() if cls in filtered_classes}

    # Write the filtered data to a new file
    with open('filtered_dbpedia_event.schema', 'w') as file:
        file.write(str(filtered_classes) + '\n')
        file.write(str(filtered_properties) + '\n')
        file.write(str(filtered_class_properties_dict))

    print("Now, we will verify the filtered data and write the verified data to a new file.")
    # Verify and filter class-properties pairs
    verified_class_properties_dict = {}
    for cls in filtered_classes:
        relevant_properties = []
        for prop in filtered_properties:
            if is_property_relevant_to_class(cls, prop):
                relevant_properties.append(prop)
                print(f"{prop} is relevant to {cls}")
        if relevant_properties:
            verified_class_properties_dict[cls] = relevant_properties
    verified_classes = list(verified_class_properties_dict.keys())
    verified_properties = list(set([prop for props in verified_class_properties_dict.values() for prop in props]))
    # Write the verified data to a new file
    with open('verified_dbpedia_event.schema', 'w') as file:
        file.write(str(verified_classes) + '\n')
        file.write(str(verified_properties) + '\n')
        file.write(str(verified_class_properties_dict))

def read_dataset(file_path):
    with open(file_path, 'r') as f:
        for line in f:
            yield json.loads(line)

def read_schema_from_dataset(train, test, dev):
    schema = {}
    for dataset in [train, test, dev]:
        for row in dataset:
            for sentence in row["events"]:
                for event in sentence:
                    if event[0][2] not in schema:
                        schema[event[0][2]] = []
                    for arg in event[1:]:
                        if arg[2] not in schema[event[0][2]]:
                            schema[event[0][2]].append(arg[2])

    with open('dbp_event.schema', 'w') as file:
        file.write(str(list(schema.keys())) + '\n')
        file.write(str(list(set([prop for props in schema.values() for prop in props]))) + '\n')
        file.write(str(schema))

    return



path = "../data/training/with_entities/"
train_path = path+"dbpe_eq_train.json"
test_path = path+"dbpe_eq_test.json"
dev_path = path+"dbpe_eq_dev.json" 

read_schema_from_dataset(list(read_dataset(train_path)), list(read_dataset(test_path)), list(read_dataset(dev_path)))
run_queries()

with open('verified_dbpedia_event.schema', 'r') as file:
    verified_classes = ast.literal_eval(file.readline().strip())
    verified_properties = ast.literal_eval(file.readline().strip())
    verified_class_properties_dict = ast.literal_eval(file.readline().strip())

train_for_deletion = set()
test_for_deletion = set()
dev_for_deletion = set()

for dataset_path in [train_path, test_path, dev_path]:
    dataset = list(read_dataset(dataset_path)) 
    filtered_dataset = []

    schema = {}

    for row in dataset:
        doc_key = row["doc_key"]
        sentences = row["sentences"]
        filtered_row = {"doc_key":doc_key,"dataset":"event extraction","sentences":sentences, "events":[]}
        for sentence in row["events"]:
            filtered_sentence = []
            for event in sentence:
                filtered_event = []
                if event[0][2] not in verified_classes:
                    continue
                for arg in event[1:]:
                    if arg[2] not in verified_properties:
                        continue
                    elif arg[2] in verified_class_properties_dict[event[0][2]]:
                        filtered_event.append(event[0])
                        filtered_event.append(arg)
                        if event[0][2] not in schema:
                            schema[event[0][2]] = []
                        if arg[2] not in schema[event[0][2]]:
                            schema[event[0][2]].append(arg[2])
                if filtered_event:
                    merge = []
                    tmp = []
                    for event_part in filtered_event:
                        if event_part[2] not in merge:
                            merge.append(event_part[2])
                            tmp.append(event_part)
                        else:
                            continue
                    filtered_event = tmp
                    filtered_sentence.append(filtered_event)
            if filtered_sentence:
                filtered_row["events"].append(filtered_sentence)
        if filtered_row["events"]:
            filtered_dataset.append(filtered_row)
        else:
            if dataset_path == train_path:
                train_for_deletion.add(doc_key)
            elif dataset_path == test_path:
                test_for_deletion.add(doc_key)
            elif dataset_path == dev_path:
                dev_for_deletion.add(doc_key)

    with open(dataset_path, 'w') as file:
        for row in filtered_dataset:
            file.write(json.dumps(row) + '\n')

    filtered_classes = list(schema.keys())
    filtered_properties = list(set([prop for props in schema.values() for prop in props]))

    with open('filtered_dbpedia_event2.schema', 'w') as file:
        file.write(str(filtered_classes) + '\n')
        file.write(str(filtered_properties)+'\n')
        file.write(str(schema))

    with open('/home/kuculo/T-SEE/data/training/t2e/dbpe_unlabelled_event.schema', 'w') as file:
        file.write(str(filtered_classes) + '\n')
        file.write(str(filtered_properties)+'\n')
        file.write(str(schema))


row_indices = {}
for folder in ["dygiepp", "re", "t2e"]:
    files = ["dbpe_eq_train.json", "dbpe_eq_test.json", "dbpe_eq_dev.json"]
    if folder == "t2e":
        files[2] = "dbpe_eq_val.json"
    for file, rows2delete in zip(files, [train_for_deletion, test_for_deletion, dev_for_deletion]):
        file_split = file.split("_")[-1].split(".")[0]
        if file_split == "val":
            file_split = "dev"
        if file_split not in row_indices:
            row_indices[file_split] = []
        with jsonlines.open("../data/training/"+folder+"/"+file, "r") as f:
            dataset = list(f)
        if folder == "dygiepp":
            for i, d in enumerate(dataset):
                if d["doc_key"] in rows2delete:
                    row_indices[file_split].append(i)
        print(row_indices)
        filtered_dataset = [d for i, d in enumerate(dataset) if i not in row_indices[file_split]]

        with jsonlines.open("../data/training/"+folder+"/"+file, "w") as f:
            f.write_all(filtered_dataset)


create_training_data("../data/training/")

create_full_dataset()