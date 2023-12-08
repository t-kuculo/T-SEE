import sys, os
import ast
import os
from classes import * 
import ndjson
from collections import Counter,  OrderedDict
import shelve
import time
import json
import datetime
import datefinder
import shelve
import csv
import re
rx = r"\.(?=\S)"
import copy
from sklearn.model_selection import train_test_split
import jsonlines
import spacy  

nlp = spacy.load("en_core_web_md")
nlp.add_pipe("entityLinker")

def remove_non_ascii(string):
    return ''.join(char for char in string if ord(char) < 256)

print("Loading shelves")
dbpedia_ids_to_wikidata_ids = shelve.open('../data/shelves/dbpedia_to_wikidata_en')
print(1)
#types_dbo = shelve.open('../data/shelves/types_dbo')
print(2)
#types_wd = shelve.open('../data/shelves/types_wd')
print(3)
instance_types = shelve.open("../data/shelves/instance_types")
print(4)




def first_stage_eval_data():
    '''
    This function loops through all the files in the raw_data directory and creates a dictionary of the following form:
    {parent_event_wd_id: {subevent_wd_id@subevent_link: {"samples":[(link, sentence.text, sentence.links)], "groundtruth":{}}}}
    
    The samples are the sentences in which the subevent is mentioned. The groundtruth is empty at this stage.
    '''
    directory = "../data/raw_data/"
    dataset = {}
    dygiepp_data = []
    i = 0
    e = time.time()
    context_window = 0
    for u, filename in enumerate(os.listdir(directory)):
        print("%d out of %d completed"%(u, len(os.listdir(directory))))
        if ".gitignore" in filename:
            continue
        with open(directory+filename, "r") as f:
            json_list = ndjson.load(f)
        for g,js in enumerate(json_list):
            Article_obj = Article(js)
            parent_event_link = ("_".join(Article_obj.article_name.split(" ")))
            try:
                parent_event_wd_id = dbpedia_ids_to_wikidata_ids[parent_event_link]
            except (KeyError):
                continue
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
                                    subevent_wd_id = dbpedia_ids_to_wikidata_ids[subevent_link] 
                                except (KeyError):
                                    continue
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

    with open('eval_data_with_context_window.txt', 'w') as f:
        json.dump(dataset,f)
    
first_stage_eval_data()



with open('eval_data_with_context_window.txt', 'r') as f:
    dataset = json.load(f)
    
print("First stage done")

def load_all_properties():
    '''
    all_subevent_properties.csv is a file containing all the Wikidata properties of all the subevents in the dataset.
    This function loads the file and creates a dictionary of the following form:
    {subevent_wd_id: {property: [value1, value2, ...]}}
    '''
    subevents_properties = {}
    with open("all_subevent_properties.csv", "r") as f:
        lines = f.readlines()[1:]
        for row in lines:
            row = row.rstrip().split("\t")
            event, property, value = row[0], row[1], row[2]
            if event not in subevents_properties:
                subevents_properties[event] = {}
                
            if property not in subevents_properties[event]:
                subevents_properties[event][property] = []
            subevents_properties[event][property].append(value)
    return subevents_properties

subevents_properties = load_all_properties()

def filter_out_events_without_properties():
    '''
    This function filters out subevents that have no properties in the full raw dataset.
    '''
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
    with open('all_data.txt', 'w') as f:
        json.dump(dataset,f)

filter_out_events_without_properties()

with open('all_data.txt', 'r') as f:
    dataset = json.load(f)

#if the_data folder doesn't exist create it
if not os.path.exists("the_data"):
    os.makedirs("the_data")


def extract_data(dataset, instance_types, subevents_properties, nlp):
    ''' 
    Processes a given dataset to extract and save detailed information about subevents and their properties.

    Parameters:
    - dataset (list/dict): The dataset containing events and subevents information.
    - instance_types (dict): Shelve object mapping types to Wikidata IDs.
    - subevents_properties (dict): Dictionary of properties related to subevents.
    - nlp (spacy.Language): A spaCy NLP model object.

    This function loops through the dataset and extracts the following information for each sample:
    - the anchor text of the subevent
    - the context of the sample
    - the token range of the anchor text in the context
    - the character range of the anchor text in the context
    - the tokenized context
    - the untokenized context
    - the groundtruth properties of the subevent
    - the token range of the groundtruth properties in the context
    - the character range of the groundtruth properties in the context
    - the wikidata id of the groundtruth properties

    Then it saves the data in the following form:
    {parent_event_wd_id: {subevent_wd_id@subevent_link: [{"sample_event_link":sample_event_link, "untokenized_context":untokenized_context, "tokenized_context":tokenized_context, "sample_groundtruth":sample_groundtruth, "ground_event_types":ground_event_types}]}}
    
    Where sample_event_link is a dictionary containing the following information:
    - the anchor text of the subevent
    - the token range of the anchor text in the context
    - the character range of the anchor text in the context
    - the wikidata id of the subevent
    Where sample_groundtruth is a dictionary containing the following information as values:
    new_groundtruth[property].append([start, end, start_char, end_char, property, doc.text[start_char:end_char]]
    - the token range of the groundtruth properties in the context
    - the character range of the groundtruth properties in the context
    - the wikidata id of the groundtruth properties
    - the value of the groundtruth properties (the number in case of quantities, the (iso format) date in case of time expressions, the entity string in case of entities)
    - optionally the wikidata id of the value of the entity values
    Where ground_event_types is a list of the event classes of the subevent.
    '''
    new_dataset = {}
    for num, event_wd_id in enumerate(dataset):
        print_progress(num, len(dataset))
        new_dataset = process_event(event_wd_id, dataset, instance_types, subevents_properties, nlp, new_dataset)
        new_dataset = save_data(num, new_dataset)
    save_data(num*10, new_dataset)  # Final save to capture remaining data

def print_progress(current, total):
    print(f"{current} out of {total}")

def process_event(event_wd_id, dataset, instance_types, subevents_properties, nlp, new_dataset):
    ''' 
    Processes all subevents of a given event.

    Parameters:
    - event_wd_id (str): Wikidata ID of the event being processed.
    - dataset (dict): The dataset containing events and their related data.
    - instance_types (dict): Shelve object mapping types to Wikidata IDs.
    - subevents_properties (dict): Dictionary of properties related to subevents.
    - nlp (spacy.Language): A spaCy NLP model object.
    - new_dataset (dict): The dataset being constructed with processed information.

    Returns:
    dict: Updated dataset with processed information for the given event.

    This function iterates through each subevent of an event and processes them individually.
    '''
    for subevent in dataset[event_wd_id]:
        new_dataset = process_subevent(subevent, event_wd_id, dataset, instance_types, subevents_properties, nlp, new_dataset)
    return new_dataset

def process_subevent(subevent, event_wd_id, dataset, instance_types, subevents_properties, nlp, new_dataset):
    '''
    Processes individual subevents within an event, extracting and structuring relevant information.

    Parameters:
    - subevent (str): Identifier of the subevent being processed.
    - event_wd_id (str): Wikidata ID of the parent event.
    - dataset (dict): The dataset containing events and subevents information.
    - instance_types (dict): Shelve object mapping types to Wikidata IDs.
    - subevents_properties (dict): Dictionary of properties related to subevents.
    - nlp (spacy.Language): A spaCy NLP model object.
    - new_dataset (dict): The dataset being constructed with processed information.

    Returns:
    dict: Updated dataset with processed information from the subevent.

    This function processes each sample of a subevent, skipping subevents with missing links or no types (also skipping those whose type is "SocietalEvent" or "Event"). 
    It handles the processing of subevent samples and updates the dataset accordingly.
    '''
    try:
        subevent_wd_id, subevent_link = subevent.split("@")
    except ValueError:
        print("Skipping subevent with no link")
        return new_dataset

    if subevent_wd_id in instance_types:
        subevent_types = instance_types[subevent_wd_id]
    else:
        print("Skipping subevent with no types")
        return new_dataset
    
    subevent_types = list(subevent_types)
    groundtruth_properties = dataset[event_wd_id][subevent]["groundtruth"]
    tmp = subevent_types
    if "Q1656682" in subevent_types: #event
        subevent_types.remove("Q1656682") #event
    if not subevent_types:
        return new_dataset
    
    subevent_types = list(subevent_types)
    groundtruth_properties = dataset[event_wd_id][subevent]["groundtruth"]
    groundtruth_properties = {k:v for k, v in groundtruth_properties.items()}
    groundtruth_values = {wd:k for k,v in groundtruth_properties.items() for wd in v}

    if groundtruth_values == {}:
        return new_dataset

    for sample in dataset[event_wd_id][subevent]["samples"]:
        new_dataset = process_sample(sample, event_wd_id, subevent_wd_id, subevent_types, subevents_properties, groundtruth_properties, groundtruth_values, nlp, new_dataset)
    return new_dataset

def preprocess_sample_context(sample, subevent_wd_id, subevents_properties, rx):
    '''
    Preprocesses the context of each sample, focusing on samples relevant to the subevent.

    Parameters:
    - sample (tuple): A tuple containing the subevent and its context.
    - subevent_wd_id (str): Wikidata ID of the subevent.
    - subevents_properties (dict): Dictionary of properties related to subevents.
    - rx (regex.Pattern): A compiled regular expression pattern for processing the text.

    Returns:
    tuple: A tuple containing the sample and its preprocessed context, or (None, None) for irrelevant or non-compliant samples.

    The function eliminates samples likely to be lists (contain a '*' or are too short) and those with too short contexts.
    '''
    if subevent_wd_id not in subevents_properties:
        return None, None

    sample_context = re.sub(rx, ". ", sample[1])
    sample_context = sample_context.replace("-", " - ").replace("–", " – ").replace("  ", " ")

    if "*" in sample_context and len(sample_context) < 100:
        return None, None
    if len(sample_context) < 30:
        return None, None
    
    return sample, sample_context

def process_sample_nlp(sample_subevent, sample_context, nlp):
    '''
    Processes the context of each sample using spaCy for NLP tasks.

    Parameters:
    - sample_subevent (dict): Dictionary containing details of the subevent.
    - sample_context (str): The context text of the sample.
    - nlp (spacy.Language): A spaCy NLP model object.

    Returns:
    tuple: A tuple containing the spaCy object for the anchor text and the processed document, or (None, None) if processing fails.

    The function applies NLP processing to the context and anchor text of a sample. It handles exceptions such as missing anchor text.
    '''
    try:
        subevent_anchor = nlp(sample_subevent["anchor_text"])
        doc = nlp(sample_context)
        return subevent_anchor, doc
    except:
        print("Skipping sample with no anchor text")
        return None, None

def process_anchor_text(sample_subevent, doc):
    '''
    Finds the token and character range of the anchor text of each sample in the context.
    We make sure to account for special cases such as the anchor text being a substring of a token or the anchor text being a substring of a bigger word.


    Parameters:
    - sample_subevent (dict): Dictionary containing details of the subevent, including anchor text.
    - doc (spacy.tokens.Doc): A spaCy Document object representing the tokenized context.

    Returns:
    dict/bool (False): A dictionary containing the token and character range of the anchor text, or False if the anchor text is not found in the context.
    '''
    total = []
    first = False
    try:
        subevent_anchor = nlp(sample_subevent["anchor_text"])   
    except:
        return False
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
        return False
    if not found:
        return False
    return sample_subevent


def process_time_information(sample_context, sample_subevent, doc, groundtruth_values, new_groundtruth):
    '''
    Extracts and processes time expressions from the sample context and updates the ground truth data.
    In the case of partial matches (e.g. the context contains the year and month but not the day), the function stores the time expression as happening on the 1st of the month.


    Parameters:
    - sample_context (str): The textual context of the sample.
    - sample_subevent (dict): Dictionary containing details of the subevent, including anchor text.
    - doc (spacy.tokens.Doc): A spaCy Document object representing the tokenized context.
    - groundtruth_values (dict): Dictionary containing ground truth values related to time expressions.
    - new_groundtruth (dict): Dictionary to which extracted time information will be added.

    Returns:
    dict: Updated new_groundtruth dictionary with time information extracted from the sample context.

    This function identifies time expressions within the sample context using date parsing techniques. It accounts for partial matches (e.g., year and month without day), and stores the time expressions in a structured format within the new_groundtruth dictionary. The function is designed to handle various time formats and align them with corresponding ground truth values.
    '''
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
            return new_groundtruth
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
            new_groundtruth[property].append([start, end, start_char, end_char, property, ttime[1]]) 
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
    return new_groundtruth


def process_entities(doc, groundtruth_values, new_groundtruth):
    '''
    Identifies and processes entities in the sample context, updating the groundtruth data with relevant entity information.

    Parameters:
    - doc (spacy.tokens.Doc): A spaCy Document object representing the tokenized context.
    - groundtruth_values (dict): Dictionary mapping entities to their ground truth values.
    - new_groundtruth (dict): The dictionary where processed entity information will be added.

    Returns:
    dict: Updated new_groundtruth dictionary with information about identified entities.

    This function processes each entity found in the doc. It uses linked entity data to find corresponding DBpedia IDs and checks these against the groundtruth_values. If a match is found, the entity information (including its text span and DBpedia ID) is added to the new_groundtruth dictionary. This function is integral for linking recognized entities in the text with their semantic identifiers in external knowledge bases.
    '''
    for ent in doc._.linkedEntities:
        qid = "Q"+str(ent.identifier)
        if qid in groundtruth_values:
            property = groundtruth_values[qid]
            if property not in new_groundtruth:
                new_groundtruth[property] = []
            new_groundtruth[property].append([ent.span.start, ent.span.end-1, ent.span.start_char, ent.span.end_char, property, ent.span.text, qid])
    return new_groundtruth

def process_literals_and_quantities(doc, groundtruth_values, sample_context, new_groundtruth):
    '''
    This function finds the literals and quantities in the context of each sample and stores them in the groundtruth dictionary if they are in the groundtruth values.
    We make sure to account for special cases such as the literal/quantity being a substring of a token or the literal/quantity being a substring of a bigger word, 
    as well as accounting for difference in formatting (e.g. 1,000 vs 1000).
    Additionally, if a sentence is counting something (e.g. "There were 5 people in the room, and 2 outside"), we add up the numbers and store 
    the sum as the value if it is in the groundtruth values.
    '''
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
            new_groundtruth[property].append([start, end, start_char, end_char, property, doc.text[start_char:end_char]]) 
    return new_groundtruth


def process_sample(sample, event_wd_id, subevent_wd_id, subevent_types, subevents_properties, groundtruth_values, nlp, new_dataset):
    '''
    Processes individual samples from a subevent to extract and structure relevant information.

    Parameters:
    - sample (tuple): A tuple containing the subevent and its context.
    - event_wd_id (str): Wikidata ID of the event.
    - subevent_wd_id (str): Wikidata ID of the subevent.
    - subevent_types (list): List of types associated with the subevent.
    - subevents_properties (dict): Dictionary of properties related to subevents.
    - groundtruth_values (dict): Ground truth values associated with properties.
    - nlp (spacy.Language): A spaCy NLP model object.
    - new_dataset (dict): The dataset being constructed with processed information.

    Returns:
    dict: Updated new_dataset with processed information from the sample.

    This function preprocesses the context of each sample, processes it using spaCy for NLP tasks, and extracts information such as anchor text ranges, entity information, and time and quantity data. The extracted information is then added to the new_dataset.
    '''
    new_groundtruth = {}
    sample_subevent = sample[0]
    if subevent_wd_id not in subevents_properties or "P31" not in subevents_properties[subevent_wd_id]:
        return new_dataset
    
    subevent_class_type = subevents_properties[subevent_wd_id]["P31"]
    sample_subevent["types"] = subevent_class_type

    sample, sample_context = preprocess_sample_context(sample, subevent_wd_id, subevents_properties, rx)
    
    if sample_context is None:
        return new_dataset

    _, doc = process_sample_nlp(sample_subevent, sample_context, nlp)
    if doc is None:
        return new_dataset

    if not process_anchor_text(sample_subevent, doc):
        return new_dataset
    else:
        sample_subevent = process_anchor_text(sample_subevent, doc)
    new_groundtruth = process_time_information(sample_context, sample_subevent, doc, groundtruth_values, new_groundtruth)
    new_groundtruth = process_entities(doc, groundtruth_values, new_groundtruth)
    new_groundtruth = process_literals_and_quantities(doc, groundtruth_values, sample_context, new_groundtruth)

    if event_wd_id not in new_dataset:
        new_dataset[event_wd_id] = {}
    if subevent not in new_dataset[event_wd_id]:
        new_dataset[event_wd_id][subevent] = []
    if new_groundtruth:
        new_sample = {"sample_event_link":sample_subevent, "untokenized_context":sample_context,"tokenized_context":[t.text for t in doc], "sample_groundtruth": new_groundtruth, "ground_event_types": subevent_types}
        new_dataset[event_wd_id][subevent].append(new_sample)
    return new_dataset


def save_data(num, new_dataset):
    '''
    This function saves the data every 100 samples then clears the new_dataset dictionary.
    '''

    if num % 100 == 0 or not new_dataset:
        with open(f"the_data/wd_data{num}.json", "w") as f:
            json.dump(new_dataset, f)
        new_dataset = {}
    return new_dataset

extract_data(dataset, instance_types, subevents_properties, nlp)


dataset = {}

for subdir, dirs, files in os.walk("the_data"):
    for file in files:
        filepath = subdir + os.sep + file
        print(filepath)
        if filepath.startswith("wd_data"):
            with open("the_data/"+file, "r") as f:
                dataset.update(json.load(f))

with open("the_data/wd__full_data.json","w") as f:
    json.dump(dataset, f)



event_class_counter = Counter()


with open("the_data/wd__full_data.json","r") as f:
    dataset = json.load(f)


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

with open("wde__clean_data.json","w") as f:
    json.dump(clean_dataset, f)

with open("wde__clean_data.json","r") as f:
    clean_dataset = json.load(f)



def filter_clean_data(filter_size=50):
    '''
    The function filters out samples with event classes that have less than filter_size samples and properties that have less than filter_size samples.
    '''
    filtered_dataset = {}
    for key in clean_dataset:
        for subkey in clean_dataset[key]:
            for sample in clean_dataset[key][subkey]:
                new_sample = copy.deepcopy(sample)
                for type in sample["sample_event_link"]["types"]:
                    if type in country_specific_event_classes + bad_event_classes:
                        new_sample["sample_event_link"]["types"].remove(type)
                if "P31" in sample["sample_groundtruth"]:
                    del new_sample["sample_groundtruth"]["P31"]
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

    with open("wde__filtered_data"+str(filter_size)+".json","w") as f:
        json.dump(filtered_dataset, f)

def second_pass(data, class_filter, prop_filter, label ="2nd"):
    '''
    Given the filtered dataset, this function filters out samples with event classes that have less than class_filter samples and properties that have less than prop_filter samples.
    '''
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

    with open("__"+label+"_pass_filtered_data.json","w") as f:
        json.dump(filtered_dataset, f)

    with open("class_counter.json", "w") as f:
        json.dump(c1,f)
    
    with open("property_counter.json", "w") as f:
        json.dump(c2,f)

filter_clean_data(100)

second_pass("wde__filtered_data100.json", class_filter = 100, prop_filter = 50, label="3rd")


with open("class_counter.json", "r") as f:
    C = json.load(f)


with open("__3rd_pass_filtered_data.json","r") as f:
    dataset = json.load(f)

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


with open("label_class.json", "r") as f:
    class_labels = json.load(f)

with open("label_prop.json", "r") as f:
    prop_labels = json.load(f)

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

with open("wde_all_event_classes.txt","w") as f:
    for line in list(all_event_classes):
        f.write(f"{line}\n")


#choose minority class
def select_minority_classes(dataset):
    '''
    Given the generated dataset, this function selects the minority class for each sample and adds it to the sample_event_link dictionary.'''
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

all_samples = select_minority_classes(dataset)

n = len(all_samples)
def third_pass(all_samples, class_counter, property_counter, class_filter = 100, prop_filter=50):
    '''
    Given selected minority classes for each sample, this function filters out samples with minority classes
    that have less than class_filter samples and properties that have less than prop_filter samples.
    '''
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


with open("../data/training/t2e/wde_unlabelled_event.schema","w") as f:
    f.write(f"{list(event_classes)}\n")
    f.write(f"{list(event_properties)}\n")
    f.write(f"{event_class_properties}\n")

event_classes = [class_labels[cl] for cl in event_classes]
event_properties = [prop_labels[prop] for prop in event_properties]
for cl in event_class_properties:
    event_class_properties[cl] = [prop_labels[prop] for prop in event_class_properties[cl]]
event_class_properties = {class_labels[cl]:v for cl,v in event_class_properties.items()}

with open("../data/training/t2e/wde_labelled_event.schema","w") as f:
    f.write(f"{list(event_classes)}\n")
    f.write(f"{list(event_properties)}\n")
    f.write(f"{event_class_properties}\n")


tmp = copy.deepcopy(all_samples)
tmp2 = {}
j = 0
for i, sample in enumerate(tmp):
    if "untokenized_context_with_links" in sample:
        context = sample["untokenized_context_with_links"]
        print(j)
        j+=1 
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

all_samples = []
for i, context in enumerate(tmp2):
    new_sample = {}
    tmp_sample = tmp2[context]
    if len(tmp_sample)==1:
        cl = tmp_sample[0]["sample_event_link"]["event_class"]
        props = {prop:v for prop, v in tmp_sample[0]["sample_groundtruth"].items()}
        new_sample["untokenized_context"] = tmp_sample[0]["untokenized_context_with_links"]
        doc = nlp(new_sample["untokenized_context"])
        new_sample["tokenized_context"]=[t.text for t in doc]
        new_sample["groundtruth_events"] = [{cl : {"sample_event_link":tmp_sample[0]["sample_event_link"], "sample_groundtruth":props}}]
        all_samples.append(new_sample)
    else:
        for i, sample in enumerate(tmp2[context]):
            cl = sample["sample_event_link"]["event_class"]
            props = {prop:v for prop, v in sample["sample_groundtruth"].items()}
            new_sample["untokenized_context"] = tmp_sample[0]["untokenized_context_with_links"]
            new_sample["tokenized_context"]=[t.text for t in doc]
            if "groundtruth_events" not in new_sample:
                new_sample["groundtruth_events"] = [{cl : {"sample_event_link":tmp_sample[0]["sample_event_link"], "sample_groundtruth":props}}]
            else:
                new_sample["groundtruth_events"].append({cl : {"sample_event_link":tmp_sample[0]["sample_event_link"], "sample_groundtruth":props}})
            all_samples.append(new_sample)

def create_baseline_training_data(all_samples, path = "../data/training"):
    '''
    The function creates appropriate data formatting for the baseline models and for our own approach and saves them in the appropriate directories.
    '''
    t2e_training_samples = []
    relation_extraction_samples = []
    dygiepp_training_samples = []
    for sample_num, sample in enumerate(all_samples):
        if sample_num % 100 == 0:
            print("%d / %d" %(sample_num, len(all_samples)))
        sentence_tokenized_context = []
        sentence_tokenized_events = []
        re_tokenized_events = []
        doc = nlp(sample["untokenized_context"])
        sync = dict()
        for sent in doc.sents:
            sentence_tokenized_context.append([])
            sentence_tokenized_events.append([])
            re_tokenized_events.append([])
            for t in sent:
                sentence_tokenized_context[-1].append(t.text)
                for event in sample["groundtruth_events"]:
                    for event_class in event:
                        if t.i == event[event_class]["sample_event_link"]["token_range"][0]:
                            sentence_tokenized_events[-1].append([[t.i, event_class]])
                            re_tokenized_events[-1].append([[t.i,event[event_class]["sample_event_link"]["token_range"][1],event_class]])
                            for t2 in sent:
                                for property in event[event_class]["sample_groundtruth"]:
                                    if isinstance(event[event_class]["sample_groundtruth"][property][0], list):
                                        if event[event_class]["sample_groundtruth"][property][0][0] == t2.i:
                                            if event_class not in sync:
                                                sync[event_class] = []
                                            sync[event_class].append(property)
                                            sentence_tokenized_events[-1][-1].append([event[event_class]["sample_groundtruth"][property][0][0],event[event_class]["sample_groundtruth"][property][0][1],property])
                                            re_tokenized_events[-1][-1].append([event[event_class]["sample_groundtruth"][property][0][0],event[event_class]["sample_groundtruth"][property][0][1],property])
                                    else:
                                        if event[event_class]["sample_groundtruth"][property][0] == t2.i:
                                            if event_class not in sync:
                                                sync[event_class] = []
                                            sync[event_class].append(property)
                                            sentence_tokenized_events[-1][-1].append([event[event_class]["sample_groundtruth"][property][0],event[event_class]["sample_groundtruth"][property][1],property])
                                            re_tokenized_events[-1][-1].append([event[event_class]["sample_groundtruth"][property][0],event[event_class]["sample_groundtruth"][property][1],property])
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

    indices = list(range(len(dygiepp_training_samples)))
    train_idx, test_idx = train_test_split(indices, test_size=0.3)
    dev_idx, test_idx = train_test_split(test_idx, test_size=0.5)
    train=[dygiepp_training_samples[i] for i in train_idx]
    test=[dygiepp_training_samples[i] for i in test_idx]
    dev=[dygiepp_training_samples[i] for i in dev_idx]

    with jsonlines.open(path+"/dygiepp/wde_eq_train.json","w") as f:
        f.write_all(train)

    with jsonlines.open(path+"/dygiepp/wde_eq_test.json","w") as f:
        f.write_all(test)

    with jsonlines.open(path+"/dygiepp/wde_eq_dev.json","w") as f:
        f.write_all(dev)

    train=[t2e_training_samples[i] for i in train_idx]
    test=[t2e_training_samples[i] for i in test_idx]
    dev=[t2e_training_samples[i] for i in dev_idx]

    with jsonlines.open(path+"/t2e/wde_eq_train.json","w") as f:
        f.write_all(train)

    with jsonlines.open(path+"/t2e/wde_eq_test.json","w") as f:
        f.write_all(test)

    with jsonlines.open(path+"/t2e/wde_eq_val.json","w") as f:
        f.write_all(dev)

    train=[relation_extraction_samples[i] for i in train_idx]
    test=[relation_extraction_samples[i] for i in test_idx]
    dev=[relation_extraction_samples[i] for i in dev_idx]

    with jsonlines.open(path+"/re/wde_eq_train.json","w") as f:
        f.write_all(train)

    with jsonlines.open(path+"/re/wde_eq_test.json","w") as f:
        f.write_all(test)

    with jsonlines.open(path+"/re/wde_eq_dev.json","w") as f:
        f.write_all(dev)

create_baseline_training_data(all_samples, path = "../data/training")

def create_sparse_full_data(tokenized_path, untokenized_path, output_path):
    '''
    Given the tokenized and untokenized data, this function creates the data in the format of the full data making sure adjust the indices and the character ranges, then save
    the data in the output path for the relation extraction model.
    '''
    data = []
    tokenized_samples = []
    untokenized_samples = []
    with open("../data/training/t2e/wde_unlabelled_event.schema","r") as f:
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
                    if role_type not in tmp[event_type]["arguments"]:
                        tmp[event_type]["arguments"][role_type] = []
                    start_index, end_index = get_correct_indices(u_sentence, [token for sent in t_sample["sentences"] for token in sent], token_start, token_end)
                    
                    arg_string = u_sentence[start_index:end_index]
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
                                answ["answer_start"] = u_sentence.index(argument_string)
                                answrs.append(answ)
                            data[-1]["paragraphs"][-1]["qas"].append({"question":question, "id":str(k), "answers":answrs, "is_impossible":False})
    return data


def get_correct_indices(detokenized_sentence, tokenized_sentence, token_start, token_end):
    minimum = 100
    start_indices = [m.start(0) for m in re.finditer(re.escape(tokenized_sentence[token_start]), detokenized_sentence)]
    end_indices = [m.end(0) for m in re.finditer(re.escape(tokenized_sentence[token_end]), detokenized_sentence)]

    for end_index in end_indices:
        for start_index in start_indices:
            if start_index > end_index:
                continue
            if end_index-start_index < minimum and end_index-start_index>=len(tokenized_sentence[token_start]):
                minimum = end_index-start_index
                best_start_index = start_index
                best_end_index = end_index
    return best_start_index, best_end_index
 


def create_training_data(path):
    '''
    Given the path to the training data, this function further processes the data and creates the training data for the relation extraction model and saves it in the appropriate directory.
    '''
    train = create_sparse_full_data(path+"re/wde_eq_train.json", path+"t2e/wde_eq_train.json", path+"re/wde_re_train.json")
    train_dataset = {"version":"v2.0", "data":train}
    with open(path+'re/wde2_sparse_re_train.json', 'w') as f:
        json.dump(train_dataset, f)
    train_dataset = {}
    train = {}

    test = create_sparse_full_data(path+"re/wde_eq_test.json", path+"t2e/wde_eq_test.json", path+"re/wde_re_test.json")
    test_dataset = {"version":"v2.0", "data":test}
    with open(path+'re/wde2_sparse_re_test.json', 'w') as f:
        json.dump(test_dataset, f)
    test_dataset={}
    test = {}

    dev = create_sparse_full_data(path+"re/wde_eq_dev.json", path+"t2e/wde_eq_val.json", path+"re/wde_re_dev.json")
    dev_dataset = {"version":"v2.0", "data":dev}
    with open(path+'re/wde2_sparse_re_dev.json', 'w') as f:
        json.dump(dev_dataset, f)
    dev_dataset = {}
    dev = {}
create_training_data("../data/training/")

