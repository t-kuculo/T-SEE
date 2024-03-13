from openai import OpenAI
import json
import ast
import spacy
import os
import traceback
from openai import AzureOpenAI
import jsonlines

# Load the SpaCy model
nlp = spacy.load("en_core_web_sm")

OPENAI_API_KEY='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'
client = OpenAI(api_key = OPENAI_API_KEY)


#client = AzureOpenAI(
#  azure_endpoint = "https://l3s-kuculo.openai.azure.com/", 
#  api_key=os.getenv("AZURE_OPENAI_KEY"),  
#  api_version="2024-02-15-preview"
#)
 
 

  #-------------------------------------------------------------------------------

def format_entities(text):
    # Process the text through SpaCy NLP pipeline
    doc = nlp(text)
    
    # Initialize an empty list to keep track of formatted tokens
    formatted_tokens = []
    last_index = 0
    
    # Iterate over the identified entities
    for ent in doc.ents:
        # Append the text before the entity
        formatted_tokens.append(text[last_index:ent.start_char])
        # Append the entity formatted with its label
        formatted_tokens.append(f"[{ent.text}]({ent.label_})")
        last_index = ent.end_char
    
    # Append any remaining text after the last entity
    formatted_tokens.append(text[last_index:])
    
    # Join all the parts together
    formatted_string = ''.join(formatted_tokens)
    
    return formatted_string

def convert_to_wikidata(extracted_events):
    converted_predictions = []
    with open("../processing/wikidata_class_labels.json", "r") as f:
        wikidata_class_labels = json.load(f)
    with open("../processing/wikidata_property_labels.json", "r") as f:
        wikidata_property_labels = json.load(f)
    
    class_label_to_id = {v: k for k, v in wikidata_class_labels.items()}
    property_label_to_id = {v: k for k, v in wikidata_property_labels.items()}


    for row in extracted_events:
        converted = {}
        if row == None:
            converted_predictions.append({})
            continue
        for cl in row:
            if cl in class_label_to_id:
                converted[class_label_to_id[cl]] = {}
                for prop in row[cl]:
                    if prop in property_label_to_id:
                        converted[class_label_to_id[cl]][property_label_to_id[prop]] = row[cl][prop]
        converted_predictions.append(converted)
    return converted_predictions

def write_error_log(error_info, mode="dbpedia"):
    with open(f"{mode}_error_tracker.jsonl", "a") as f:
        f.write(json.dumps(error_info) + "\n")

def write_subset(subset, output_file):
    with open(output_file, 'w') as f:
        for item in subset:
            f.write(json.dumps(item) + '\n')

def eval_scores(path):
    macroP = 0
    macroR = 0
    macroF1 = 0
    totalTP = 0
    totalFP = 0
    totalFN = 0
    
    with open(path,"r") as f:
        scores = json.load(f)
    # for testing, eliminate classes with only true negatives
    scores = {cl: scores[cl] for cl in scores if scores[cl]["tp"]+scores[cl]["fp"]+scores[cl]["fn"] > 0}
    scoresF = {}
    for cl in scores:
        totalTP += scores[cl]["tp"]
        totalFP += scores[cl]["fp"]
        totalFN += scores[cl]["fn"]
        if scores[cl]["tp"] == 0:
            precision = 0
            recall = 0
            f1 = 0
        else:
            precision = scores[cl]["tp"]/(scores[cl]["tp"]+scores[cl]["fp"])
            recall = scores[cl]["tp"]/(scores[cl]["tp"]+scores[cl]["fn"])
            f1 = 2*precision*recall/(precision+recall)
        scoresF[cl] = (precision,recall,f1)

    microP = totalTP/(totalTP+totalFP+0.00000000001)
    microR = totalTP/(totalTP+totalFN+0.00000000001)
    microF1 = microP*2*microR/(microP+microR+0.00000000001)
    for cl in scoresF:
        macroP += scoresF[cl][0]
        macroR += scoresF[cl][1]
        macroF1 += scoresF[cl][2]
    macroP /= len(scoresF)
    macroR /= len(scoresF)
    macroF1 /= len(scoresF) 
    #macroF1 = macroP*2*macroR/(macroP+macroR)

    print(" &","%.2f" % macroP, "&", "%.2f" % macroR, "&", "%.2f" % macroF1, "&","%.2f" % microP, "&","%.2f" % microR, "&","%.2f" % microF1, " \\")
    
def save_results(results, file_path):
    with open(file_path, 'w') as f:
        json.dump(results, f, indent=4)

def _calculate_f1_scores(data, mode="dbpedia", subset=""):
    scores = {}
    property_scores = {}
    
    version1_tp = 0
    version1_fp = 0
    version1_tn = 0
    version1_fn = 0

    for cl in all_event_types:
        scores[cl] = {"tp":0, "fp":0, "tn":0, "fn":0}

    for prop in all_properties:
        property_scores[prop] = {"tp":0, "fp":0, "tn":0, "fn":0}

    for item in data:

        ground_dict = {}
        output_dict = {}
        output_dict = item["predicted"]
        ground_dict = item["ground"]
        
        if output_dict == None:
            output_dict = {}

        for cl in ground_dict:
            for prop in ground_dict[cl]:
                ground_dict[cl][prop] = ground_dict[cl][prop][0]

        done_already = set()

        tmp_ground_dict = ground_dict.copy()
        tmp_output_dict = output_dict.copy()

        while ground_dict != {} or output_dict != {}:
            for cl in tmp_output_dict:
                if cl not in all_event_types:
                    del output_dict[cl]

            for cl in scores:
                if cl in ground_dict and cl in output_dict:
                    scores[cl]["tp"] += 1
                    del ground_dict[cl]
                    del output_dict[cl]
                elif cl in ground_dict and cl not in output_dict:
                    scores[cl]["fn"] += 1
                    del ground_dict[cl]
                elif cl not in ground_dict and cl in output_dict:
                    scores[cl]["fp"] += 1
                    del output_dict[cl]
                elif cl not in ground_dict and cl not in output_dict and cl not in done_already:
                    scores[cl]["tn"] += 1
                done_already.add(cl)

            for cl1 in tmp_ground_dict:
                for cl2 in tmp_output_dict:
                    for prop in property_scores:
                        if prop in tmp_ground_dict[cl1] and prop in tmp_output_dict[cl2]:
                            found = False
                            output_value = tmp_output_dict[cl2][prop]
                            ground_value = tmp_ground_dict[cl1][prop]

                            if output_value == None or output_value == [] or output_value == "":
                                continue

                            if isinstance(output_value, int):
                                output_value = str(output_value)
                                if output_value == ground_value:
                                    property_scores[prop]["tp"] += 1
                                    version1_tp += 1
                                else:
                                    property_scores[prop]["fp"] += 1
                                    property_scores[prop]["fn"] += 1
                                    version1_fp += 1
                                    version1_fn += 1
                                continue
                            
                            if isinstance(output_value, dict):
                                # interesting artefact
                                with open("StrangeCase.json", "a") as f:
                                    json.dump({"output": output_value, "ground": ground_value}, f)
                                continue

                            if isinstance(output_value, list):
                                for val in output_value:
                                    val = str(val)
                                    if val == ground_value or val in ground_value or ground_value in val:
                                        property_scores[prop]["tp"] += 1
                                        version1_tp += 1
                                        found = True
                                    elif val != ground_value and val not in ground_value and ground_value not in val:
                                        property_scores[prop]["fp"] += 1
                                        version1_fp += 1
                                if not found:
                                    property_scores[prop]["fn"] += 1
                                    version1_fn += 1
                                    version1_fn += 1

                            elif isinstance(output_value, str):
                                if output_value == ground_value or output_value in ground_value or ground_value in output_value:
                                    property_scores[prop]["tp"] += 1
                                    version1_tp += 1
                                elif output_value != ground_value and output_value not in ground_value and ground_value not in output_value:
                                    property_scores[prop]["fp"] += 1
                                    property_scores[prop]["fn"] += 1
                                    version1_fp += 1
                                    version1_fn += 1




                            elif ground_value == output_value or ground_value in output_value or output_value in ground_value:
                                property_scores[prop]["tp"] += 1
                                version1_tp += 1
                            else:
                                property_scores[prop]["fp"] += 1
                                property_scores[prop]["fn"] += 1
                                version1_fp += 1
                                version1_fn += 1

                        elif prop not in tmp_ground_dict[cl1] and prop in tmp_output_dict[cl2]:
                            output_value = tmp_output_dict[cl2][prop]
                            if output_value == None or output_value == [] or output_value == "":
                                continue

                            if isinstance(tmp_output_dict[cl2][prop], list):
                                property_scores[prop]["fp"] += len([i for i in tmp_output_dict[cl2][prop] if i and i!= "" and i != None and i != []])
                            else:
                                property_scores[prop]["fp"] += 1
                            version1_fp += 1

                        elif prop in tmp_ground_dict[cl1] and prop not in tmp_output_dict[cl2]:
                            property_scores[prop]["fn"] += 1
                            version1_fn += 1

                        elif prop not in tmp_ground_dict[cl1] and prop not in tmp_output_dict[cl2]:
                            property_scores[prop]["tn"] += 1

            # Clean up matched event classes
            for cl in done_already:
                if cl in tmp_ground_dict:
                    del tmp_ground_dict[cl]
                if cl in tmp_output_dict:
                    del tmp_output_dict[cl]

        property_r = version1_tp / (version1_tp + version1_fn+0.00000000001)
        property_p = version1_tp / (version1_tp + version1_fp+0.00000000001)
        prop_f1 = (2 * (property_p * property_r) / (property_p + property_r+0.00000000001))

        with open(f"{mode}_scores.json", "w") as f:
            json.dump(scores, f)   
        with open(f"{mode}_property_scores.json", "w") as f:
            json.dump(property_scores, f)


def calculate_f1_scores(data, mode="dbpedia", subset=""):
    scores = {cl: {"tp": 0, "fp": 0, "tn": 0, "fn": 0} for cl in all_event_types}
    property_scores = {prop: {"tp": 0, "fp": 0, "tn": 0, "fn": 0} for prop in all_properties}
    d = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}  # Aggregate counts for all properties

    def is_match(ground_val, predicted_val):
        if ground_val is None or predicted_val is None:
            return False  # Cannot have a match if one of the values is None
        if isinstance(ground_val, list):
            return any(str(gv) in predicted_val or predicted_val in str(gv) for gv in ground_val)
        if isinstance(predicted_val, list):
            return any(ground_val in predicted_val or predicted_val in str(ground_val))
        return str(ground_val) in predicted_val or predicted_val in str(ground_val)

    def process_property(ground_val, predicted_val, prop, d):
        # Function to check if the predicted value is effectively empty
        def is_effectively_empty(val):
            if val in [None, "", [], "unknown"]:
                return True
            if isinstance(val, list):
                return all(is_effectively_empty(item) for item in val)
            return False

        # If predicted value is effectively empty, treat it as no prediction
        if is_effectively_empty(predicted_val):
            if ground_val not in [None, "", []]:  # Ground truth exists but effective prediction is empty
                property_scores[prop]["fn"] += 1
                d["fn"] += 1
            return d

        matched = False
        if isinstance(predicted_val, list):
            for pv in predicted_val:
                if pv not in [None, "", []] and is_match(ground_val, str(pv)):
                    property_scores[prop]["tp"] += 1
                    d["tp"] += 1
                    matched = True
                    break  # Match found, no need to check other values
            if not matched and ground_val not in [None, "", []]:
                property_scores[prop]["fn"] += 1
                d["fn"] += 1
            extra_fp = len(predicted_val) - (1 if matched else 0)
            property_scores[prop]["fp"] += extra_fp
            d["fp"] += extra_fp
        else:
            predicted_val = str(predicted_val)  # Ensure predicted_val is a string for comparison
            if is_match(ground_val, predicted_val):
                property_scores[prop]["tp"] += 1
                d["tp"] += 1
            else:
                property_scores[prop]["fp"] += 1
                d["fp"] += 1
                if ground_val not in [None, "", []]:
                    property_scores[prop]["fn"] += 1
                    d["fn"] += 1
        return d


    for item in data:
        output_dict = item.get("predicted", {})
        ground_dict = item["ground"]

        for cl in set(ground_dict.keys()) | set(output_dict.keys()):
            ground_props = ground_dict.get(cl, {})
            output_props = output_dict.get(cl, {})
            if cl in ground_dict and cl in output_dict:
                scores[cl]["tp"] += 1
            elif cl in ground_dict:
                scores[cl]["fn"] += 1
            elif cl in output_dict:
                scores[cl]["fp"] += 1

            for prop in set(ground_props.keys()) | set(output_props.keys()):
                ground_val = ground_props.get(prop, [None])[0]
                predicted_val = output_props.get(prop)
                d = process_property(ground_val, predicted_val, prop, d)

    with open(f"{mode}_scores.json", "w") as f:
        json.dump(scores, f)
    with open(f"{mode}_property_scores.json", "w") as f:
        json.dump(property_scores, f)

    return d
def write_prompt_and_output(prompt, output, mode="dbpedia"):
    with open(f"{mode}_prompt_and_output.jsonl", "a") as f:
        f.write(json.dumps({"prompt": prompt, "output": output}) + "\n")

def classify_events(sentence, event_classes, run_count, mode="dbpedia"):
    if mode == "wikidata":
        with open("../processing/wikidata_class_labels.json", "r") as f:
            wikidata_class_labels = json.load(f)
        event_classes = [wikidata_class_labels[cls] for cls in event_classes]
    messages = [
        {
            "role": "system",
            "content": f"""
            Your task is to analyze the sentence and classify events that are in the sentence. 
            An event is identified by an action or a mention of an event.
            You will only consider events that are likely to have their own Wikipedia page. 

            For example, in the sentence "John married Mary in Paris on 12th December 2019 just before the German-French War started.", 
            the events are "Marriage" and "Military Conflict". However, "Marriage" is not to be considered an event as it is unlikely to have its own Wikipedia page.

            You must select event types from the following list of event types and return it formatted as a list of strings of event types:
            {event_classes} 

            Note: The events that you should identify are links in Wikipedia, they may not be referred to directly by name in the sentence 
            but a specific word or phrase in the sentence may link to the event. E.g. in "Senator McCain also got 10% higher approval rating compared to 2010", 2010 is a link to the event "United States Senate elections, 2010" even though it is not mentioned directly in the sentence.
            """
        },
        {
            "role": "user",
            "content": sentence
        }
    ]
   
    
    with open(f"{mode}_event_extraction_prompts.json", "a") as f:
        f.write(json.dumps(messages))
        f.write("\n")
    
    #completion = client.chat.completions.create(
        #model="gpt-3.5-turbo",
        #messages=messages
    #)
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages = messages,
        temperature=0.7,
        max_tokens=800,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None)

    #completion = client.chat.completions.create(
    #    model="gpt-35-turbo", 
    #    messages = messages,
    #    temperature=0.7,
    #    max_tokens=800,
    #    top_p=0.95,
    #    frequency_penalty=0,
    #    presence_penalty=0,
    #    stop=None)
    
    write_prompt_and_output(sentence, completion.choices[0].message.content, mode)
    
    # Assuming the output is a well-formed JSON string or a JSON-convertible structure
    try:
        event_classes = ast.literal_eval(completion.choices[0].message.content)
        return event_classes
    except Exception as e:
        print("The output could not be parsed as JSON. Please check the format.")
        write_error_log({"type": "classify_events", "prompt": sentence, "The output could not be parsed as JSON. Please check the format. Error": str(e),"LLM output":completion.choices[0].message.content, "run_count": run_count}, mode)
        return None

def extract_properties(sentence, event_classes, event_schema, run_count, mode="dbpedia"):
    if mode == "dbpedia":
        event_properties = {event: [event_schema[event]] for event in event_classes}
    elif mode == "wikidata":
        with open("../processing/wikidata_class_labels.json", "r") as f:
            wikidata_class_labels = json.load(f)
        class_label_to_id = {v: k for k, v in wikidata_class_labels.items()}

        with open("../processing/wikidata_property_labels.json", "r") as f:
            wikidata_property_labels = json.load(f)
        
        event_properties = {}
        for event in event_classes:
            if event not in class_label_to_id:
                print(f"{event} not in class_label_to_id")
                continue
            else:
                event_properties[event] = [wikidata_property_labels[prop] for prop in event_schema[class_label_to_id[event]]]
            

    messages = [
        {
            "role": "system",
            "content": f"""
                Your task is to extract the properties of the events that are in a given sentence. You will only consider properties that are likely to be associated with the given event classes. Extract the properties of the events and return a JSON object with the event classes as the keys and the properties as the value.
                The property values can be dates, entities, or quantities. If there is no specific value for a property, you must not include it in the JSON object.
                The extracted property values must fit their respective property types. 
                For example, if the property is "date", the value must be able to be formatted as a date (e.g. "12th December 2019" or "2019" in the case of a year).
                Similarly, if the property is "location", the value must be a location.
                If there are multiple values for a property, you must include all the values in a list.


                Consider the following example:

                Sentence:
                John married Mary the first day of the start of the COVID-19 pandemic, on 12th December 2019.
                It was only a few days later that in the winter of 2019, German-French War destroyed the cities of Paris and Berlin.

                Event classes and their potential properties:
                - Pandemic: city, time
                - Military Conflict: city, date, participant

                Output:
                {{
                    "Pandemic": {{
                        "startDate": ["12th December 2019"],
                    }},
                    "Military Conflict": {{
                        "city": ["Paris", "Berlin"]
                        "date": ["2019"]
                    }}
                }}

                This is your task:

                Sentence:
                {sentence}

                Event classes and their potential properties:
                {event_properties}


            """
        },
        {
            "role": "user",
            "content": sentence
        }
    ]

    with open(f"{mode}_arg_extr_prompts.json", "a") as f:
        f.write(json.dumps(messages))
        f.write("\n")

    #completion = client.chat.completions.create(
    #    model="gpt-35-turbo", 
    #    messages = messages,
    #    temperature=0.7,
    #    max_tokens=800,
    #    top_p=0.95,
    #    frequency_penalty=0,
    #    presence_penalty=0,
    #    stop=None)
        
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages = messages,
        temperature=0.7,
        max_tokens=800,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None)
    
    write_prompt_and_output(sentence, completion.choices[0].message.content, mode)
    
    # Assuming the output is a well-formed JSON string or a JSON-convertible structure
    try:
        print(completion.choices[0].message.content)
        event_properties = ast.literal_eval(completion.choices[0].message.content)
        return event_properties
    except Exception as e:
        print("The output could not be parsed as JSON. Please check the format.")
        write_error_log({"type": "extract_properties", "prompt": sentence, "The output could not be parsed as JSON. Please check the format. Error": str(e),"LLM output":completion.choices[0].message.content, "run_count": run_count}, mode)
        return None

def read_dataset(file_path):
    with open(file_path, 'r') as f:
        for line in f:
            yield json.loads(line)


def format_data(structure, tokens):
    formatted_result = {}

    for sentence in structure:  # Iterate through sentences
        for event in sentence:  # Iterate through events in a sentence
            if not event:  # Skip if the event list is empty
                continue

            event_class_info = event[0]  # First list in an event is the class info
            start_idx, end_idx, event_class = event_class_info
            event_class_value = " ".join(tokens[start_idx:end_idx + 1])

            if event_class not in formatted_result:
                formatted_result[event_class] = {}

            for property_info in event[1:]:  # Iterate through properties in an event
                if len(property_info) == 5:
                    prop_start_idx, prop_end_idx, property_type, property_value, entity_link = property_info
                else:
                    prop_start_idx, prop_end_idx, property_type, property_value = property_info
  
                if property_type in formatted_result[event_class]:
                    formatted_result[event_class][property_type].append(property_value)
                else:
                    formatted_result[event_class][property_type] = [property_value]

    return formatted_result

def load_data(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)
    return data


def step_wise_process_and_evaluate(dataset_file, output_file, all_event_types, event_role_types, mode="dbpedia"):
    state_file = f"{mode}_processing_state.json"
    error_count = 0
    class_error_count = 0
    property_error_count = 0
    run_count = 0
    predictions = []

    if mode == "wikidata":
        with open("../processing/wikidata_class_labels.json", "r") as f:
            wikidata_class_labels = json.load(f)
        class_label_to_id = {v: k for k, v in wikidata_class_labels.items()}

        with open("../processing/wikidata_property_labels.json", "r") as f:
            wikidata_property_labels = json.load(f)
        property_label_to_id = {v: k for k, v in wikidata_property_labels.items()}
        


    ground_truth = list(read_dataset(dataset_file))#[len(predictions):]

    for item in ground_truth:
        try:
            sentence = " ".join(word for sentence in item['sentences'] for word in sentence)
            ground = item["events"]
            run_count += 1

            extracted_event_classes = classify_events(sentence, all_event_types, run_count, mode)

            if extracted_event_classes == None:
                write_error_log({"type": "classify_events", "error": "No classes extracted", "ground": ground, "run_count": run_count}, mode)
                class_error_count += 1
                error_count += 1
                continue
            # Check the classes are in the schema
            cleaned_event_classes = []
            if mode == "dbpedia":
                for event_class in extracted_event_classes:
                    if event_class not in all_event_types:
                        write_error_log({"type": "unexpected_class", "error": f"{event_class} not in schema", "ground": ground, "run_count": run_count}, mode)
                        error_count += 1
                        class_error_count += 1
                    else:
                        cleaned_event_classes.append(event_class)
            elif mode == "wikidata":
                for event_class in extracted_event_classes:
                    if event_class not in class_label_to_id:
                        write_error_log({"type": "unexpected_class", "error": f"{event_class} not in schema", "ground": ground, "run_count": run_count}, mode)
                        error_count += 1
                        class_error_count += 1
                    else:
                        cleaned_event_classes.append(event_class)
            extracted_event_classes = cleaned_event_classes
                        
            # If None is returned, log the error and continue
            if extracted_event_classes == []:
                write_error_log({"type": "classify_events", "error": "No valid classes extracted", "ground": ground, "run_count": run_count}, mode)
                class_error_count += 1
                error_count += 1
                continue


            extracted_events = extract_properties(sentence, extracted_event_classes, event_role_types, run_count, mode)
            if extracted_events == None:
                write_error_log({"type": "extract_properties", "error": "No properties extracted", "ground": ground, "run_count": run_count}, mode)
                property_error_count += 1
                error_count += 1
                continue
            # Check the properties are in the schema
            cleaned_extracted_events = {}
            if mode == "dbpedia":
                for event in extracted_events:
                    if event not in cleaned_extracted_events:
                        cleaned_extracted_events[event] = {}
                    for prop in extracted_events[event]:
                        if prop not in event_role_types[event]:
                            write_error_log({"type": "unexpected_property", "error": f"{prop} not in schema, {event_role_types[event]}", "run_count": run_count}, mode)
                            error_count += 1
                            property_error_count += 1
                        else:
                            cleaned_extracted_events[event][prop] = extracted_events[event][prop]
                extracted_events = cleaned_extracted_events
            elif mode == "wikidata":
                for event in extracted_events:
                    if event not in class_label_to_id:
                        write_error_log({"type": "unexpected_class", "error": f"{event} not in schema, {class_label_to_id}", "ground": ground, "run_count": run_count}, mode)
                        error_count += 1
                        class_error_count += 1
                    else:
                        cleaned_extracted_events[class_label_to_id[event]] = {}
                        for prop in extracted_events[event]:
                            if prop not in property_label_to_id:
                                write_error_log({"type": "unexpected_property", "error": f"{prop} not in schema, {property_label_to_id}", "run_count": run_count}, mode)
                                error_count += 1
                                property_error_count += 1
                            elif prop in property_label_to_id and property_label_to_id[prop] not in event_role_types[class_label_to_id[event]]:
                                write_error_log({"type": "unexpected_property", "error": f"{prop} not in schema, doesn't fit the event, {event_role_types[class_label_to_id[event]]}", "run_count": run_count}, mode)
                                error_count += 1
                                property_error_count += 1
                            else:
                                cleaned_extracted_events[class_label_to_id[event]][property_label_to_id[prop]] = extracted_events[event][prop]

            extracted_events = cleaned_extracted_events
            
            tokens = sentence.split()
            ground =  format_data(ground, tokens)
            #if mode == "wikidata" and extracted_events is not None:
                #extracted_events = convert_to_wikidata(extracted_events)

            predictions.append({
                "sentence": sentence,
                "predicted": extracted_events,
                "ground": ground
            })

        except Exception as e:
            traceback.print_exc()
            write_error_log({"type": "unexpected_error", "error": str(e), "run_count": run_count}, mode)

        finally:
            # Save state periodically
            with open(state_file, 'w') as f:
                f.write(json.dumps({"total_error_count": error_count, "class_error_count":class_error_count, "property_error_count":property_error_count,"run_count": run_count}) + "\n")
                for item in predictions:
                    f.write(json.dumps(item) + "\n")

    save_results(predictions, output_file)

if __name__ == "__main__":
    
    mode = "dbpedia"
    data_path = "/home/kuculo/T-SEE/data/training/with_entities/"
    dataset = f"{data_path}dbpe_eq_test.json"
    schema_path = f"/home/kuculo/T-SEE/processing/filtered_{mode}_event2.schema"
    with open(schema_path,"r") as f:
        lines = [line.rstrip() for line in f]

    all_event_types = ast.literal_eval(lines[0])
    all_properties = ast.literal_eval(lines[1])
    all_event_role_types = ast.literal_eval(lines[2])


    #step_wise_process_and_evaluate(dataset, f'{mode}_output.json', all_event_types, all_event_role_types, mode)

    # Load the data from the JSON file
    #data = load_data(f'{mode}_output.json')
    #data = load_data('output_prefix_temporal_event_distribution.json')
    for path in ['output_prefix_temporal_event_distribution.json', 'output_prefix_complex_sentence_structures.json',
                 'output_prefix_geographical_diversity.json', 'output_prefix_named_entity_diversity.json',
                 'output_prefix_semantic_diversity.json', 'output_prefix_sentence_length.json']:
        data = load_data(path)


        # Calculate F1 scores
        calculate_f1_scores(data, mode)

        print(path)

        eval_scores(f"{mode}_scores.json")
        eval_scores(f"{mode}_property_scores.json")


