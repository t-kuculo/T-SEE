import json
import csv 
import pandas as pd
import numpy as np
from collections import Counter
import json
import csv 
import jsonlines
from collections import Counter
import ast
import copy
from rem import run_relation_extraction

def predict(mode, extra="_with_focal_loss", training_set="training"):
    '''
    Run the relation extraction model on predicted classes and properties from the MLC model.

    This function processes the output from the MLC model, applies the relation extraction model, and saves the results in a JSON file. 

    Parameters:
    - mode (str): The mode of operation, which dictates the dataset and processing strategy.
    - extra (str, optional): Additional specifier for file naming, defaults to "_with_focal_loss", the currently used loss function for the MLC model.
    - training_set (str, optional): The name of the training set directory, defaults to "training".

    Returns:
    None; outputs are written to a file.

    Raises:
    - FileNotFoundError: If specified data files are not found.
    - JSONDecodeError: If input JSON files are improperly formatted.

    Note:
    - Ensure that the file paths and directory structure conform to the expected format.
    - This function is part of the post-processing workflow following MLC model output.
    '''
    all_ground_samples=[]
    path_to_ground_data = "../data/"+training_set+"/t2e/"+mode+"_eq_test.json"
    x= 0
    write_samples = []

    classes = list(pd.read_csv("../data/"+training_set+"/mlc_data/"+mode+"_multilabel_test.csv").columns)[2:]
    with open("../data/"+training_set+"/t2e/"+mode+"_unlabelled_event.schema","r") as f:
        for o, line in enumerate(f):
            if o ==2:
                classprop =  ast.literal_eval(line)

    with open(path_to_ground_data, 'r') as json_file:
        json_list = list(json_file)
    
    for json_str in json_list:
        all_ground_samples.append(json.loads(json_str))

    with open("minority_classes/mlc_output/"+mode+extra+".csv", "r") as f, jsonlines.open("../data/"+training_set+"/dygiepp/"+mode+"_eq_test.json","r") as f2:
        output_reader = csv.reader(f, delimiter="\t")
        ground = [o for o in f2]
        for i, (output_row, sample) in enumerate(zip(output_reader, all_ground_samples)):
            ground = get_ground_dict_from_sample(sample, mode)
            queries = []
            output = {}
            context = output_row[1]
            if i%100==0:
                print("%d out of %d"%(i, len(all_ground_samples)))
            predicted_classes = [cl for cl, prediction in zip(classes, output_row[2:]) if prediction!='False'] 
            for cl in predicted_classes:
                output[cl] = {}
                for prop in classprop[cl]:
                    queries.append(cl+","+prop)
            results = run_relation_extraction(queries, context)

            if not any(results):
                x+=1
            tmp1 = []
            tmp2 = []
            # sort queries and results by probability
            for res, query in zip(results, queries):
                if res:
                    tmp1.append(res)
                    tmp2.append(query)
                    
            results = tmp1
            results = [(ordered_dict[0]["text"], ordered_dict[0]["probability"]) for ordered_dict in results]
            queries = tmp2

            queries = [i for _,i in sorted(zip(results,queries), key=lambda x: x[0][1], reverse=True)]

            results = sorted(results, key=lambda x: x[1], reverse=True)


            results, queries = [answ for answ in results if answ[0]!="empty"],[q for q, answ in zip(queries, results) if answ[0]!="empty"]
            for res_answer, query in zip(results, queries):
                cl = query.split(",")[0]
                prop = query.split(",")[1]
                answer = res_answer[0]
                output[cl][prop] = answer
                
            output = {key:value for key,value in output.items() if key!=None}
            preserve_output = copy.deepcopy(output)
            preserve_ground = copy.deepcopy(ground)
            write_sample={"text":context, "predicted":preserve_output, "ground":preserve_ground}
            write_samples.append(write_sample)

    with jsonlines.open("minority_classes/mlc_output/"+mode+extra+"_predictions_october.json","w") as f:
        f.write_all(write_samples)

def get_ground_dict_from_sample(sample, data="dbp"):
    '''
    Extracts the ground truth classes and properties from a given sample.

    Parameters:
    - sample (dict): A dictionary representing a sample from the test set, containing event data.
    - data (str, optional): Specifies the format or source of the data. Defaults to "dbp" for DBpedia, change to "wde" for Wikidata.

    Returns:
    - dict: A dictionary where keys are ground event types and values are dictionaries of properties and their corresponding values.

    This function parses the 'event' field in the sample, splits it into components, and organizes them into a structured dictionary.
    '''
    groundtruth = sample["event"]
    groundtruth_events = groundtruth.split("<extra_id_1><extra_id_1><extra_id_0>")
    ground_dict = {}
    for en, event in enumerate(groundtruth_events):
        if en==0:
            the_split = event.split("<extra_id_0>")[2:]
            groundtruth_tokenized = [i.replace("<extra_id_1>","") if "<extra_id_1>" in i else i for i in the_split]
        else:
            the_split = event.split("<extra_id_0>")
            groundtruth_tokenized = [i.replace("<extra_id_1>","") if "<extra_id_1>" in i else i for i in the_split]
        if data=="wde":
            for token in groundtruth_tokenized:
                if  token[0]=="Q":#token[0].isupper():# for wikidata token[0]=="Q":
                    ground_event_type = token.split()[0]
                if ground_event_type not in ground_dict:
                    ground_dict[ground_event_type] = {}
                if token[0] == "P":#not token[0].isupper():# for wikidata token[0] == "P":
                    argument_type = token.split()[0]
                    argument_string = " ".join(token.split()[1:])
                    if argument_type not in ground_dict[ground_event_type]:
                        ground_dict[ground_event_type][argument_type] = []
                    ground_dict[ground_event_type][argument_type].append(argument_string)
        else:
            for token in groundtruth_tokenized:
                if  token[0].isupper():# for wikidata token[0]=="Q":
                    ground_event_type = token.split()[0]
                if ground_event_type not in ground_dict:
                    ground_dict[ground_event_type] = {}
                if not token[0].isupper():# for wikidata token[0] == "P":
                    argument_type = token.split()[0]
                    argument_string = " ".join(token.split()[1:])
                    if argument_type not in ground_dict[ground_event_type]:
                        ground_dict[ground_event_type][argument_type] = []
                    ground_dict[ground_event_type][argument_type].append(argument_string)
    return ground_dict

def eval_scores(path, Approach="Text2Event"):
    '''
    Calculates and prints the micro and macro precision, recall, and F1 scores from a JSON file containing model output.

    Parameters:
    - path (str): Path to the JSON file containing the scores data.
    - Approach (str, optional): Name of the approach or model for display purposes. Defaults to "Text2Event".

    Returns:
    None; this function prints the calculated scores.

    The function reads the model output from the specified file and computes various evaluation metrics, which are then printed.
    '''
    macroP = 0
    macroR = 0
    macroF1 = 0
    totalTP = 0
    totalFP = 0
    totalFN = 0
    
    with open(path,"r") as f:
        scores = json.load(f)
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

    microP = totalTP/(totalTP+totalFP)
    microR = totalTP/(totalTP+totalFN)
    microF1 = microP*2*microR/(microP+microR)
    for cl in scoresF:
        macroP += scoresF[cl][0]
        macroR += scoresF[cl][1]
        macroF1 += scoresF[cl][2]
    macroP /= len(scoresF)
    macroR /= len(scoresF)
    macroF1 /= len(scoresF) 
    #macroF1 = macroP*2*macroR/(macroP+macroR)

    print(Approach+" &","%.2f" % macroP, "&", "%.2f" % macroR, "&", "%.2f" % macroF1, "&","%.2f" % microP, "&","%.2f" % microR, "&","%.2f" % microF1, " \\")

def load_t2e(sample, mode):
    '''
    Processes a test set sample and returns ground truth and output dictionaries for  Text2Event data format.

    Parameters:
    - sample (dict): A sample from the test set.
    - mode (str): The mode specifying the data format or type.

    Returns:
    - tuple: A tuple containing two dictionaries, the first is the ground truth dictionary and the second is the output dictionary.

    This function is specifically designed for handling Text2Event format data and extracts event types and roles from the sample.
    '''
    ground_dict = get_ground_dict_from_sample(sample, mode)
    output = sample["t2e"][0]
    output_dict = {}
    for event in output:
        cl = event["type"]
        if cl not in output_dict:
            output_dict[cl] = {}
        for role in event["roles"]:
            property = role[1]
            arg_string = role[2]
            if property not in output_dict[cl]:
                output_dict[cl][property] = arg_string
    return ground_dict, output_dict

def load_dygiepp(sample):
    '''
    Processes a test set sample and returns ground truth and output dictionaries for the DyGIE++ data format.

    Parameters:
    - sample (dict): A sample from the test set, specifically formatted for the DyGIE++ model.

    Returns:
    - tuple: A tuple containing two dictionaries, the first is the ground truth dictionary and the second is the output dictionary.

    This function processes the 'events' and 'predicted_events' from the sample to construct the respective dictionaries.
    '''
    groundtruth = sample["events"]
    ground_dict = {}
    for gg, sentence_events in enumerate(groundtruth):
        for event in sentence_events:
            for event_part in event:
                if len(event_part) == 2:
                    event_type = event_part[1]
                    ground_dict[event_type] = {}
                elif event_part[2] not in ground_dict[event_type]:
                    ground_dict[event_type][event_part[2]]=[]
                    start = event_part[0]
                    end = event_part[1]
                    merged_sentence = [token for sentence in sample["sentences"] for token in sentence]
                    ground_dict[event_type][event_part[2]].append(merged_sentence[start:end+1])

    output = sample["predicted_events"]
    output_dict = {}
    for gg, sentence_events in enumerate(output):
        for event in sentence_events:
            for event_part in event:
                if len(event_part) == 4:
                    event_type = event_part[1]
                    output_dict[event_type] = {}
                elif event_part[2] not in output_dict[event_type]:
                    output_dict[event_type][event_part[2]]=[]
                    start = event_part[0]
                    end = event_part[1]
                    merged_sentence = [token for sentence in sample["sentences"] for token in sentence]
                    output_dict[event_type][event_part[2]].append(" ".join(merged_sentence[start:end+1]))
    
    new_ground_dict =copy.deepcopy(ground_dict)
    new_output_dict = copy.deepcopy(output_dict)

    for cl in ground_dict:
        for prop in ground_dict[cl]:
            new_ground_dict[cl][prop] = [" ".join(ground_dict[cl][prop][0])]

    for cl in output_dict:
        for prop in output_dict[cl]:
            new_output_dict[cl][prop] = output_dict[cl][prop][0]
    
    return new_ground_dict, new_output_dict

def end_to_end_eval(path_to_output_data, mode):
    '''
    Performs an end-to-end evaluation for the baselines and T-SEE. Given their output  calculates precision, recall, and F1 scores for classes and properties.

    Parameters:
    - path_to_output_data (str): Path to the JSON file containing the output data.
    - mode (str): The mode specifying the data format or type.

    Returns:
    None; this function saves the calculated scores in a JSON file.

    This function computes scores for each class and property by comparing the ground truth with the predicted output. It handles different data formats based on the specified mode.
    '''
    scores = {}
    property_scores = {}
    for cl in all_event_types:
        scores[cl] = {"tp":0,"fp":0,"tn":0,"fn":0}
    for prop in all_properties:
        property_scores[prop] = {"tp":0,"fp":0,"tn":0,"fn":0}

    version1_tp = 0
    version1_fp = 0
    version1_tn = 0
    version1_fn = 0

    all_output_samples = []
    with open(path_to_output_data, 'r') as json_file:
        json_list = list(json_file)
    
    for json_str in json_list:
        all_output_samples.append(json.loads(json_str))
 
    # for each sample in the test set grab the ground truth and the predicted classes and properties
    for i, sample in enumerate(all_output_samples):
        if "t2e" in path_to_output_data:
            ground_dict, output_dict = load_t2e(sample, mode)
        elif "dygiepp" in path_to_output_data:
            ground_dict, output_dict = load_dygiepp(sample)
        else:
            ground_dict, output_dict = sample["ground"], sample["predicted"]
            for cl in ground_dict:
                for prop in ground_dict[cl]:
                    ground_dict[cl][prop] = [ground_dict[cl][prop]]
        # for each class in the ground truth and the predicted classes and properties compute the scores
        # if the class is not in the ground truth or the predicted classes and properties, add a true negative
        # if the class is in the ground truth but not in the predicted classes and properties, add a false negative
        # if the class is in the predicted classes and properties but not in the ground truth, add a false positive
        # if the class is in the predicted classes and properties and in the ground truth, add a true positive
        # if the class is in the ground truth and in the predicted classes and properties, compute the scores for the properties in a similar fashion
        done_already = []
        while ground_dict != {} or output_dict != {}:
            for cl in scores:
                if cl in ground_dict and cl in output_dict:
                    scores[cl]["tp"] +=1
                    for property in class_prop[cl]:
                        if property in ground_dict[cl] and property in output_dict[cl]:
                            output_value = output_dict[cl][property]
                            ground_value = ground_dict[cl][property][0]
                            while isinstance(output_value, list):
                                output_value = output_value[0]
                            while isinstance(ground_value, list):
                                ground_value = ground_value[0]
                            
                            if ground_value==output_value or ground_value in output_value or output_value in ground_value:
                                property_scores[property]["tp"]+=1
                                version1_tp += 1
                            else:
                                property_scores[property]["fp"]+=1
                                property_scores[property]["fn"]+=1
                                version1_fp += 1
                                version1_fn += 1
                        elif property not in ground_dict[cl] and property in output_dict[cl]:
                            property_scores[property]["fp"]+=1
                            version1_fp += 1
                        elif property in ground_dict[cl] and property not in output_dict[cl]:
                            property_scores[property]["fn"]+=1
                            version1_fn += 1
                        elif property not in ground_dict[cl] and property not in output_dict[cl]:
                            property_scores[property]["tn"]+=1
                            version1_tn += 1
                        if property in ground_dict[cl]:
                            del ground_dict[cl][property]
                        if property in output_dict[cl]:
                            del output_dict[cl][property]
                elif cl in ground_dict and cl not in output_dict:
                    scores[cl]["fn"] +=1
                elif cl not in ground_dict and cl in output_dict:
                    scores[cl]["fp"] +=1
                elif cl not in ground_dict and cl not in output_dict and cl not in done_already:
                    scores[cl]["tn"] +=1
                done_already.append(cl)
                if cl in ground_dict :
                    del ground_dict[cl]
                if cl in output_dict:
                    del output_dict[cl]   
    property_r = version1_tp/(version1_tp+version1_fn)
    property_p = version1_tp/(version1_tp+version1_fp)
    prop_f1 = (2 * (property_p * property_r) / (property_p + property_r))

    with open(mode+"_"+"_".join(path_to_output_data.split("/")[1].split("_")[:-1])+"_scores.json","w") as f:
        json.dump(scores, f)   
    with open(mode+"_"+"_".join(path_to_output_data.split("/")[1].split("_")[:-1])+"_property_scores.json","w") as f:
        json.dump(property_scores, f)
   
if __name__=="__main__":
    
    training_set = "training"
    mode = "dbpe"
    extra = "_with_focal_loss"
    with open("../data/"+training_set+"/t2e/"+mode+"_unlabelled_event.schema","r") as f:
        lines = [line.rstrip() for line in f]
    all_event_types = ast.literal_eval(lines[0])
    all_properties = ast.literal_eval(lines[1])
    all_event_role_types = ast.literal_eval(lines[1])
    class_prop = ast.literal_eval(lines[2])
    

    predict(mode,extra,training_set)
    print("classes:")
    end_to_end_eval("minority_classes/mlc_output/"+mode+extra+"_predictions.json", mode)
    #end_to_end_eval("minority_classes/t2e_output/"+mode+"_test_output_30.json", mode)
    #end_to_end_eval("minority_classes/dygiepp_output/"+mode+"_output30.json", mode)
    
    eval_scores(mode+"_mlc_scores.json", "Our")
    #eval_scores(mode+"_t2e_scores.json")
    #eval_scores(mode+"_dygiepp_scores.json", "Dyggiepp")

    print("properties:")
    eval_scores(mode+"_mlc_property_scores.json", "Our")
    #eval_scores(mode+"_t2e_property_scores.json")
    #eval_scores(mode+"_dygiepp_property_scores.json", "Dyggiepp")




    



