import json
import jsonlines 

train = []
test = []
dev = []
wd_NER_per_property = {'P541': 'OFF', 'P1158': 'LOC', 'P621': 'TIM', 'P991': 'PER', 'P641': 'SPOR', 'P138': 'ENT', 'P3450': 'SEA', 'P397': 'PAB', 'P279': 'TYP', 'P793': 'ENT', 'P156': 'EVT', 'P576': 'TIM', 'P533': 'ENT', 'P2922': 'TIM', 'P1619': 'TIM', 'P585': 'TIM', 'P527': 'EVT', 'P1132': 'NUM', 'P1876': 'VEH', 'P361': 'EVT', 'P726': 'ENT', 'P1001': 'LOC', 'P3999': 'TIM', 'P1029': 'PER', 'P580': 'TIM', 'P1346': 'ENT', 'P1269': 'EVT', 'P710': 'ENT', 'P2257': 'TIM', 'P137': 'ENT', 'P276': 'LOC', 'P179': 'EVT', 'P155': 'EVT', 'P393': 'NUM', 'P1444': 'LOC', 'P571': 'TIM', 'P664': 'ENT', 'P1813': 'LAB', 'P620': 'TIM', 'P1889': 'ENT', 'P765': 'SUR', 'P619': 'TIM', 'P17': 'CNT', 'P199': 'DIV', 'P582': 'TIM', 'P4794': 'TIM', 'P131': 'LOC', 'P1923': 'TEA'}
NER_per_property  = {'commander':"PER","isPartOfMilitaryConflict":"ENT","territory":"LOC","date":"TIM","place":"LOC",\
    "city":"LOC","poleDriver":"PER","event":"EVT","country":"CNT","team":"ORG","location":"LOC","firstDriver":"PER","secondDriver":"PER",\
        "startDate":"TIM","secondLeader":"PER","firstLeader":"PER","affiliation":"ORG", "disease":"DIS","goldMedalist":"PER",\
            "previousMission":"EVT","launchDate":"TIM","manufacturer":"ORG","crewMember":"PER","nextMission":"EVT",\
                "genre":"GEN","championInSingleMale":"PER","championInDoubleMale":"PER","championInSingleFemale":"PER","champion":"PER",\
    }
def transform(js):
    x = js
    new = {}
    ner = []
    relations = []
    for sentence_idx in range(len(x["sentences"])):
        ner.append([])
        relations.append([])
        j = -1
        for event in x["events"][sentence_idx]:
            for i, event_part in enumerate(event):
                j +=1
                if i == 0:
                    ev_head = j
                    tmp = event_part
                    ner[-1].append([event_part[0], event_part[1]+1, "EVT"])
                else:
                    ner[-1].append([event_part[0], event_part[1]+1, NER_per_property[event_part[2]]])
                    #relations[-1].append([tmp[0], tmp[1], event_part[0], event_part[1], event_part[2]])
                    relations[-1].append([ev_head, j, event_part[2]])

    entities = [entity for sentence_entities in ner for entity in sentence_entities]
    entities = [{"type":entity[-1], "start":entity[0],"end":entity[1]} for entity in entities]
    relations = [relation for sentence_relations in relations for relation in sentence_relations]
    new_relations = []
    for relation in relations:
        # Head is the index of the event entity in the entities list, tail is the index of the argument entity in the entities list
        new_relations.append({"type":relation[-1], "head":relation[0],"tail":relation[1]})
    relations = new_relations
    new = {"tokens":[token for sentence in x["sentences"] for token in sentence],"entities":entities,"relations":relations}
    return new


with open("../../re/dbpe_eq_train.json", 'r') as json_file1, open("../../re/dbpe_eq_test.json", 'r') as json_file2, open("../../re/dbpe_eq_dev.json", 'r') as json_file3:
    json_list1 = list(json_file1)
    json_list2 = list(json_file2)
    json_list3 = list(json_file3)

    for json_str1 in json_list1:
        json_str1 = transform(json.loads(json_str1))
        train.append(json_str1)
    for json_str2 in json_list2:
        json_str2 = transform(json.loads(json_str2))
        test.append(json_str2)
    for json_str3 in json_list3:
        json_str3 = transform(json.loads(json_str3))
        dev.append(json_str3)

print(len(train))
print(len(test))
print(len(dev))

with open("data_train.json","w") as f:
    json.dump(train, f)
with open("data_test.json","w") as f:
    json.dump(test, f)
with open("data_dev.json","w") as f:
    json.dump(dev, f)



