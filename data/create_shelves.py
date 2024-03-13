import shelve
import pandas as pd
import csv
import sys
import json
import os
tmp = {}

if not os.path.isdir("shelves"):
        os.mkdir("shelves")
 
print("Read DBpedia -> Wikidata index.")

with open('data/wikidata_to_dbpedia_en.csv') as file:
    dbpedia_ids_to_wikidata_ids = pd.read_csv(file, header=None, index_col=1,

                                              sep=" ", squeeze=True).to_dict()
print("Read types (DBpedia).")
# Dealing with nan
tmp.update(dbpedia_ids_to_wikidata_ids)
for key in tmp:
    if isinstance(key, float):
        dbpedia_ids_to_wikidata_ids["nan"] = dbpedia_ids_to_wikidata_ids[key]
        del dbpedia_ids_to_wikidata_ids[key]


types_dbo = dict()
print("Read DBpedia redirects.")
with open('data/types_dbo.csv') as file_types_dbo:
    for line in csv.reader(file_types_dbo, delimiter=" "):
        entity = line[0]
        type = line[2]
        if entity in types_dbo:
                types_dbo[entity].add(type)
        else:
                types_dbo[entity] = {type}

print("Read types (Wikidata).")
types_wd = dict()
with open('data/types_wd.csv') as file_types_wd:
    for line in csv.reader(file_types_wd, delimiter=" "):
        entity = line[0]
        type = line[1]
        if entity in types_wd:
                types_wd[entity].add(type)
        else:
                types_wd[entity] = {type}



# Read file redirects.csv into index
print("Read redirects.")
with open('data/redirects_en.csv') as file_redirects:
    for line in csv.reader(file_redirects, delimiter=" "):
        source = line[0]
        target = line[1]
        if target in types_dbo and source not in types_dbo:
            types_dbo[source] = types_dbo[target]

        if target in dbpedia_ids_to_wikidata_ids and source not in dbpedia_ids_to_wikidata_ids:
            dbpedia_ids_to_wikidata_ids[source] = dbpedia_ids_to_wikidata_ids[target]

        if target in types_wd and source not in types_wd:
            types_wd[source] = types_wd[target]


all = []


dbpedia2wd_shelve = shelve.open('shelves/dbpedia_to_wikidata_en')
dbpedia2wd_shelve.update(dbpedia_ids_to_wikidata_ids)
dbpedia2wd_shelve.close()





types_dbo_shelve = shelve.open('shelves/types_dbo')
types_dbo_shelve.update(types_dbo)
types_dbo_shelve.close()

types_wd_shelve = shelve.open('shelves/types_wd')
types_wd_shelve.update(types_wd)
types_wd_shelve.close()


types = {}
errs = 0
print("starting")
csv.field_size_limit(sys.maxsize)
with open('instanceof-data.tsv') as instance_properties:
    for i, line in enumerate(csv.reader(instance_properties, delimiter="\t")):
        if i%1000000==0:
            print(i)
            print(errs)
        try:
            entity = line[0]
            label = line[1]
            wd_class = line[2]
            if entity not in types:
                types[entity] = []
            types[entity].append(wd_class)
        except(IndexError):
            errs+=1



types_shelve = shelve.open('shelves/instance_types')
types_shelve.update(types)
types_shelve.close()



