# Datasets for extraction of semantic event representation

We introduce two new large-scale datasets for the extraction of semantic event representations based on [DBpedia](https://github.com/foranonymoussubmissions2022/O-GEE/blob/main/test/datasets/full_dbpedia.jsonl) and [Wikidata](https://github.com/foranonymoussubmissions2022/O-GEE/blob/main/test/datasets/full_wikidata.jsonl). The two datasets contain consist of $43,291$ and $72,649$ samples, respectively.

|         |  DBpedia | Wikidata |
|---------|:--------:| --------:|
|Texts    | $43,291$ | $72,649$ |
|Events   | $43,377$ | $79,335$ |
|Relations| $48,280$ | $139,559$|


We derive the datasets from all Wikipedia articles of events. Event classes and relations are extracted by exploiting existing links to events and their Wikidata representations. 

![alt text](https://github.com/foranonymoussubmissions2022/O-GEE/blob/main/data/datasets/ground_truth.png)
Example illustrating how we label texts with events and relations. The Wikipedia text on the left links to the Wikidata event on the right side, which also has a
relation to an entity mentioned in the text (country: Syria)

Both datasets follow the same formatting, similar to that of DyGie++. They are .jsonl files where each line contains a json like the one below:
```
{"doc_key": 36206, 
"dataset": "event extraction", 
"sentences": [["The", "2020", "United", "States", "presidential", "election", "in", "Missouri", "was", "held", "on", "Tuesday", ",", "November", "3", ",", "2020", ",", "as", "part", "of", "the", "2020", "United", "States", "presidential", "election", "in", "which", "all", "50", "states", "plus", "the", "District", "of", "Columbia", "participated", "."]],
"events": [[[[1, 5, "Election"], [1, 1, "startDate"]]]]}
```
The "events" field is a list containing a sublist for each sentence in the "sentences" field. Each of these sublists contains another sublist per event.
An event with N arguments will be written as a list of the form:
  ```
[[trigger_token_start_index, trigger_token_end_index, event_type], 
[argument_token_start1_index, argument_token_end_index1, arg1_type], 
[argument_token_start2_index, argument_token_end_index2, arg2_type], 
..., [argument_token_startN_index, argument_token_end_indexN, argN_type]]
```


The   [event ontology distribution](https://github.com/foranonymoussubmissions2022/O-GEE/blob/main/test/datasets/event_ontology_distribution.json) file contains a dictionary describing the distribution of  event and property labels across the two datasets.

