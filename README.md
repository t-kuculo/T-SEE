# T-SEE

This is the code for T-SEE Transformer-based Semantic Event Extraction, based on the paper: "T-SEE: Transformer-based Semantic Event Extraction through
Multilabel Prediction and Relation Extraction"

![alt text](https://github.com/t-kuculo/T-SEE/blob/main/figs/example.png)


## Steps
To recreate the dataset creation, training and evaluation as described in the paper follow the steps as described below. To use our complete data for experiments with T-SEE start from [Training](#Training).
### Setting up the virtual environment
* Run the following lines of code to create a virtual environment.

```
virtualenv env
source env/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_md
python -m spacy download en
```

### Data creation and preprocessing
* Follow instructions in the [EventTextWikipediaDumper](https://github.com/foranonymoussubmissions2022/EventTextWikipediaDumper) to run [MWDumper](https://www.mediawiki.org/wiki/Manual:MWDumper) and get Wikipedia articles of events in .ndjson file format. Place the resulting files into data\raw in the project folder.
* To prepare Wikidata and Dbpedia dumps go to the ```data``` folder and run  ```./data_download.sh```.
* Run ```create_shelves.py``` to assure quick access to the dumps.
* The next step will create training data for the baselines and the multilabel classification step of our approach. Go to the ```processing``` folder and run ```create_training_data.py```.
* To train and evaluate specific baselines follow instructions in their respective subfolders in the baselines folder. 
### Training
* First reposition to the ```T-SEE```. folder. To train the multilabel classification model first format the multilabel training, validation and testing data by running  ```format_mlc_data.py```. Then follow up by running ```mlc.py```. This will train the model and generate the output of the model inside the ```/evaluation/output/minority_classes/mlc_output/```.
* To train the Relation Extraction Model first generate the appropriate format for the data by running ``` python convert_data.py```, then run ```train.sh```.
### Evaluation
* Finally to evaluate the performance of **T-SEE** and the baselines, and generate the output of our model go to the ```evaluation``` folder and run ```eval.py```. 
* (Note: To get evaluation scores for the baselines, first follow instructions in their respective subfolders in the [baselines](https://github.com/foranonymoussubmissions2022/T-SEE/blob/main/baselines) folder.)


#### Event Classification


| **Approach** | **DBpedia P** | **DBpedia R** | **DBpedia F1** | **Wikidata P** | **Wikidata R** | **Wikidata F1** |
|--------------|---------------|---------------|----------------|----------------|----------------|-----------------|
| Text2Event   | **0.94**      | **0.94**      | **0.94**       | 0.84           | 0.84           | 0.84            |
| EventGraph   | 0.75          | 0.69          | 0.72           | 0.77           | 0.52           | 0.62            |
| Approach     | 0.92          | 0.92          | 0.92           | **0.85**       | **0.85**       | **0.85**        |
| LLM*         | 0.88          | 0.89          | 0.89           | 0.53           | 0.58           | 0.55            |



#### Relation Extraction


| **Approach** | **DBpedia P** | **DBpedia R** | **DBpedia F1** | **Wikidata P** | **Wikidata R** | **Wikidata F1** |
|--------------|---------------|---------------|----------------|----------------|----------------|-----------------|
| Text2Event   | 0.74          | 0.75          | 0.74           | **0.75**       | **0.77**       | **0.76**        |
| EventGraph   | 0.72          | 0.57          | 0.64           | 0.85           | 0.16           | 0.27            |
| Approach     | **0.75**      | **0.76**      | **0.75**       | **0.75**       | **0.77**       | **0.76**        |
| LLM*         | 0.28          | 0.52          | 0.37           | 0.37           | 0.37           | 0.37            |



## Working with a new dataset
To test our approach on other events ontologies or datasets follow the steps as described below.

![alt text](https://github.com/foranonymoussubmissions2022/T-SEE/blob/main/figs/pipeline.png)

* Format the dataset as follows. Each line represents a JSON file containing text, its sentences, and events in their respective sentences:


```
{"doc_key": 36206, 
"dataset": "event extraction", 
"sentences": [["The", "2020", "United", "States", "presidential", "election", "in", "Missouri", "was", "held", "on", "Tuesday", ",", "November", "3", ",", "2020", ",", "as", "part", "of", "the", "2020", "United", "States", "presidential", "election", "in", "which", "all", "50", "states", "plus", "the", "District", "of", "Columbia", "participated", "."]],
"events": [[[[1, 5, "Election"], [1, 1, "startDate"]]]]}
```

The "events" field should contain a list containing a sublist for each sentence in the "sentences" field. Each of these sublists contains another sublist per event.
An event with N arguments will be written as a list of the form:

```
[
  [trigger_token_start_index, trigger_token_end_index, event_type], 
    [argument_token_start1_index, argument_token_end_index1, arg1_type], 
    [argument_token_start2_index, argument_token_end_index2, arg2_type], 
    ...,  
    [argument_token_startN_index, argument_token_end_indexN, argN_type]
]
```


* Then follow the instructions as described above to train the models, generate the output and evaluate the results.

### Datasets
  * For more details on the dataset formatting, creation, and statistics see the [datasets](https://github.com/t-kuculo/T-SEE/tree/main/data/datasets) folder.








