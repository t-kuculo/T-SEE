#!/usr/bin/env python
# -*- coding:utf-8 -*-
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from torch import nn
import configparser
import json
import copy
import jsonlines
from seq2seq.constrained_seq2seq import decode_tree_str
from extraction.predict_parser.tree_predict_parser import TreePredictParser
from extraction.event_schema import EventSchema
from extraction.extract_constraint import get_constraint_decoder

device = torch.device("cuda")
with torch.cuda.device('cuda'):
        torch.cuda.empty_cache()


class EventExtractor:
    def __init__(self, tokenizer, model, tree_parser, constraint_decoder):
        self.tokenizer = tokenizer
        self.model= nn.DataParallel(model)
        self.model = model.to(device)
        self.tree_parser = tree_parser
        self.constraint_decoder = constraint_decoder
    def from_pretrained(model_path):
        event_schema = EventSchema.read_from_file(f"{model_path}/event.schema")
        tokenizer = T5Tokenizer.from_pretrained(model_path)
        model = T5ForConditionalGeneration.from_pretrained(model_path)
        tree_parser = TreePredictParser(event_schema)
        constraint_decoder = get_constraint_decoder(tokenizer=tokenizer,
                                                    type_schema=event_schema,
                                                    decoding_schema="tree",
                                                    source_prefix='event: ')
        return EventExtractor(tokenizer=tokenizer,
                              model=model,
                              tree_parser=tree_parser,
                              constraint_decoder=constraint_decoder)
    def extract_event(self, text_list, constrained_decoding=False):
        text_list = ['event: ' + text for text in text_list]
        input_ids = self.tokenizer(text_list, return_tensors='pt',
                                   padding=True).input_ids
        input_ids = input_ids.to(device)
        def prefix_allowed_tokens_fn(batch_id, sent):
            # print(self.tokenizer.convert_ids_to_tokens(inputs['labels'][batch_id]))
            src_sentence = input_ids[batch_id]
            return self.constraint_decoder.constraint_decoding(src_sentence=src_sentence, tgt_generated=sent)
        outputs = self.model.generate(
            input_ids,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn if constrained_decoding else None
        )
        event = decode_tree_str(outputs.cpu(), self.tokenizer)
        event_list, _ = self.tree_parser.decode(pred_list=event, gold_list=[])
        event_list = [event['pred_record'] for event in event_list]
        return event_list

model_path ="models/dbp_enriched_30_epochs" #"models/dbp_6_epochs"#"models/20epochs"#"models/minority_classes_unlabelled2"
event_extractor = EventExtractor.from_pretrained(model_path=model_path)

def predict_events(texts):
    #with torch.cuda.device('cuda:1'):
        #torch.cuda.empty_cache()
    results = []
    events = event_extractor.extract_event(texts)
    for text, event in zip(texts, events):
        results.append((text, event))
    return results

def get_test_data_output(path_to_data = "../../data/training/t2e/wde_eq_test.json"):
    new_samples = []
    all_samples = []
    path_to_data = "../../data/training/t2e/wde_eq_test.json"
    with open(path_to_data, 'r') as json_file:
        json_list = list(json_file)
    
    for json_str in json_list:
        all_samples.append(json.loads(json_str))

    for i, sample in enumerate(all_samples):
        print("%d out of %d"%(i, len(all_samples)))
        t2e_output = predict_sentence_events([sample["text"]])
        #if t2e_output:
            #print(t2e_output)
        new_sample = copy.deepcopy(sample)
        new_sample["t2e"] = t2e_output
        new_samples.append(new_sample)
        #print(new_sample)

    with jsonlines.open("../../evaluation/output/t2e_output/wde_test_output_30.json","w") as f:
        f.write_all(new_samples)


def predict_sentence_events(sentences):
    return event_extractor.extract_event(sentences)
"""
if __name__ == "__main__":
    event_extractor = EventExtractor.from_pretrained(model_path=model_path)
    texts = ["An uprising by the citizens of Madrid broke out on 2 May, slew 150 French soldiers, and was violently stamped out by Marshal Joachim Murat's elite Imperial Guards and Mamluk cavalry.The man was captured in Los Angeles on Tuesday by bounty hunters.","The suicide bombing killed dozens in Berlin."]
events = event_extractor.extract_event(texts)
for text, event in zip(texts, events):
    print(text, event)
"""
#model_path = "pretrained_models/dyiepp_ace2005_en_t5_base"
#event_extractor = EventExtractor.from_pretrained(model_path=model_path)
get_test_data_output()