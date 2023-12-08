   
from logging import NullHandler
from re import I
#import utils
import datetime
from dateparser import parse
import json

#with open("subevents/data/transitive_class_hierarchy.json","r") as f:
    #class_hierarchy = json.load(f)


class Corpus:
    def __init__(self, Articles):
        self.Articles = dict()
        if Articles:
            list_of_articles = [article.empty_text() for article in Articles]  
            for article in list_of_articles:
                self.Articles[article.article_name] = article

    def addToCorpus(self, Articles):
        list_of_articles = [article.empty_text() for article in Articles]
        for article in list_of_articles:
            self.Articles[article.article_name] = article

class Article:
    def __init__(self, article_json):
        self.types = article_json["types"]
        self.article_name = article_json["name"]
        self.language = article_json["language"]
        self.id = article_json["id"]
        self.old = article_json["paragraphs"]
        self.raw_paragraphs = self.old["paragraphs"]#article_json["paragraphs"]
        self.completeTitle = self.old["completeTitle"]
        self.title = self.old["title"]
        self.input_paragraphs = self.get_input_paragraphs(self.raw_paragraphs, [])
        self.large_input_paragraphs = self.get_large_paragraphs(self.input_paragraphs)
        self.events = []

    def __str__(self):
        return self.article_name + "\n" +  "\n\n".join(str(event) for paragraph_events in self.events for event in paragraph_events)

    def get_input_paragraphs(self, paragraphs, input_paragraphs):
        for paragraph in paragraphs:# self.raw_paragraphs:
            subparagraphs = paragraph
            if "paragraphs" in subparagraphs:
                self.get_input_paragraphs(subparagraphs["paragraphs"], input_paragraphs)
            if "sentences" not in subparagraphs:
                continue
            input_paragraphs.append(InputParagraph(subparagraphs))
        return input_paragraphs

    def get_large_paragraphs(self, paragraphs):
        large_paragraphs = []
        temp = []
        seenTitles = []
        seenTitles = ''
        # should maybe make it a string variable instead in case of repeating section titles
        for paragraph in paragraphs:
            if paragraph.completeTitle not in seenTitles:
                #seenTitles.append(paragraph.completeTitle)
                seenTitles=paragraph.completeTitle
                if len(temp) != 0:
                    large_paragraphs.append(LargeInputParagraph(temp))
                    temp = []
            temp.append(paragraph)
        return large_paragraphs

    def empty_text(self):
        self.raw_paragraphs = None
        self.input_paragraphs = None
        self.large_input_paragraphs = None
        self.old = None
        return self

class LargeInputParagraph:
    def __init__(self, input_paragraphs):
        self.completeTitle =  input_paragraphs[0].completeTitle
        self.sentences = []
        for p in input_paragraphs:
            self.sentences += p.sentences
        self.text = [sentence.text for sentence in self.sentences]
       

class InputParagraph:   
    def __init__(self, subparagraph):
        self.completeTitle =  subparagraph["completeTitle"]
        self.title = subparagraph["title"]
        self.sentences = self.get_sentences(subparagraph["sentences"])
        self.text = [sentence.text for sentence in self.sentences]
        
    def get_sentences(self, subparagraph):
        sentences = []
        for sentence in subparagraph:
            sentences.append(Sentence(sentence))
        return sentences


class Sentence:
    def __init__(self, sentence):
        self.text = sentence["text"].replace("  ", " ") # testing a bugfix for double spacing that QA, and ED models fix resulting in lack of indexing overlap
        self.links = sentence["links"]


class Event:
    def __init__(self, event_type, event_trigger, arguments):
        #self.type = event_type
        self.trigger = event_trigger
        self.arguments = arguments
        self.wikidata_type = event_type


    def __str__(self):
        return "Event Type: " + self.type + "\nEvent Trigger: " + self.trigger +"\nArguments:\n" + "\n".join([str(arg) for arg in self.arguments])


class Argument:
    def __init__(self, arg_type, arg_text, constraints, last_time):
        self.type = []
        # if multiple answers have same probability, take all of them as the text of the argument
        self.texts = arg_text
        self.wikidata_types = []
        self.dbpedia_types = []
        self.links = []
        self.wikidata_id = []
        self.property = arg_type
        self.property_label = None
        self.constraints = constraints
        self.last_time = last_time

    def __str__(self):
        if self.links != []:
            #return "type : "+ repr(self.type) +"|"+ self.text + " | wikidataID: " +self.wikidata_id + " wikidata_types: " + repr(self.wikidata_types)
            return "property : "+ repr(self.property) +"type : "+ repr(self.type) +"|"+ self.text + " | wikidataID: " + "["+", ".join(self.wikidata_id) + "] wikidata_types: " +\
                "[" + str(self.wikidata_types)
        else:
            return " : ".join([str(self.property), str(self.text)])
    
    def __bool__(self):
        # Returns True if object has a wikidata_type and fits constraints, otherwise returns False
        if isinstance(self.wikidata_types, set):
            self.wikidata_types = list(self.wikidata_types)
        tmp = self.wikidata_types[:]
        already_popped = 0
        # create condition for counting links in the case of "number of participants" properties
        if self.wikidata_id:
            for i, link in enumerate(tmp):
                #if set(link).update(set(getAncestors(link))) & set(self.constraints):
                if link in class_hierarchy:
                    parents = class_hierarchy[link]
                else:
                    parents = [link]
                    with open("instance_that_should_have_been_type.txt","a") as dd:
                        dd.write(link)
                if set(parents) & set(self.constraints):
                    self.type = self.dbpedia_types 
                    return True    
                elif link!="None":
                    self.wikidata_types.pop(i-already_popped)
                    already_popped += 1
                    if self.type:
                        self.type.pop(i-already_popped)
                    if self.dbpedia_types:
                        self.dbpedia_types.pop(i-already_popped)
                    if self.links:
                        self.links.pop(i-already_popped)
        else:
            # sort argument texts from largest to smallest, if find a date or quantity stop
            self.texts = sorted(self.texts, key = len, reverse = True)
            for text in self.texts:
                #date = utils.isDate(text)
                date = parse(text, settings={'RELATIVE_BASE': self.last_time})
                if date and "TIMEX" in self.constraints:
                    self.texts = date
                    self.last_time = date
                    self.type = "TIMEX"
                    
                    return True
                elif any(i.isdigit() for i in text.split()) and "QUANTITY" in self.constraints:
                    self.texts = [i for i in text.split() if i.isdigit()][0]
                    self.type = "QUANTITY"
                    return True
        return False