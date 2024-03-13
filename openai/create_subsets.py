import json
import random
from collections import Counter
import spacy

nlp = spacy.load("en_core_web_sm")  # Make sure to have the spaCy model installed

def generate_subsets(input_file, output_prefix, subset_size, seed=42):
    random.seed(seed)
    all_entries = []

    # Load the dataset
    with open(input_file, 'r') as f:
        all_entries = json.load(f)

    # Define different subset generation strategies
    strategies = {
        'sentence_length': subset_by_sentence_length,
        'geographical_diversity': subset_by_geographical_diversity,
        'temporal_event_distribution': subset_by_temporal_event_distribution,
        'named_entity_diversity': subset_by_named_entity_diversity,
        'complex_sentence_structures': subset_by_complex_sentence_structures,
        'semantic_diversity': subset_by_semantic_diversity,
    }

    # Generate subsets for each strategy
    for strategy_name, strategy_func in strategies.items():
        subset = strategy_func(all_entries, subset_size)
        output_file = f'{output_prefix}_{strategy_name}.json'
        write_subset(subset, output_file)

def subset_by_sentence_length(entries, size):
    # Select entries based on sentence length
    sorted_entries = sorted(entries, key=lambda e: len(e['sentence']), reverse=True)
    return sorted_entries[:min(size, len(sorted_entries))]

def _subset_by_geographical_diversity(entries, size):
    # Select entries with diverse geographical locations mentioned across all event classes
    def count_geographical_places(entry):
        count = 0
        for event_class, properties in entry['predicted'].items():
            count += len(properties.get('place', []))
        return count

    sorted_entries = sorted(entries, key=count_geographical_places, reverse=True)
    return sorted_entries[:min(size, len(sorted_entries))]

def subset_by_geographical_diversity(entries, size):
    def count_geographical_places(entry):
        doc = nlp(entry['sentence'])  # Assuming 'text' contains the full text of the entry
        count = sum(1 for ent in doc.ents if ent.label_ in ["GPE", "LOC"])  # GPE and LOC are typical labels for geographical places
        return count

    sorted_entries = sorted(entries, key=count_geographical_places, reverse=True)
    return sorted_entries[:min(size, len(sorted_entries))]

def _subset_by_temporal_event_distribution(entries, size):
    # Select entries with explicit date mentions across all event classes
    date_entries = [e for e in entries if any('date' in properties for event_class, properties in e['predicted'].items())]
    return random.sample(date_entries, min(size, len(date_entries)))

def subset_by_temporal_event_distribution(entries, size):
    def has_temporal_expression(entry):
        doc = nlp(entry['sentence'])  # Assuming 'text' contains the full text of the entry
        return any(ent.label_ == "DATE" for ent in doc.ents)  # DATE is a common label for temporal expressions

    date_entries = [e for e in entries if has_temporal_expression(e)]
    return random.sample(date_entries, min(size, len(date_entries)))


def _subset_by_named_entity_diversity(entries, size):
    # Select entries with a high diversity of named entities mentioned across all event classes and their properties
    def count_unique_named_entities(entry):
        counts = Counter()
        for event_class, properties in entry['predicted'].items():
            for key, values in properties.items():
                if isinstance(values, list):  # Assuming all named entities are listed under their properties
                    counts.update(values)
        return len(counts)

    sorted_entries = sorted(entries, key=count_unique_named_entities, reverse=True)
    return sorted_entries[:min(size, len(sorted_entries))]

def subset_by_named_entity_diversity(entries, size):
    def count_unique_named_entities(entry):
        doc = nlp(entry['sentence'])  # Assuming 'text' contains the full text of the entry
        unique_entities = {ent.text for ent in doc.ents}  # Use a set to ensure uniqueness
        return len(unique_entities)

    sorted_entries = sorted(entries, key=count_unique_named_entities, reverse=True)
    return sorted_entries[:min(size, len(sorted_entries))]


def subset_by_complex_sentence_structures(entries, size):
    # Function to calculate the depth of a syntactic parse tree
    def parse_tree_depth(sent):
        doc = nlp(sent)
        depths = [sum(1 for _ in token.ancestors) for token in doc]
        return max(depths) if depths else 0

    # Select entries with the most complex sentence structures based on parse tree depth
    sorted_entries = sorted(entries, key=lambda e: parse_tree_depth(e['sentence']), reverse=True)
    return sorted_entries[:min(size, len(sorted_entries))]

def subset_by_semantic_diversity(entries, size):
    # Function to calculate semantic diversity based on the variety of verb phrases and their arguments
    def semantic_diversity(sent):
        doc = nlp(sent)
        verbs = set()
        for token in doc:
            if token.pos_ == "VERB":
                verbs.add(token.lemma_)
        return len(verbs)

    # Select entries with high semantic diversity
    sorted_entries = sorted(entries, key=lambda e: semantic_diversity(e['sentence']), reverse=True)
    return sorted_entries[:min(size, len(sorted_entries))]

def write_subset(subset, output_file):
    with open(output_file, 'w') as f:
        json.dump(subset, f, indent=4)

mode = "dbpedia"
# Example usage
input_file = f'{mode}_output.json'
output_prefix = 'output_prefix'
subset_size = 100  # Specify the desired subset size

# Generate subsets
generate_subsets(input_file, output_prefix, subset_size)
