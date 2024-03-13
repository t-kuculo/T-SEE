import torch
from torch import nn
from torch.utils.data import DataLoader, SequentialSampler
from transformers import (
    AutoModelForQuestionAnswering, AutoTokenizer, 
    squad_convert_examples_to_features, SquadExample
)
from transformers.data.metrics.squad_metrics import compute_predictions_logits
from transformers.data.processors.squad import SquadResult, SquadExample


# Configuration Constants
MODEL_NAME_OR_PATH = "../T-SEE/models/dbpe_with_spacy_model" 
USE_OWN_MODEL = True
MAX_SEQ_LENGTH = 384
DOC_STRIDE = 128
MAX_QUERY_LENGTH = 64
N_BEST_SIZE = 1
MAX_ANSWER_LENGTH = 30
DO_LOWER_CASE = True
NULL_SCORE_DIFF_THRESHOLD = 0.0
BATCH_SIZE = 10

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def to_list(tensor):
    """Convert a tensor to a list."""
    return tensor.detach().cpu().tolist()

def setup_model(model_name_or_path):
    """Setup and return the model and tokenizer."""
    model = AutoModelForQuestionAnswering.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
    model = nn.DataParallel(model).to(device)
    return model, tokenizer

def create_examples(queries, context_text):
    """Create and return SquadExample objects for each query."""
    examples = []
    for i, query in enumerate(queries):
        example = SquadExample(
            qas_id=str(i),
            question_text=query,
            context_text=context_text,
            answer_text=None,
            start_position_character=None,
            title="Predict",
            is_impossible=False,
            answers=None
        )
        examples.append(example)
    return examples

def run_prediction(queries, context_text, model, tokenizer):
    """Run model prediction on the given queries and context."""
    examples = create_examples(queries, context_text)
    features, dataset = squad_convert_examples_to_features(
        examples=examples,
        tokenizer=tokenizer,
        max_seq_length=MAX_SEQ_LENGTH,
        doc_stride=DOC_STRIDE,
        max_query_length=MAX_QUERY_LENGTH,
        is_training=False,
        return_dataset="pt",
        threads=1
    )

    eval_dataloader = DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=BATCH_SIZE)
    all_results = []

    for batch in eval_dataloader:
        model.eval()
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids": batch[2]}
            example_indices = batch[3]
            outputs = model(**inputs)
            for i, example_index in enumerate(example_indices):
                eval_feature = features[example_index.item()]
                unique_id = int(eval_feature.unique_id)
                output = [to_list(output[i]) for output in outputs.to_tuple()]
                start_logits, end_logits = output
                result = SquadResult(unique_id, start_logits, end_logits)
                all_results.append(result)

    # Define file paths for outputs
    output_prediction_file = "predictions.json"
    output_nbest_file = "nbest_predictions.json"
    output_null_log_odds_file = "null_predictions.json"

    predictions = compute_predictions_logits(
        examples,
        features,
        all_results,
        N_BEST_SIZE,
        MAX_ANSWER_LENGTH,
        DO_LOWER_CASE,
        output_prediction_file,
        output_nbest_file,
        output_null_log_odds_file,
        False,  # verbose_logging
        True,  # version_2_with_negative
        NULL_SCORE_DIFF_THRESHOLD,
        tokenizer
    )
    return predictions

def run_relation_extraction(queries, context):
    """
    Given a list of queries and a context, return a list of relations and their scores.
    """
    predictions = run_prediction(queries, context, model, tokenizer)
    return [predictions[key] for key in predictions.keys()]

# Main execution
model, tokenizer = setup_model(MODEL_NAME_OR_PATH)
# Example usage
# results = run_relation_extraction(queries, context)
