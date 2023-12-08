import json
from datasets import load_dataset

input_filename1 = "../data/training/re/wde_sparse_re_train.json"
input_filename2  ="../data/training/re/wde_sparse_re_test.json"
input_filename3 ="../data/training/re/wde_sparse_re_dev.json"

output_filename1 = "training_data/wde_sparse_re_train2.json"
output_filename2 = "training_data/wde_sparse_re_test2.json"
output_filename3 = "training_data/wde_sparse_re_dev2.json"

def convert(input_filename, output_filename):
    title = 0
    with open(input_filename) as f:
        dataset = json.load(f)

    with open(output_filename, "w") as f:
        for article in dataset["data"]:
            title = article["title"]
            for paragraph in article["paragraphs"]:
                context = paragraph["context"]
                answers = {}
                for qa in paragraph["qas"]:
                    question = qa["question"]
                    idx = qa["id"]
                    answers["text"] = [a["text"] for a in qa["answers"]]
                    answers["answer_start"] = [a["answer_start"] for a in qa["answers"]]
                    f.write(
                        json.dumps(
                            {
                                "id": idx,
                                "title": title,
                                "context": context,
                                "question": question,
                                "answers": answers,
                            }
                        )
                    )
                    f.write("\n")

convert(input_filename1, output_filename1)
convert(input_filename2, output_filename2)
convert(input_filename3, output_filename3)