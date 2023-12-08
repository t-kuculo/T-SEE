allennlp predict models/model \
    ../../data/training/minority_classes/dygiepp/wde_eq_test.json \
    --predictor dygie \
    --include-package dygie \
    --use-dataset-reader \
    --output-file ../../evaluation/output/dygiepp_output/test_output.json \
    --silent

#    --cuda-device 0 \