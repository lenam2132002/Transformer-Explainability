## Reproducing results on SecBERT

1. Download the pretrained weights:

- Download `classifier.zip` from https://drive.google.com/file/d/1hsnMRRapoghiKPqlO9I3Xv349jKi5-Kg/view?usp=sharing
- mkdir -p `./bert_models/capec`
- unzip classifier.zip -d ./bert_models/capec/

2. Download the dataset:

- Download `capec.zip` from https://drive.google.com/file/d/1QEMAvtxTKdZgez3RQtIETL6N6vNy8Vel/view?usp=sharing
- unzip capec.zip -d ./data/

3. Now you can run the model.

Example:
```
python BERT_rationale_benchmark/models/pipeline/capec_pipeline.py --data_dir data/capec/ --output_dir bert_models/capec/ --model_params BERT_params/capec_bert.json
```
To control which algorithm to use for explanations change the `method` variable in `BERT_rationale_benchmark/models/pipeline/bert_pipeline.py` (There are 2 methos 'transformer_attribution' which is my method and 'growth_truth' which is growth_truth).
Running this command will create a directory for the method in `bert_models/movies/<method_name>`.

In order to run f1 test with k, run the following command:
```
python BERT_rationale_benchmark/metrics.py --data_dir data/capec/ --split test --results bert_models/capec/ours/identifier_results_k.json
```

Also, in the method directory there will be created `.tex` files containing the explanations extracted for each example. This corresponds to our visualizations in the supplementary.


