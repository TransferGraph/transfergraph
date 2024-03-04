# Model Selection with Model Zoo via Graph Learning
Under review in ICDE 2024

In this study, we introduce **TransferGraph**, a novel framework that reformulates model selection as a graph learning
problem. TransferGraph constructs a graph using extensive metadata extracted from models and datasets, while capturing
their intrinsic relationships. Through comprehensive experiments across 12 real datasets, we demonstrate TransferGraph’s
effectiveness in capturing essential model-dataset relationships, yielding up to a 21.8% improvement in correlation
between predicted performance and the actual fine-tuning results compared to the state-of-the-art methods.

![image](https://github.com/zLizy/transferability_graph/blob/main/img/overview.jpg)

## Setup used in ICDE 2024 submission

We include most of the artifacts for reproducing our results. The setup is documented through files in the `resources/`
folder. This includes metadata of datasets and models used, as well as the fine-tuning performances and LogME scores.

To reproduce our setup, or create a new one we advice to follow the instructions below.

## Instructions

### Dataset embeddings

The only types of artifacts not included in this repository are the models, datasets and dataset embeddings. Datasets
and models are downloaded from HuggingFace by default when using any of the scripts in `tools/`. Dataset embeddings can
be computed using `tools/embed_dataset.py`. Example usage:

```bash
python tools/embed_dataset.py --dataset_path=rotten_tomatoes --model_name=EleutherAI/gpt-neo-125m --task_type=sequence_classification --embedding_method="domain_similarity"
```

However, we've also uploaded the embeddings we used
in [our HuggingFace organization](https://huggingface.co/TransferGraph) as datasets.

### Baseline scores (LogME)

Other baselines can be computed using `tools/transferability_score.py`, or found under `resources/`. We've included all
metrics from https://github.com/Ba1Jun/model-selection-nlp, however did not implement everything for direct usage.
Example usage of the script:

```bash
python tools/transferability_score.py --dataset_path=rotten_tomatoes --model_name=roberta-large --task_type=sequence_classification --metric="LogME"
```

### Fine-tuning scores

Fine-tuning can be done using `tools/finetune.py`. Scores used in our experiments can be found under `resources/` under
the respective `records.csv` file. This also includes all the hyper parameters used. Example usage of the script:

```bash
python tools/finetune.py --dataset_path=tweet_eval --dataset_name=irony --model_name=distilbert-base-uncased --task_type=sequence_classification --batch_size=64 --num_train_epochs=1 --learning_rate=2e-4 --peft_method=lora
```

### Obtain model and dataset features by graph learning   
*  Run **TransferGraph** to map model-dataset relationships in a graph and use GNN to train node representations.
```console
cd ..
cd tools
python3 run.py                                                                
```
### Predict the model performance 
Learn a simple regression model, e.g., XGBoost, to predict model performance using the features along with other metadata.
```console
cd tools
python3 train_prediction_model.py
```
### Evaluation
We use **Pearson correlation** as evaluation metric. We compare the predicted model performance with the actual fine-tuning results.

## Batch experiments
We vary the configurations for experiments. To run experiments, use `run_graph.sh`.
```python
./run_graph.sh
```
### Confugurations
* contain_dataset_feature - whether include dataset features as node features
* gnn_method - GNN algorithms to learn from a graph
* test_dataset - the dataset that models fine-tuned on
* top_neg_K - the percentage of the models with the lowest transerability score
* top_pos_K - the percentage of the models with the highest transerability score
* accu_neg_thres - the percentage of the least performing models regarded as negative edges in a graph
* accu_pos_thres - the percentage of the highest performing models regarded as negative edges in a graph
* hidden_channels - dimension of the latent representations (e.g., 128)
* finetune_ratio - the amount of fine-tuning history used to train the GNN
* dataset_embed_method - method to extract latent representation of datasets



