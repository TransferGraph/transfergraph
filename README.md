# Model Selection with Model Zoo via Graph Learning
Under review in ICDE 2024

In this study, we introduce **TransferGraph**, a novel framework that reformulates model selection as a graph learning problem. TransferGraph constructs a graph using extensive metadata extracted from models and datasets, while capturing their intrinsic relationships. Through comprehensive experiments across 12 real datasets, we demonstrate TransferGraph’s effectiveness in capturing essential model-dataset relationships, yielding up to a 21.8% improvement in correlatio between predicted performance and the actual fine-tuning results compared to the state-of-the-art methods.

![image](https://github.com/zLizy/transferability_graph/blob/main/img/overview.jpg)

## Model zoo and data collection
**Datasets**: We use 11 vision datasets, including 10 datasets from the public transfer learning benchmark [VTAB](https://github.com/google-research/task_adaptation), and [StanfordCars](https://pytorch.org/vision/stable/generated/torchvision.datasets.StanfordCars.html). All the datasets are publicly available online.

**Models**: We construct a model zoo with 185 heterogeneous pre-trained models. These models vary in terms of various aspects, e.g., pre-trained dataset, architectures, and various other metadata features. All the models are available from [HuggingFace](https://huggingface.co/models).

## Instructions
### Data preparation
* Collect metadata (dataset, model), e.g., attributes, performance. The files are under `doc/`
* Obtain **Transferability score** - **LogME**.
```console
cd LogME
DATASET_NAME=pets
METHOD=LogME
python3 compute.py --dataset_name ${DATASET_NAME} --method=${METHOD}
```
### Obtain model and dataset features by graph learning   
*  Run **TransferGraph** to map model-dataset relationships in a graph and use GNN to train node representations.
```console
cd ..
cd graph
python3 run.py                                                                
```
### Predict the model performance 
Learn a simple regression model, e.g., XGBoost, to predict model performance using the features along with other metadata.
```console
cd methods
python3 train_with_linear_regression.py
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



