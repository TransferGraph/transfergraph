from typing import Tuple

import numpy
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedModel


def extract_features(dataloader: DataLoader, model: PreTrainedModel) -> Tuple[numpy.ndarray, numpy.ndarray, int]:
    labels_tensor = torch.zeros(1, ).to(model.device)
    features_tensor = None

    for batch in tqdm(dataloader):
        batch = {k: v.to(model.device) for k, v in batch.items()}

        with torch.no_grad():
            outputs = model(**batch, output_hidden_states=True)
            output = outputs.hidden_states[-1].mean(dim=1)

            if features_tensor is None:
                feature_dimension = output.shape[1]
                features_tensor = torch.zeros(1, feature_dimension).to(model.device)

            feature = torch.flatten(output, start_dim=1)
            features_tensor = torch.cat((features_tensor, feature), 0)
            labels_tensor = torch.cat((labels_tensor, batch['labels'].to(model.device)), 0)

    features_tensor = features_tensor.cpu().detach().numpy()[1:]
    labels_tensor = labels_tensor.cpu().detach().numpy()[1:]

    return features_tensor, labels_tensor, feature_dimension


def extract_features_without_labels(dataloader: DataLoader, model: PreTrainedModel) -> Tuple[numpy.ndarray, numpy.ndarray, int]:
    features_tensor = None

    for batch in tqdm(dataloader):
        batch = {k: v.to(model.device) for k, v in batch.items() if k != 'labels'}

        with torch.no_grad():
            outputs = model(**batch, output_hidden_states=True)
            output = outputs.hidden_states[-1].mean(dim=1)

            if features_tensor is None:
                feature_dimension = output.shape[1]
                features_tensor = torch.zeros(1, feature_dimension).to(model.device)

            feature = torch.flatten(output, start_dim=1)
            features_tensor = torch.cat((features_tensor, feature), 0)

    features_tensor = features_tensor.cpu().detach().numpy()[1:]

    return features_tensor, feature_dimension
