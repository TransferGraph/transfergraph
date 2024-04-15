# Code for various metrics from https://github.com/Ba1Jun/model-selection-nlp
import argparse
import logging
import os
import time

import pandas as pd
from transformers import PreTrainedModel

from transfergraph.config import get_root_path_string
from transfergraph.dataset.base_dataset import BaseDataset
from transfergraph.transferability_estimation.baseline.methods.utils import TransferabilityMetric, TransferabilityDistanceFunction
from transfergraph.transferability_estimation.feature_utils import extract_features

logger = logging.getLogger(__name__)


class TransferabilityEstimatorFeatureBased:
    def __init__(
            self,
            dataset: BaseDataset,
            model: PreTrainedModel,
            transferability_metric: TransferabilityMetric,
            args: argparse.Namespace
    ):
        self.dataset = dataset
        self.model = model
        self.transferability_metric = transferability_metric
        self.args = args

    def score(self):
        logger.info("***** Calculating transferability score *****")
        logger.info(f"  Target dataset name = {self.dataset.name}")
        logger.info(f"  Model name = {self.model.name_or_path}")
        logger.info(f"  Metric = {self.transferability_metric.name}")

        if self.transferability_metric == TransferabilityMetric.LOG_ME:
            from transfergraph.transferability_estimation.baseline.methods.feature_based.logme import LogME
            metric = LogME()
        elif self.transferability_metric == TransferabilityMetric.NLEEP:
            from transfergraph.transferability_estimation.baseline.methods.feature_based.nleep import NLEEP
            metric = NLEEP()
        elif self.transferability_metric == TransferabilityMetric.PARC:
            from transfergraph.transferability_estimation.baseline.methods.feature_based.parc import PARC
            metric = PARC(TransferabilityDistanceFunction.CORRELATION)
        else:
            raise Exception(f"Unexpected TransferabilityMetric: {self.transferability_metric}")

        time_start = time.time()

        features_tensor, labels_tensor, _ = extract_features(self.dataset.train_loader, self.model)
        score = metric.score(features_tensor, labels_tensor)

        logger.info(f"  Score: {score}")

        self._save_result_to_csv(score, time_start)

    def _read_transferability_score_records(self, file_name: str) -> pd.DataFrame:
        file = os.path.join(file_name)

        if not os.path.exists(file):
            all_column = ["model", "target_dataset"] + list(vars(self.args).keys()) + ["runtime", "score"]
            return pd.DataFrame(columns=all_column)
        else:
            return pd.read_csv(file, index_col=0)

    def _save_result_to_csv(self, score, time_start) -> None:
        file_name = os.path.join(get_root_path_string(), 'resources/experiments', self.args.task_type, 'transferability_score_records.csv')
        result_record = self._read_transferability_score_records(file_name)

        training_record = vars(self.args)
        training_record["model"] = self.model.name_or_path
        training_record['target_dataset'] = self.dataset.name
        training_record["runtime"] = time.time() - time_start
        training_record["score"] = score

        result_record = pd.concat([result_record, pd.DataFrame(training_record, index=[0])], ignore_index=True)
        result_record.to_csv(file_name)
