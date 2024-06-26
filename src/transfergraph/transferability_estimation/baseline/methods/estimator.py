# Code for various metrics from https://github.com/Ba1Jun/model-selection-nlp
import argparse
import logging
import os
import time

import pandas as pd
from transformers import PreTrainedModel

from transfergraph.config import get_root_path_string
from transfergraph.dataset.base_dataset import BaseDataset
from transfergraph.transferability_estimation.baseline.methods.utils import TransferabilityMethod, TransferabilityDistanceFunction
from transfergraph.transferability_estimation.feature_utils import extract_features

logger = logging.getLogger(__name__)


class TransferabilityEstimatorFeatureBased:
    def __init__(
            self,
            dataset: BaseDataset,
            model: PreTrainedModel,
            all_baseline: list,
            args: argparse.Namespace
    ):
        self.dataset = dataset
        self.model = model
        self.all_baseline = all_baseline
        self.args = args

    def score(self):
        logger.info("***** Calculating transferability score *****")
        logger.info(f"  Target dataset name = {self.dataset.name}")
        logger.info(f"  Model name = {self.model.name_or_path}")

        time_start = time.time()

        features_tensor, labels_tensor, _ = extract_features(self.dataset.train_loader, self.model)

        time_feature_extract = time.time() - time_start
        result_record_list = list()

        for baseline_method in self.all_baseline:
            logger.info(f"  Metric = {baseline_method.name}")

            if baseline_method == TransferabilityMethod.LOG_ME:
                from transfergraph.transferability_estimation.baseline.methods.logme import LogME
                metric = LogME()
            elif baseline_method == TransferabilityMethod.NLEEP:
                from transfergraph.transferability_estimation.baseline.methods.nleep import NLEEP
                metric = NLEEP()
            elif baseline_method == TransferabilityMethod.PARC:
                from transfergraph.transferability_estimation.baseline.methods.parc import PARC
                metric = PARC(TransferabilityDistanceFunction.CORRELATION)
            elif baseline_method == TransferabilityMethod.H_SCORE:
                from transfergraph.transferability_estimation.baseline.methods.hscore import HScore
                metric = HScore()
            elif baseline_method == TransferabilityMethod.REG_H_SCORE:
                from transfergraph.transferability_estimation.baseline.methods.hscore_reg import HScoreR
                metric = HScoreR()
            else:
                raise Exception(f"Unexpected TransferabilityMetric: {baseline_method}")

            time_method_start = time.time()
            score = metric.score(features_tensor, labels_tensor)
            time_method = time.time() - time_method_start

            logger.info(f" {baseline_method} Score: {score}")

            result_record_list.append(self._save_result_to_csv(score, baseline_method, time_feature_extract, time_method))

        result_record = pd.concat(result_record_list, ignore_index=True)
        file_name = os.path.join(
            get_root_path_string(),
            'resources/experiments',
            self.args.task_type.value,
            'transferability_score_records.csv'
            )

        if os.path.isfile(file_name):
            result_record.to_csv(file_name, mode='a', index=True, header=False)
        else:
            result_record.to_csv(file_name, mode='w', index=True)

    def _read_transferability_score_records(self, file_name: str) -> pd.DataFrame:
        file = os.path.join(file_name)

        if not os.path.exists(file):
            all_column = ["model", "target_dataset"] + list(vars(self.args).keys()) + ["runtime", "score"]
            all_column.remove("all_baseline")
            all_column.append("baseline")

            return pd.DataFrame(columns=all_column)
        else:
            return pd.read_csv(file, index_col=0)

    def _save_result_to_csv(self, score, baseline, time_feature_extract, time_method) -> pd.DataFrame:
        training_record = vars(self.args)

        if "all_baseline" in training_record:
            del training_record["all_baseline"]

        training_record['metric'] = baseline
        training_record["model"] = self.model.name_or_path
        training_record['target_dataset'] = self.dataset.name
        training_record["runtime"] = time_feature_extract + time_method
        training_record["score"] = score

        return pd.DataFrame(training_record, index=[0])
