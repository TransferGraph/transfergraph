from torch.utils.data import DataLoader

ALL_DATASET_CONFIG = {
    "glue": {
        "source": "huggingface",
        "tasks": {
            "cola": {
                "all_feature_key": ["sentence"]
            },
            "mnli": {
                "all_feature_key": ["premise", "hypothesis"]
            },
            "mrpc": {
                "all_feature_key": ["sentence1", "sentence2"]
            },
            "qnli": {
                "all_feature_key": ["question", "sentence"]
            },
            "qqp": {
                "all_feature_key": ["question1", "question2"]
            },
            "rte": {
                "all_feature_key": ["sentence1", "sentence2"]
            },
            "sst2": {
                "all_feature_key": ["sentence"]
            },
            "stsb": {
                "all_feature_key": ["sentence1", "sentence2"]
            },
            "wnli": {
                "all_feature_key": ["sentence1", "sentence2"]
            }
        }
    },
    "tweet_eval": {
        "source": "huggingface",
        "tasks": {
            "sentiment": {
                "all_feature_key": ["text"]
            },
            "emotion": {
                "all_feature_key": ["text"]
            },
            "irony": {
                "all_feature_key": ["text"]
            },
            "hate": {
                "all_feature_key": ["text"]
            },
            "offensive": {
                "all_feature_key": ["text"]
            },
            "emoji": {
                "all_feature_key": ["text"]
            },
        }
    },
    "dbpedia_14": {
        "source": "huggingface",
        "all_feature_key": ["title", "content"]
    },
    "amazon_polarity": {
        "source": "huggingface",
        "all_feature_key": ["title", "content"]
    },
    "ag_news": {
        "source": "huggingface",
        "all_feature_key": ["text"]
    },
    "rotten_tomatoes": {
        "source": "huggingface",
        "all_feature_key": ["text"]
    },
    "dair-ai/emotion": {
        "source": "huggingface",
        "all_feature_key": ["text"]
    },
    "imdb": {
        "source": "huggingface",
        "all_feature_key": ["text"]
    },
    "badmatr11x/offensive": {
        "source": "huggingface",
        "all_feature_key": ["tweet"]
    },
    "financial_phrasebank": {
        "source": "huggingface",
        "tasks": {
            "sentences_50agree": {
                "all_feature_key": ["sentence"]
            },
            "sentences_66agree": {
                "all_feature_key": ["sentence"]
            },
            "sentences_75agree": {
                "all_feature_key": ["sentence"]
            },
            "sentences_allagree": {
                "all_feature_key": ["sentence"]
            }
        }
    },
    "chiapudding/kaggle-financial-sentiment": {
        "source": "huggingface",
        "all_feature_key": ["Sentence"],
        "label_key": "Sentiment"
    },
    "zeroshot/twitter-financial-news-sentiment": {
        "source": "huggingface",
        "all_feature_key": ["text"]
    },
    "strombergnlp/offenseval_2020": {
        "source": "huggingface",
        "tasks": {
            "da": {
                "all_feature_key": ["text"]
            },
            "en": {
                "all_feature_key": ["text"]
            },
            "tr": {
                "all_feature_key": ["text"]
            },
            "gr": {
                "all_feature_key": ["text"]
            },
            "ar": {
                "all_feature_key": ["text"]
            }
        },
        "label_key": "subtask_a"
    },
    "hate_speech18": {
        "source": "huggingface",
        "all_feature_key": ["text"]
    },
    "HHousen/quora": {
        "source": "huggingface",
        "all_feature_key": ["sentence1", "sentence2"]
    },
    "trec": {
        "source": "huggingface",
        "all_feature_key": ["text"],
        "label_key": "fine_label"
    },
    "nli_tr": {
        "source": "huggingface",
        "tasks": {
            "snli_tr": {
                "all_feature_key": ["premise", "hypothesis"]
            },
            "multinli_tr": {
                "all_feature_key": ["premise", "hypothesis"]
            },
        }
    },
    "xnli": {
        "source": "huggingface",
        "tasks": {
            "en": {
                "all_feature_key": ["premise", "hypothesis"]
            }
        }
    },
    "wikimedia/wikipedia": {
        "source": "huggingface",
        "tasks": {
            "20231101.en": {
                "all_feature_key": ["title", "text"]
            }
        }
    },
    "common_language": {
        "source": "huggingface",
        "all_feature_key": ["sentence"],
        "label_key": "language"
    },
    "bertweet-covid19-cased-tweet-data": {
        "source": "local",
        "type": "text",
        "train_path": "data/bertweet-covid19-cased-tweet-data/COVIDTweets_cased_train.txt",
        "validation_path": "data/bertweet-covid19-cased-tweet-data/COVIDTweets_cased_valid.txt",
        "all_feature_key": ["text"],
    },
    "mteb/amazon_reviews_multi": {
        "source": "huggingface",
        "tasks": {
            "en": {
                "all_feature_key": ["text"],
            }
        }
    },
    "cifar100": {
        "source": "huggingface",
        "label_key": "fine_label"
    },
    "stanfordcars": {
        "source": "huggingface",
        "alias": "Multimodal-Fatima/StanfordCars_train",
        "feature_key": "image",
        "label_key": "label"
    },
    "caltech101": {
        "source": "huggingface",
        "alias": "clip-benchmark/wds_vtab-caltech101",
        "feature_key": "webp",
        "label_key": "cls"
    },
    "svhn": {
        "source": "huggingface",
        "alias": "clip-benchmark/wds_vtab-svhn",
        "feature_key": "webp",
        "label_key": "cls"
    },
    "flowers": {
        "source": "huggingface",
        "alias": "nelorth/oxford-flowers",
        "feature_key": "image",
        "label_key": "label"
    },
    "smallnorb_label_elevation": {
        "source": "huggingface",
        "alias": "clip-benchmark/wds_vtab-smallnorb_label_elevation",
        "feature_key": "webp",
        "label_key": "cls"
    },
    "dtd": {
        "source": "huggingface",
        "alias": "cansa/Describable-Textures-Dataset-DTD",
        "feature_key": "image",
        "label_key": "label"
    },
    "pets": {
        "source": "huggingface",
        "alias": "clip-benchmark/wds_vtab-pets",
        "feature_key": "webp",
        "label_key": "cls"
    },
}


class BaseDataset:
    def __init__(
            self,
            name: str,
            train_loader: DataLoader,
            eval_loader: DataLoader,
            all_class: list,
    ):
        self.name = name
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.all_class = all_class
