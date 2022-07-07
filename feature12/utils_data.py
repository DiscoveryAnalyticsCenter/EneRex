import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Union

import torch
from filelock import FileLock
from torch.utils.data.dataset import Dataset

from transformers import (
    InputFeatures,
    PreTrainedTokenizer,
)

#new editions
# from utils_dataProcessor import glue_convert_examples_to_features, SciProcessor
from utils_dataProcessor import glue_convert_examples_to_features, SciProcessor


logger = logging.getLogger(__name__)


@dataclass
class SciDataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    listOfStringMode: bool = field(
        default=False, metadata={"help": "Create dataset from given listOfString"}
    )

    


class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"


class SciDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    args: SciDataTrainingArguments
    output_mode: str
    features: List[InputFeatures]

    def __init__(
        self,
        args: SciDataTrainingArguments,
        tokenizer: PreTrainedTokenizer,
        listOfString,
        limit_length: Optional[int] = None,
        mode: Union[str, Split] = Split.train,
        cache_dir: Optional[str] = None,
    ):
        self.args = args
        
        #change
        self.processor = SciProcessor()
        self.output_mode = "classification"
        
        if isinstance(mode, str):
            try:
                mode = Split[mode]
            except KeyError:
                raise KeyError("mode is not a valid split name")

        

        label_list = self.processor.get_labels()
        self.label_list = label_list

        
        logger.info(f"Creating features from List of Strings")

        if mode == Split.dev:
            examples = self.processor.get_dev_examples(listOfString)
        elif mode == Split.test:
            examples = self.processor.get_test_examples(listOfString)
        else:
            examples = self.processor.get_train_examples(listOfString)
        
        if limit_length is not None:
            examples = examples[:limit_length]

            
        self.features = glue_convert_examples_to_features(
            examples,
            tokenizer,
            max_length=args.max_seq_length,
            label_list=label_list,
            output_mode=self.output_mode,
        )


    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]

    def get_labels(self):
        return self.label_list