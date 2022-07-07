import logging
import os
from enum import Enum
from typing import List, Optional, Union

#necessary but not finding parent package because of ..
# from ...file_utils import is_tf_available
# from ...tokenization_utils import PreTrainedTokenizer
# from .utils import DataProcessor, InputExample, InputFeatures

from transformers import (
    is_tf_available,
    PreTrainedTokenizer,
    DataProcessor,
    InputExample,
    InputFeatures
)


if is_tf_available():
    import tensorflow as tf

logger = logging.getLogger(__name__)


def glue_convert_examples_to_features(
    examples: Union[List[InputExample], "tf.data.Dataset"],
    tokenizer: PreTrainedTokenizer,
    max_length: Optional[int] = None,
    task=None,
    label_list=None,
    output_mode=None,
):
    """
    Loads a data file into a list of ``InputFeatures``
    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length. Defaults to the tokenizer's max_len
        task: GLUE task
        label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
        output_mode: String indicating the output mode. Either ``regression`` or ``classification``
    Returns:
        If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
        containing the task-specific features. If the input is a list of ``InputExamples``, will return
        a list of task-specific ``InputFeatures`` which can be fed to the model.
    """
    if is_tf_available() and isinstance(examples, tf.data.Dataset):
        if task is None:
            raise ValueError("When calling glue_convert_examples_to_features from TF, the task parameter is required.")
        return _tf_glue_convert_examples_to_features(examples, tokenizer, max_length=max_length, task=task)
    return _glue_convert_examples_to_features(
        examples, tokenizer, max_length=max_length, label_list=label_list, output_mode=output_mode
    )


if is_tf_available():

    def _tf_glue_convert_examples_to_features(
        examples: tf.data.Dataset, tokenizer: PreTrainedTokenizer, task=str, max_length: Optional[int] = None,
    ) -> tf.data.Dataset:
        """
        Returns:
            A ``tf.data.Dataset`` containing the task-specific features.
        """
        processor = glue_processors[task]()
        examples = [processor.tfds_map(processor.get_example_from_tensor_dict(example)) for example in examples]
        features = glue_convert_examples_to_features(examples, tokenizer, max_length=max_length, task=task)

        def gen():
            for ex in features:
                yield (
                    {
                        "input_ids": ex.input_ids,
                        "attention_mask": ex.attention_mask,
                        "token_type_ids": ex.token_type_ids,
                    },
                    ex.label,
                )

        return tf.data.Dataset.from_generator(
            gen,
            ({"input_ids": tf.int32, "attention_mask": tf.int32, "token_type_ids": tf.int32}, tf.int64),
            (
                {
                    "input_ids": tf.TensorShape([None]),
                    "attention_mask": tf.TensorShape([None]),
                    "token_type_ids": tf.TensorShape([None]),
                },
                tf.TensorShape([]),
            ),
        )


def _glue_convert_examples_to_features(
    examples: List[InputExample],
    tokenizer: PreTrainedTokenizer,
    max_length: Optional[int] = None,
    label_list=None,
    output_mode=None,
):
    if max_length is None:
        max_length = tokenizer.max_len

    processor = SciProcessor()
    if label_list is None:
        label_list = processor.get_labels()
        logger.info("Using label list %s" % (label_list))
    if output_mode is None:
        output_mode = "classification"
        logger.info("Using output mode %s" % (output_mode))

    label_map = {label: i for i, label in enumerate(label_list)}

    def label_from_example(example: InputExample) -> Union[int, float, None]:
        if example.label is None:
            return None
        return label_map[example.label]
        raise KeyError(output_mode)

    labels = [label_from_example(example) for example in examples]

    batch_encoding = tokenizer(
        [(example.text_a, example.text_b) for example in examples],
        max_length=max_length,
        padding="max_length",
        truncation=True,
    )

    features = []
    for i in range(len(examples)):
        inputs = {k: batch_encoding[k][i] for k in batch_encoding}

        feature = InputFeatures(**inputs, label=labels[i])
        features.append(feature)

    for i, example in enumerate(examples[:5]):
        logger.info("*** Example ***")
        logger.info("guid: %s" % (example.guid))
        logger.info("features: %s" % features[i])

    return features


class SciProcessor(DataProcessor):
    
    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, listOfString):
        """See base class."""
        return self._create_examples(listOfString, "train")

    def get_dev_examples(self, listOfString):
        """See base class."""
        return self._create_examples(listOfString, "dev")

    def get_test_examples(self, listOfString):
        """See base class."""
        return self._create_examples(listOfString, "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        test_mode = set_type == "test"
    
        if test_mode:
            if len(lines[0]) >= 2 :
                text_index = 1
            else:
                text_index = 0
        else:
            text_index = 3


        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[text_index]
            label = None if test_mode else line[1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples



# class OutputMode(Enum):
#     classification = "classification"
#     regression = "regression"

# glue_tasks_num_labels = {
#     "cola": 2,
#     "mnli": 3,
#     "mrpc": 2,
#     "sst-2": 2,
#     "sts-b": 1,
#     "qqp": 2,
#     "qnli": 2,
#     "rte": 2,
#     "wnli": 2,
# }

# glue_processors = {
#     "cola": ColaProcessor,
#     "mnli": MnliProcessor,
#     "mnli-mm": MnliMismatchedProcessor,
#     "mrpc": MrpcProcessor,
#     "sst-2": Sst2Processor,
#     "sts-b": StsbProcessor,
#     "qqp": QqpProcessor,
#     "qnli": QnliProcessor,
#     "rte": RteProcessor,
#     "wnli": WnliProcessor,
# }

# glue_output_modes = {
#     "cola": "classification",
#     "mnli": "classification",
#     "mnli-mm": "classification",
#     "mrpc": "classification",
#     "sst-2": "classification",
#     "sts-b": "regression",
#     "qqp": "classification",
#     "qnli": "classification",
#     "rte": "classification",
#     "wnli": "classification",
# }