# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import numpy as np
import tensorflow_hub as hub

from bert.tokenization import bert_tokenization

#Functions in this file are from the following two repositories:
#https://github.com/soutsios/pos-tagger-bert/blob/master/pos_tagger_bert.ipynb
#https://github.com/google-research/bert

class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.
  When running eval/predict on the TPU, we need to pad the number of examples
  to be a multiple of the batch size, because the TPU requires a fixed batch
  size. The alternative is to drop the last batch, which is bad because it means
  the entire output data won't be generated.
  We use this class instead of `None` because treating `None` as padding
  batches could cause silent errors.
  """


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


def create_tokenizer_from_hub_module(bert_path):
    """
    Get the vocab file and casing info from the Hub module.
    :param bert_path: (str) path to either internet address or s3 location of bert
    :return:
    """
    BertTokenizer = bert_tokenization.FullTokenizer
    bert_layer = hub.KerasLayer(bert_path,
                                trainable=False)

    vocabulary_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
    to_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
    tokenizer = BertTokenizer(vocabulary_file, to_lower_case)

    return tokenizer


def convert_single_example(tokenizer, example, tag2int, max_seq_length=256):
    """
    Converts a single `InputExample` into a single `InputFeatures`.
    :param tokenizer: tokenizer created by create_tokenizer_from_hub_module
    :param example: example created by convert_text_to_examples
    :param tag2int: (dict) dictionary of tags to corresponding integer conversion
    :param max_seq_length: (int) length of input example (input size of bert model)
    :return: input_ids, input_masks, segment_ids (all three as input for BERT model,
    and label_ids (true labels, useful for testing).
    At inference, we create placeholder label_ids that we don't reuse (eg '-PAD-')
    """

    if isinstance(example, PaddingInputExample):
        input_ids = [0] * max_seq_length
        input_mask = [0] * max_seq_length
        segment_ids = [0] * max_seq_length
        label_ids = [0] * max_seq_length
        return input_ids, input_mask, segment_ids, label_ids
    
    tokens_a = example.text_a
    if len(tokens_a) > max_seq_length-2:
        tokens_a = tokens_a[0 : (max_seq_length-2)]

    # Token map will be an int -> int mapping between the `orig_tokens` index and
    # the `bert_tokens` index.

    # bert_tokens == ["[CLS]", "john", "johan", "##son", "'", "s", "house", "[SEP]"]
    # orig_to_tok_map == [1, 2, 4, 6]   
    orig_to_tok_map = []              
    tokens = []
    segment_ids = []
    
    tokens.append("[CLS]")
    segment_ids.append(0)
    orig_to_tok_map.append(len(tokens)-1)
    for token in tokens_a:
        orig_to_tok_map.append(len(tokens))
        tokens.extend(tokenizer.tokenize(token))
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)
    orig_to_tok_map.append(len(tokens)-1)
    input_ids = tokenizer.convert_tokens_to_ids([tokens[i] for i in orig_to_tok_map])
    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)
    
    label_ids = []
    labels = example.label
    label_ids.append(0)
    label_ids.extend([tag2int[label] for label in labels])
    label_ids.append(0)
    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        label_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length

    return input_ids, input_mask, segment_ids, label_ids


def convert_examples_to_features(tokenizer, examples, tag2int, max_seq_length=256):
    """
    Convert a set of `InputExample`s to a list of `InputFeatures`.
    :return: numpy arrays
    """

    input_ids, input_masks, segment_ids, labels = [], [], [], []
    for example in examples:
        input_id, input_mask, segment_id, label = convert_single_example(
            tokenizer, example, tag2int, max_seq_length
        )
        input_ids.append(input_id)
        input_masks.append(input_mask)
        segment_ids.append(segment_id)
        labels.append(label)
    return (
        np.array(input_ids),
        np.array(input_masks),
        np.array(segment_ids),
        np.array(labels),
    )


def convert_text_to_examples(texts, labels):
    """
    Create InputExamples
    :param texts: (list of lists of str) list of sentences where each sentence is a list of words
    :param labels: (list of lists of str) list of lists of words/subwords labels
    :return: list of InputExample objects
    """
    InputExamples = []
    for text, label in zip(texts, labels):
        InputExamples.append(
            InputExample(guid=None, text_a=text, text_b=None, label=label)
        )
    return InputExamples
