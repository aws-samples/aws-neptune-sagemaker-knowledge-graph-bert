# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import os
import json
import logging

import numpy as np
import tensorflow as tf

import source.bert_preprocessing as berpre
import source.postprocessing as postpro
import source.sentence_preprocessing as senpre

logging.info("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Move to s3 as output of training job
logging.info("reading int2tag and tag2int from model artifact")
with open('/opt/ml/model/int2tag.json') as f:
    int2tag = json.load(f)
with open('/opt/ml/model/tag2int.json') as f:
    tag2int = json.load(f)
    
n_tags = len(tag2int)

logging.info("reading max sequence length from model artifact")
with open('/opt/ml/model/max_sequence_length.txt') as f:
    MAX_SEQUENCE_LENGTH = int(f.read())

logging.info("reading bert path from model artifact (path pointing to s3)")
with open('/opt/ml/model/bert_path.txt') as f:
    bert_path = f.read()
    
# Instantiate tokenizer
logging.info("instantiate bert tokenizer")
tokenizer = berpre.create_tokenizer_from_hub_module(bert_path)
logging.info("bert tokenizer instantiated correctly")


def input_handler(data, context):
    """ Pre-process request input before it is sent to TensorFlow Serving REST API
    Args:
        data (obj): the request data, in format of dict or string
        context (Context): an object containing request and configuration details
    Returns:
        (dict): a JSON-serializable dict that contains request body and headers
    """
    logging.info("Starting input handler")
    
    if context.request_content_type == 'application/json':
        # pass through json (assumes it's correctly formed)
        d = json.load(data)
    
    # Splitting input into words
    logging.info("splitting input into words")
    global ids
    global sentences
    ids = [line.get('id', '') for line in d]
    sentences = [line.get('sentence', '').split() for line in d]
    
    logging.info("sentence preprocessing (split with max sequence length)")
    global split_sentences
    global split_idx
    split_sentences, split_idx = senpre.split_and_duplicate_index(sentences, MAX_SEQUENCE_LENGTH)

    logging.info("creating tags placement (-PAD- for unlabelled data)")
    tags_placement = []
    for sentence in split_sentences:
        tags_placement.append(['-PAD-']*len(sentence))
    
    logging.info("calling convert text to examples")
    bert_example = berpre.convert_text_to_examples(split_sentences, tags_placement)
    
    logging.info("convert examples to bert features")
    (input_ids, input_masks, segment_ids, _
    ) = berpre.convert_examples_to_features(tokenizer, bert_example, tag2int, max_seq_length=MAX_SEQUENCE_LENGTH+2)
    
    logging.info("convert bert features to necessary format for tensorflow serving")
    input_ids = input_ids.tolist()
    input_masks = input_masks.tolist()
    segment_ids = segment_ids.tolist()

    result = {'inputs':
                  {"input_word_ids": input_ids,
                   "input_mask": input_masks,
                   "input_type_ids": segment_ids
                   }
              }

    result = json.dumps(result)
    logging.info("returning right input for model in TFS format")
    return result


def output_handler(data, context):
    """Post-process TensorFlow Serving output before it is returned to the client.
    Args:
        data (obj): the TensorFlow serving response
        context (Context): an object containing request and configuration details
    Returns:
        (bytes, string): data to return to client, response content type
    """
    logging.info("Entering output handler")
    
    if data.status_code != 200:
        raise ValueError(data.content.decode('utf-8'))

    logging.info("reading predictions as data.content")
    response_content_type = context.accept_header
    prediction = data.content
    pred = json.loads(prediction.decode('utf-8'))
    
    logging.info("postpro1: select argmax")
    pred_max = [list(np.array(k).argmax(-1)) for k in pred['outputs']]
    
    logging.info("postpro2: numbers to tags")
    y_pred = postpro.y2label(pred_max, int2tag, mask=0)
    
    logging.info("postpro3: remapping splits to origin index")
    flat_y_pred, _ = postpro.map_split_preds_to_idx(y_pred, split_idx)
    
    logging.info("postpro4: output formatting to dicts")
    nerc_prop_list = [postpro.preds_to_dict_single(s,y) for s,y in zip(sentences,flat_y_pred)]
    
    logging.info("postpro5: mapping back to id")
    pred_dict_list = [{'id':ids[i],'sentence':' '.join(sentences[i]),'nerc_properties':nerc_prop_list[i]} for i, x in enumerate(ids)]
    
    logging.info("postpro6: list of dicts to json")
    pred_json = json.dumps(pred_dict_list)

    return pred_json, response_content_type
