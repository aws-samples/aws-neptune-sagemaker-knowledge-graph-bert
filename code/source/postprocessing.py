# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# Functions to map predictions back to their original sentence in NERC due to the splitting induced by the limit
# input size in BERT

def map_split_preds_to_idx(split_preds, split_idx):
    """
    Maps back the sentences being split
    (due to bert input size constraints) and their predictions,
    to the original sentence index

    :param split_preds: list of lists of tag predictions for each word
    :param split_idx: list of deduplicated indexes when splitting sentences
    :return: list of grouped predictions for each initial sentence and respective initial sentence id
    """

    idx = []
    flat_preds = []
    flat = []
    
    for i, _id in enumerate(split_idx):
        # When the i changes, we append the flat list to flat_preds
        if i != 0 and _id != split_idx[i-1]:
            flat_preds.append(flat)
            flat = split_preds[i]
            idx.append(split_idx[i-1])
        else:
            flat += split_preds[i]
    # last one:
    flat_preds.append(flat)
    idx.append(split_idx[-1])
    
    return flat_preds, idx


def y2label(y_pred, int2tag, mask=0):
    """
    Transforms numbers to the corresponding label
    :param y_pred: list of lists of tag predictions as integers for each word in a sentence
    :param int2tag: (dict) dictionary of integers to their corresponding tag
    :param mask: (int) integer corresponding to '-PAD-' tag
    :return: list of lists of tag predictions as real tag (not integer)
    """
    out_pred = []
    for pred in y_pred:
        predicted_tags = []
        for token in pred:
            if token != mask:
                predicted_tags.append(int2tag[str(token)])
        out_pred.append(predicted_tags)
    return out_pred


def preds_to_dict_single(sent, y_pred_tags):
    """
    Postprocessing to format tag predictions into a dictionary of {tagged propertis: corresponding words}
    for each initial sentence.

    Function to transform the output into a dict of Tags and Values
    Same function as preds_to_dict_single with lower casing and replacing spaces by underscores in tags

    :param sent: (list) list of initial sentences as list of words
    :param y_pred_tags: (list) list of grouped predictions created by map_split_preds_to_idx
    :return: dictionary of {tagged propertis: corresponding words}
    for each initial sentence.
    """
    properties = {}
    for word, tag in zip(sent, y_pred_tags):
        tag_values = properties.get(tag, [])# Get tag if existing, otherwise set it to an empty list
        tag_values.append(word)
        properties[tag] = tag_values

    if "O" in properties.keys():
        del properties["O"]

    conc_properties = dict((k.lower().replace(' ', '_'), list(v)) for k, v in properties.items())
        
    return conc_properties

#Used for reporting test performance 
def y2label_for_report(y_true, y_pred, int2tag, mask=0):
    zipped = zip(y_true.flat, y_pred.flat)
    out_true = []
    out_pred = []
    for zip_i in zipped:
        a, b = tuple(zip_i)
        if a != mask:
            out_true.append(int2tag[a])
            out_pred.append(int2tag[b])
    return out_true, out_pred
