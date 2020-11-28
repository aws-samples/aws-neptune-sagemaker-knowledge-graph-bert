# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# Functions essentially used in train.py


def create_sentences_out_of_dataframe(data):
    """
    Create sentences out of a dataframe of tagged data containing the columns "Words" and "Tags"
    :param data: (pandas DataFrame) where the first element of each line corresponds to a word,
    where ### is the end of a sentence word, and where the second element of each line is the tag of this word
    :return: (list of lists of tuples) tag sentences as list of lists of tuples (word, tag)
    """
    sentence_data = list(zip(data['Sentence #'], data['Word'], data['Tag']))

    tagged_sentences = []
    tag_sent = []

    for line in sentence_data:
        if line[0] == line[0]: # When we meet a "Sentence: " for a sentence start
            if tag_sent: # Other cases
                tagged_sentences.append(tag_sent)
                tag_sent = []
                tag_sent.append((line[1], line[2]))
            if not tag_sent: # First case
                tag_sent.append((line[1], line[2]))
        elif line[0] != line[0]: # Check if NaN
            tag_sent.append((line[1],line[2]))
    # Last case
    tagged_sentences.append(tag_sent)   
    
    return tagged_sentences


# Some usefull functions
def tag_sequence(sentences):
    return [[t for w, t in sentence] for sentence in sentences]


def text_sequence(sentences):
    return [[w for w, t in sentence] for sentence in sentences]


def from_iob_to_io(sentences):
    """
    Transforms the IOB tags in sentences (output of create_sentences_out_of_dataframe) to IO tags
    :param sentences: (list of list of tuples)
    :return: (list of list of tuples)
    """
    clean_sentences=[]
    for desc in sentences:
        sublist=[]
        for x in desc:
            l = list(x)
            tag = l[1]
            if 'B-' in tag:
                tag = tag.replace('B-', '')
            elif 'I-' in tag:
                tag = tag.replace('I-', '')
            elif 'b-' in tag:
                tag = tag.replace('b-', '')
            elif 'i-' in tag:
                tag = tag.replace('i-', '')
            t = tuple([l[0], tag])
            sublist.append(t)
        clean_sentences.append(sublist)
    return clean_sentences


def split(sentences, max):
    """ Splits sentences (as list of lists of tuples), to list of lists of len(max) or less """

    new = []
    for data in sentences:
        new.append(([data[x:x+max] for x in range(0, len(data), max)]))
    new = [val for sublist in new for val in sublist]
    return new


def split_and_duplicate_index(sentences, max):
    """
    Splits sentences (as list of lists of tuples), to list of lists of len(max) or less
    And keeps track of the sentence "index" (usefull for inference)
    """

    new = []
    index = []
    for i, data in enumerate(sentences):
        new.append(([data[x:x+max] for x in range(0, len(data), max)]))
        index.append([i for x in range(0, len(data), max)])
    new = [val for sublist in new for val in sublist]
    index = [val for sublist in index for val in sublist]
    return new, index
