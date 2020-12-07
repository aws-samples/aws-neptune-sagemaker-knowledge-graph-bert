# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
import os
import argparse
import json

import pandas as pd
import tensorflow as tf

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback, TensorBoard
from tensorflow.keras.utils import to_categorical

from sklearn.metrics import recall_score, precision_score, classification_report, accuracy_score, confusion_matrix, f1_score

import source.custom_layer as custlay
import source.bert_preprocessing as berpre
import source.postprocessing as postpro
import source.sentence_preprocessing as senpre

def main(args):
    #To-do: 
    #-Fix loggin bug and switch all prints to loggers
    
    print("Container structure:")
    model_dir = args.container_model_dir
    print("internal docker model_dir:", model_dir)
    
    print("epochs: ", args.epochs)
    print("batch size: ", args.batch_size)
    
    MAX_SEQUENCE_LENGTH = args.max_sequence_length
    
    print("saving parameters necessary for inference")
    f = open(os.path.join(model_dir, "max_sequence_length.txt"),"w")
    f.write(str(MAX_SEQUENCE_LENGTH))
    f.close()
    
    f = open(os.path.join(model_dir, "bert_path.txt"),"w")
    f.write(str(args.bert_path))
    f.close()
    
    print("getting data")
    train_data = pd.read_csv(os.path.join(args.train, 'train.csv'), engine='python')
    val_data = pd.read_csv(os.path.join(args.validation, 'val.csv'), engine='python')
    test_data = pd.read_csv(os.path.join(args.eval, 'test.csv'), engine='python')
    
    print("preprocessing data")
    train_sentences = senpre.create_sentences_out_of_dataframe(train_data)
    val_sentences = senpre.create_sentences_out_of_dataframe(val_data)
    test_sentences = senpre.create_sentences_out_of_dataframe(test_data)
    
    train_sentences = senpre.from_iob_to_io(train_sentences)
    val_sentences = senpre.from_iob_to_io(val_sentences)
    test_sentences = senpre.from_iob_to_io(test_sentences)

    tags = set([item for sublist in train_sentences+test_sentences+val_sentences for _, item in sublist])
    print("number of tags after IO conversion:", str(len(tags)))
    tag2int = {}
    int2tag = {}
    for i, tag in enumerate(sorted(tags)):
        tag2int[tag] = i+1
        int2tag[i+1] = tag
    # Special character for the tags
    tag2int['-PAD-'] = 0
    int2tag[0] = '-PAD-'
    n_tags = len(tag2int)
    
    print("saving tag2int and int2tag to directory")
    j = json.dumps(tag2int)
    f = open(os.path.join(model_dir, "tag2int.json"), "w")
    f.write(j)
    f.close()
    
    j = json.dumps(int2tag)
    f = open(os.path.join(model_dir, "int2tag.json"), "w")
    f.write(j)
    f.close()
    
    print("splitting sentences")
    train_sentences = senpre.split(train_sentences, MAX_SEQUENCE_LENGTH)
    val_sentences = senpre.split(val_sentences, MAX_SEQUENCE_LENGTH)
    test_sentences = senpre.split(test_sentences, MAX_SEQUENCE_LENGTH)
    
    train_text = senpre.text_sequence(train_sentences)
    test_text = senpre.text_sequence(test_sentences)
    val_text = senpre.text_sequence(val_sentences)

    train_label = senpre.tag_sequence(train_sentences)
    test_label = senpre.tag_sequence(test_sentences)
    val_label = senpre.tag_sequence(val_sentences)
    
    # Instantiate tokenizer
    print("instantiate bert tokenizer")
    tokenizer = berpre.create_tokenizer_from_hub_module(args.bert_path)
    
    # Convert data to InputExample format
    print("convert data to bert examples")
    train_examples = berpre.convert_text_to_examples(train_text, train_label)
    test_examples = berpre.convert_text_to_examples(test_text, test_label)
    val_examples = berpre.convert_text_to_examples(val_text, val_label)
    
    # Convert to features
    print("convert to bert features")
    (train_input_ids, train_input_masks, train_segment_ids, train_labels_ids
    ) = berpre.convert_examples_to_features(tokenizer, train_examples, tag2int, max_seq_length=MAX_SEQUENCE_LENGTH+2)
    (test_input_ids, test_input_masks, test_segment_ids, test_labels_ids
    ) = berpre.convert_examples_to_features(tokenizer, test_examples, tag2int, max_seq_length=MAX_SEQUENCE_LENGTH+2)
    (val_input_ids, val_input_masks, val_segment_ids, val_labels_ids
    ) = berpre.convert_examples_to_features(tokenizer, val_examples, tag2int, max_seq_length=MAX_SEQUENCE_LENGTH+2)
    
    # One-hot encode labels
    print("convert labels to categorical")
    train_labels = to_categorical(train_labels_ids, num_classes=n_tags)
    test_labels = to_categorical(test_labels_ids, num_classes=n_tags)
    val_labels = to_categorical(val_labels_ids, num_classes=n_tags)

    print('bert tokenization over')
    print("configuring model")  
    
    model = custlay.build_model(max_seq_length = MAX_SEQUENCE_LENGTH+2,
                                n_tags=n_tags,
                                lr=args.learning_rate,
                                drop_out=args.drop_out,
                                bert_path=args.bert_path
                               )
    
    print("start training")
    print("temporary weights will be saved to:", (os.path.join(model_dir, 'ner_model.h5')))
    
    cp = ModelCheckpoint(filepath=os.path.join(model_dir, 'ner_model.h5'),
                         monitor='val_accuracy',
                         save_best_only=True,
                         save_weights_only=True,
                         verbose=1)

    early_stopping = EarlyStopping(monitor = 'val_accuracy', patience = 5)

    history = model.fit([train_input_ids, train_input_masks, train_segment_ids], 
                        train_labels,
                        validation_data=([val_input_ids, val_input_masks, val_segment_ids], val_labels),
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                        shuffle=True,
                        verbose=1,
                        callbacks=[cp, early_stopping]
                       )
    
    print("training over")
    
    print("loading best h5 weights")
    # Reload best saved checkpoint:
    model.load_weights(os.path.join(model_dir, 'ner_model.h5'))
    
    print("content of model_dir:", (os.path.join(model_dir)))
    os.system(f'ls {model_dir}')
    
    print("save best model to ProtoBuff and right directory for TensorFlow Serving")
    # Note: This directory structure will need to be followed - see notes for the next section
    model_version = '1'
    export_dir = os.path.join(model_dir, 'model/', model_version)
    model.save(export_dir)
    print("saving done")
    
    # Reporting test set performance
    print("predicting on test set")
    y_true = test_labels.argmax(-1)
    y_pred = model.predict([test_input_ids, test_input_masks, test_segment_ids]).argmax(-1)
    
    print("creating classification report")
    out_true, out_pred = postpro.y2label_for_report(y_true, y_pred, int2tag, mask=0)
    report = classification_report(out_true, out_pred, digits=4, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    
    print(report_df)
    
    print("saving classification report to model directory")
    report_df.to_csv(os.path.join(model_dir, "classification_report.csv"))
    
    print('Removing h5 file as it is not used for Serving')
    os.system(f'rm {model_dir}/ner_model.h5')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--train',type=str,required=False,default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--validation',type=str,required=False,default=os.environ.get('SM_CHANNEL_VALIDATION'))
    parser.add_argument('--eval',type=str,required=False,default=os.environ.get('SM_CHANNEL_EVAL'))
    parser.add_argument('--container_model_dir',type=str,default=os.environ.get('SM_MODEL_DIR'), help='The directory where the model will be stored inside the docker. This folder is then compressed into a model.tar.gz sent to the s3 location associated with the training job')
    parser.add_argument('--max_sequence_length',type=int, default=70)
    parser.add_argument('--learning_rate',type=float,default=0.00004, help='Initial learning rate.')
    parser.add_argument('--epochs',type=int, default=50)
    parser.add_argument('--batch_size',type=int, default=16)
    parser.add_argument('--drop_out',type=float, default=0.0)
    parser.add_argument('--bert_path',type=str, default='https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3')
    
    args, _ = parser.parse_known_args()

    main(args)
