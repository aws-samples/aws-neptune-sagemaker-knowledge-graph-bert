# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.layers import Embedding, Dense, Input, concatenate, Layer, Lambda, Dropout, Activation
from tensorflow.keras import backend as K
import tensorflow_hub as hub

tf.random.set_seed(172)

class BertLayer(Layer):
    """
    This class defines the Bert Layer by downloading the tensorflow hub module either from
    internet or from a saved location in s3
    """
    def __init__(self, bert_path="https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3",
                 output_representation='sequence_output', trainable=True,  **kwargs):
        super(BertLayer, self).__init__(**kwargs)
        self.bert_path = bert_path
        print(self.bert_path)
        self.trainable = trainable
        self.output_representation = output_representation
        self.bert_layer = hub.KerasLayer(bert_path)
        
    def build(self, input_shape):
        """
        s = ["/cls/", "pooler_transform/","Variable"]
        for var in self.bert.variables:
            #print(var.name, var.shape)
            if any(x in var.name for x in s):
                self._non_trainable_weights.append(var)
            else:
                self._trainable_weights.append(var)

        print('Trainable weights:',len(self._trainable_weights))
        """
        super(BertLayer, self).build(input_shape)

    def call(self, model_inputs):
        inputs = [K.cast(x, dtype="int32") for x in model_inputs]
        input_word_ids, input_mask, segment_ids = inputs
        bert_inputs = dict(
            input_word_ids=input_word_ids, input_mask=input_mask, input_type_ids=segment_ids
        )

        embed_output = self.bert_layer(inputs=bert_inputs)["sequence_output"]
        
        return embed_output

    def compute_mask(self, inputs, mask=0):
        return K.not_equal(inputs[0], 0)

    def compute_output_shape(self, input_shape):
        if self.output_representation == 'pooled_output':
            return (None, 768)
        else:
            return (None, None, 768)
        
        
def build_model(max_seq_length,
                n_tags, 
                lr=0.00004,
                drop_out=0,
                bert_path="https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3"
               ):
    """
    This function builds the architecture of the model using Keras layers as well
    as the custom BertLayer
    """
    in_id = tensorflow.keras.layers.Input(shape=(max_seq_length,), name="input_word_ids")
    in_mask = tensorflow.keras.layers.Input(shape=(max_seq_length,), name="input_mask")
    in_segment = tensorflow.keras.layers.Input(shape=(max_seq_length,), name="input_type_ids")
    model_inputs = [in_id, in_mask, in_segment]
    
    bert_layer = BertLayer(bert_path=bert_path)
    bert_layer_output = bert_layer(model_inputs)
    dropout_layer = tensorflow.keras.layers.Dropout(rate=drop_out, noise_shape=None, seed=None)(bert_layer_output)
    model_outputs = tensorflow.keras.layers.Dense(n_tags, activation=tensorflow.keras.activations.softmax)(dropout_layer)
    model = tensorflow.keras.models.Model(inputs=model_inputs, outputs=model_outputs)

    model.compile(optimizer=tensorflow.keras.optimizers.Adam(lr=lr), loss=tensorflow.keras.losses.categorical_crossentropy, metrics=['accuracy'])

    model.summary(100)
    return model
