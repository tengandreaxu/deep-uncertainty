#!/usr/bin/env python
# coding: utf-8
import sys
sys.executable

import os
import pandas as pd
import numpy as  np
from sklearn.model_selection import train_test_split
from sklearn import metrics

from nltk import RegexpTokenizer
import os
import json
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from official import nlp
from official.nlp import optimization  # to create AdamW optmizer
from official.nlp import bert
from tensorflow import keras
import official.nlp.bert.bert_models
import official.nlp.bert.configs
import official.nlp.bert.run_classifier
import official.nlp.bert.tokenization
import official.nlp.data.classifier_data_lib
import official.nlp.modeling.losses
import official.nlp.modeling.models
import official.nlp.modeling.networks
import matplotlib.pyplot as plt

import logging
logging.basicConfig(level=logging.INFO)

def encode_sentence(s, tokenizer):
    """
    Encodes the SEP symbol, and converts
    the messages into BERT tokens
    """
    tokens = list(tokenizer.tokenize(s))
    tokens.append('[SEP]') # 102
    return tokenizer.convert_tokens_to_ids(tokens)


def bert_encode(dataset_tf, tokenizer):
    """
    Encodes the Tweets into BERT encoding
    """
    num_examples = dataset_tf.shape[0]

    # ragged tensor for non-uniform shapes
    messages = tf.ragged.constant([encode_sentence(s, tokenizer) for s in dataset_tf.message])

    # add CLS at the beginning of each sentence
    cls = [tokenizer.convert_tokens_to_ids(['[CLS]'])]*messages.shape[0]
    input_word_ids = tf.concat([cls, messages], axis=-1)

    # input_mask has the same shape of input_word_ids
    # but it has 1 whenever the corresponding input_word_ids
    # is not a padding
    input_mask = tf.ones_like(input_word_ids).to_tensor() # adds padding!

    # The input_type_ids only have one value (0) because this is a single sentence input
    # https://www.tensorflow.org/tutorials/text/classify_text_with_bert
    type_cls = tf.zeros_like(cls)
    type_s1 = tf.zeros_like(messages)
    input_type_ids = tf.concat([type_cls, type_s1], axis=-1).to_tensor()

    inputs = {
      'input_word_ids': input_word_ids.to_tensor(),
      'input_mask': input_mask,
      'input_type_ids': input_type_ids}

    return inputs

def get_bert_classifier(drop_out_rate:float, activation_function: str):
    # https://github.com/tensorflow/models/blob/master/official/nlp/bert/bert_models.py
    bert_config_file = os.path.join(gs_folder_bert, "bert_config.json")
    config_dict = json.loads(tf.io.gfile.GFile(bert_config_file).read())
    config_dict['hidden_dropout_prob'] = drop_out_rate
    config_dict['hidden_act'] = activation_function
    logging.info('Loading config:')
    logging.info(config_dict)
    bert_config = bert.configs.BertConfig.from_dict(config_dict)
    bert_classifier, bert_encoder = bert.bert_models.classifier_model(bert_config, num_labels=2)

    # load pre-trained model checkpoint
    checkpoint = tf.train.Checkpoint(model=bert_encoder)
    checkpoint.restore(os.path.join(gs_folder_bert, 'bert_model.ckpt')).assert_consumed()
    return bert_classifier, bert_encoder

def plot_history(history):
    # Plot history: MAE
    plt.plot(history.history['loss'], label='Sparse Categorical CrossEntropy (training data)')
    plt.plot(history.history['val_loss'], label='Sparse Categorical CrossEntropy (Validation)')
    plt.title('Sparse Categorical CrossEntropy')
    plt.ylabel('SCC Value')
    plt.xlabel('No. epoch')
    plt.legend(loc="upper left")
    plt.show()
    plt.close()


if __name__ == '__main__':
    """
    """
    logging.info('Start training bert')
    tf.get_logger().setLevel('ERROR')

    AMT_RESULTS = os.path.expanduser('cleaned_labeled_tweets.csv')
    dataset = pd.read_csv(AMT_RESULTS)
    dataset.message = dataset.message.astype(str)

    gs_folder_bert = "gs://cloud-tpu-checkpoints/bert/keras_bert/uncased_L-12_H-768_A-12"
    tf.io.gfile.listdir(gs_folder_bert)

    # Set up tokenizer to generate Tensorflow dataset
    tokenizer = bert.tokenization.FullTokenizer(
        vocab_file=os.path.join(gs_folder_bert, "vocab.txt"),
        do_lower_case=True)

    train, validate, test = np.split(dataset, [int(.7*len(dataset)), int(.85*len(dataset))])

    logging.info(f'Train shape: {train.shape[0]}, Validation Shape: {validate.shape[0]}, Test Shape: {test.shape[0]}')

    #train
    tweet_train = bert_encode(train, tokenizer)
    tweet_train_labels = np.asarray(train.final_label.tolist()).astype('int').reshape((-1,1))
    tweet_train['input_word_ids']

    #test
    tweet_test =bert_encode(test, tokenizer)
    tweet_test_labels = np.asarray(test.final_label.values).astype('int').reshape((-1,1))

    #validation
    tweet_validate = bert_encode(validate, tokenizer)
    tweet_validate_labels = np.asarray(validate.final_label.values).astype('int').reshape((-1,1))


    EPOCHS = [1,2,3,4,5]
    BATCH_SIZE = [16,32,64]
    DROP_OUT_RATE = [0.1, 0.2,0.25,0.3, 0.35,0.4, 0.45,0.5]
    ACTIVATION_FUNCTION = ['swish', 'gelu']
    histories = []

    best_accuracy = 0
    for epochs in EPOCHS:
        for batch_size in BATCH_SIZE:
            for drop_out_rate in DROP_OUT_RATE:
                for activation_function in ACTIVATION_FUNCTION:

                    train_data_size = len(tweet_train_labels)
                    steps_per_epoch = int(train_data_size / batch_size)
                    num_train_steps = steps_per_epoch * epochs
                    warmup_steps = int(epochs * train_data_size * 0.1 / batch_size)

                    # creates an optimizer with learning rate schedule
                    optimizer = nlp.optimization.create_optimizer(2e-5, num_train_steps=num_train_steps, num_warmup_steps=warmup_steps)

                    bert_classifier, bert_encoder = get_bert_classifier(drop_out_rate, activation_function)

                    metrics = [tf.keras.metrics.SparseCategoricalAccuracy(), tf.keras.metrics.CategoricalAccuracy()]
                    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

                    bert_classifier.compile(optimizer=optimizer,loss=loss,metrics=metrics)

                    bert_classifier.fit(
                        tweet_train,
                        tweet_train_labels,
                        validation_data=(tweet_validate, tweet_validate_labels),
                        batch_size=32,
                        epochs=epochs
                    )

                    class_names = ['NEGATIVE', 'POSITIVE']
                    y_predicted = bert_classifier.predict(tweet_test)
                    y_predicted

                    predicted = []
                    for i, logits in enumerate(y_predicted):
                        class_idx = tf.argmax(logits).numpy()
                        p = tf.nn.softmax(logits)[class_idx]
                        predicted.append(class_idx)
                        name = class_names[class_idx]
                        logging.info("Example {} prediction: {} ({:4.1f}%)".format(i, name, 100*p))

                    # Evaluate the performance of the classifier on the original Test-Set
                    output_classification_report = metrics.classification_report(
                        tweet_test_labels,
                        predicted,
                        target_names=['NEGATIVE', 'POSITIVE']
                    )

                    logging.info("---------------Classification Report----------------")
                    logging.info(output_classification_report)
                    logging.info("----------------------------------------------------")

                    # Compute the confusion matrix
                    confusion_matrix = metrics.confusion_matrix(tweet_test_labels, predicted)

                    logging.info("Confusion Matrix: True-Classes X Predicted-Classes")
                    logging.info(confusion_matrix)

                    logging.info("Matthews corrcoefficent")
                    logging.info(metrics.matthews_corrcoef(tweet_test_labels, predicted))

                    logging.info("Normalized Accuracy")
                    normalized_accuracy = metrics.accuracy_score(tweet_test_labels, predicted)
                    logging.info(normalized_accuracy)

                    params = dict()
                    params['epochs'] = epochs
                    params['batch_size'] = batch_size
                    params['dropout_rate'] = drop_out_rate
                    params['activation_function'] = activation_function
                    params['normalized_accuracy'] = normalized_accuracy
                    with open('bert_training_results.jsonl', 'a') as f:
                        json.dump(params, f)
                        f.write('\n')


                    if normalized_accuracy > best_accuracy:
                        logging.info('-'*47)
                        logging.info(f'NEW BEST ACCURACY: {normalized_accuracy}')
                        logging.info('-'* 47)



