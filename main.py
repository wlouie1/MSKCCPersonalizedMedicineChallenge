from constants import *
import numpy as np
np.random.seed(NP_RANDOM_SEED)
import random as rn
rn.seed(RANDOM_SEED)
import pandas as pd

import os
import sys
import cPickle as pickle
import json

from text_preprocessing_utils import *
from model_utils import *

from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import log_loss, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

from keras.utils import plot_model
from keras import optimizers
from keras import backend as K
from attention import *

def load_variants():
    train_variants_stage_1 = pd.read_csv(TRAIN_VARIANTS_S1_PATH, header=0, index_col=0)
    test_variants_stage_1 = pd.read_csv(TEST_VARIANTS_S1_PATH, header=0, index_col=0)
    test_variants_stage_2 = pd.read_csv(TEST_VARIANTS_S2_PATH, header=0, index_col=0)
    return train_variants_stage_1, test_variants_stage_1, test_variants_stage_2

def load_text():
    train_text_stage_1 = pd.read_csv(TRAIN_TEXT_S1_PATH, sep="\|\|", engine='python', header=None, skiprows=1, names=['ID','Text'], index_col=0)
    test_text_stage_1 = pd.read_csv(TEST_TEXT_S1_PATH, sep="\|\|", engine='python', header=None, skiprows=1, names=['ID','Text'], index_col=0)
    test_text_stage_2 = pd.read_csv(TEST_TEXT_S2_PATH, sep="\|\|", engine='python', header=None, skiprows=1, names=['ID','Text'], index_col=0)
    return train_text_stage_1, test_text_stage_1, test_text_stage_2

def load_test_S1_labels():
    S1_labels = pd.read_csv(TEST_S1_LABELS_PATH, header=0, index_col=0)
    return pd.to_numeric(S1_labels.idxmax(axis=1).str[5:])

def load_test_S2_labels():
    S2_labels = pd.read_csv(TEST_S2_LABELS_PATH, header=0, index_col=0)
    return pd.to_numeric(S2_labels.idxmax(axis=1).str[5:])

def load_data():
    train_variants_stage_1, test_variants_stage_1, test_variants_stage_2 = load_variants()
    train_text_stage_1, test_text_stage_1, test_text_stage_2 = load_text()

    # Train data: Stage 1 only
    train_data_S1 = pd.concat([train_text_stage_1, train_variants_stage_1], axis=1)

    # Stage 1 Test data (only valid ones, excluding machine generated instances)
    test_data_S1 = pd.concat([test_text_stage_1, test_variants_stage_1], axis=1)
    test_S1_labels = load_test_S1_labels()
    test_data_S1['Class'] = test_S1_labels
    test_data_S1 = test_data_S1[test_data_S1['Class'].notnull()]
    test_data_S1['Class'] = test_data_S1['Class'].astype('int64')

    # Stage 2 Test data (only valid ones, excluding machine generated instances)
    test_data_S2 = pd.concat([test_text_stage_2, test_variants_stage_2], axis=1)
    test_S2_labels = load_test_S2_labels()
    test_data_S2['Class'] = test_S2_labels
    test_data_S2 = test_data_S2[test_data_S2['Class'].notnull()]
    test_data_S2['Class'] = test_data_S2['Class'].astype('int64')

    # Train data: Stage 1 train + Stage 1 test
    train_data_S1_train_and_test = pd.concat([train_data_S1, test_data_S1], ignore_index=True)

    # Train data: Stage 1 train + Stage 1 test + Stage 2 test
    train_data_S1_and_S2 = pd.concat([train_data_S1, test_data_S1, test_data_S2], ignore_index=True)

    return train_data_S1, test_data_S1, train_data_S1_train_and_test, test_data_S2, train_data_S1_and_S2

def get_labels(data, mode='raw'):
    classes = data['Class'].values
    if mode == 'class_condensed':
        classes_map = {1:1,
                    2:2,
                    3:3,
                    4:1,
                    5:3,
                    6:4,
                    7:2,
                    8:5,
                    9:5}
        classes = np.array([classes_map[class_val] for class_val in classes])
    elif mode == 'likelihood':
        # 1 = sure
        # 2 = likely
        # 3 = inconclusive
        classes_map = {
            1:2,
            2:2,
            3:1,
            4:1,
            5:2,
            6:3,
            7:1,
            8:2,
            9:1
        }
        classes = np.array([classes_map[class_val] for class_val in classes])
    return to_categorical(classes - 1)

def process_text_data(train_data, test_data, test2_data, embeddings_bin_path, is_gensim_model=False, binary=True):
    if os.path.isfile(TRAIN_GENE_TEXT):
        print "Loading TRAIN_GENE_TEXT..."
        with open(TRAIN_GENE_TEXT, 'rb') as r:
            train_gene_text = pickle.load(r)
        print "Loading TRAIN_VAR_TEXT..."
        with open(TRAIN_VAR_TEXT, 'rb') as r:
            train_variation_text = pickle.load(r)
        print "Loading TEST_GENE_TEXT..."
        with open(TEST_GENE_TEXT, 'rb') as r:
            test_gene_text = pickle.load(r)
        print "Loading TEST_VAR_TEXT..."
        with open(TEST_VAR_TEXT, 'rb') as r:
            test_variation_text = pickle.load(r)

        document_train_gene_sentence_words = None
        document_train_var_sentence_words = None
        document_test_gene_sentence_words = None
        document_test_var_sentence_words = None
        document_test2_gene_sentence_words = None
        document_test2_var_sentence_words = None
        print "Loading DOCUMENT_TRAIN_GENE_SENTENCE_WORDS..."
        with open(DOCUMENT_TRAIN_GENE_SENTENCE_WORDS, 'rb') as r:
            document_train_gene_sentence_words = pickle.load(r)
        print "Loading DOCUMENT_TRAIN_VAR_SENTENCE_WORDS..."
        with open(DOCUMENT_TRAIN_VAR_SENTENCE_WORDS, 'rb') as r:
            document_train_var_sentence_words = pickle.load(r)
        print "Loading DOCUMENT_TEST_GENE_SENTENCE_WORDS..."
        with open(DOCUMENT_TEST_GENE_SENTENCE_WORDS, 'rb') as r:
            document_test_gene_sentence_words = pickle.load(r)
        print "Loading DOCUMENT_TEST_VAR_SENTENCE_WORDS..."
        with open(DOCUMENT_TEST_VAR_SENTENCE_WORDS, 'rb') as r:
            document_test_var_sentence_words = pickle.load(r)
        if test2_data is not None:
            print "Loading TEST2_GENE_TEXT..."
            with open(TEST2_GENE_TEXT, 'rb') as r:
                test2_gene_text = pickle.load(r)
            print "Loading TEST2_VAR_TEXT..."
            with open(TEST2_VAR_TEXT, 'rb') as r:
                test2_variation_text = pickle.load(r)
            print "Loading DOCUMENT_TEST2_GENE_SENTENCE_WORDS..."
            with open(DOCUMENT_TEST2_GENE_SENTENCE_WORDS, 'rb') as r:
                document_test2_gene_sentence_words = pickle.load(r)
            print "Loading DOCUMENT_TEST2_VAR_SENTENCE_WORDS..."
            with open(DOCUMENT_TEST2_VAR_SENTENCE_WORDS, 'rb') as r:
                document_test2_var_sentence_words = pickle.load(r)
        print "Loading WORD_INDEX..."
        with open(WORD_INDEX, 'rb') as r:
            word_index = pickle.load(r)
    else:
        print "Loading Permissible Vocabulary..."
        permissible_vocabulary = get_allowed_vocabulary(embeddings_bin_path, is_gensim_model, binary)

        (train_gene_text, train_variation_text, 
        test_gene_text, test_variation_text, 
        test2_gene_text, test2_variation_text, word_index, 
        document_train_gene_sentence_words, document_train_var_sentence_words, 
        document_test_gene_sentence_words, document_test_var_sentence_words, 
        document_test2_gene_sentence_words, document_test2_var_sentence_words) = generate_final_text_data(train_data, test_data, test2_data,
                                        train_gene_relevant_words_file=TRAIN_GENE_RELEVANT_WORDS_FILE, 
                                        train_var_relevant_words_file=TRAIN_VAR_RELEVANT_WORDS_FILE, 
                                        test_gene_relevant_words_file=TEST_GENE_RELEVANT_WORDS_FILE, 
                                        test_var_relevant_words_file=TEST_VAR_RELEVANT_WORDS_FILE, 
                                        test2_gene_relevant_words_file=TEST2_GENE_RELEVANT_WORDS_FILE, 
                                        test2_var_relevant_words_file=TEST2_VAR_RELEVANT_WORDS_FILE,
                                        train_gene_lemmatized_words_file=TRAIN_GENE_LEMMA_WORDS_FILE, 
                                        train_var_lemmatized_words_file=TRAIN_VAR_LEMMA_WORDS_FILE, 
                                        test_gene_lemmatized_words_file=TEST_GENE_LEMMA_WORDS_FILE, 
                                        test_var_lemmatized_words_file=TEST_VAR_LEMMA_WORDS_FILE, 
                                        test2_gene_lemmatized_words_file=TEST2_GENE_LEMMA_WORDS_FILE, 
                                        test2_var_lemmatized_words_file=TEST2_VAR_LEMMA_WORDS_FILE,
                                        max_sentence_length=MAX_SENTENCE_LENGTH, max_document_length=MAX_DOCUMENT_LENGTH, 
                                        permissible_vocabulary=permissible_vocabulary, max_relevance_flank_size=2, 
                                        long_sent_treatment='truncate', verbose=1)
        print "Saving TRAIN_GENE_TEXT..."
        with open(TRAIN_GENE_TEXT, 'wb') as w:
            pickle.dump(train_gene_text, w)
        print "Saving TRAIN_VAR_TEXT..."
        with open(TRAIN_VAR_TEXT, 'wb') as w:
            pickle.dump(train_variation_text, w)
        print "Saving TEST_GENE_TEXT..."
        with open(TEST_GENE_TEXT, 'wb') as w:
            pickle.dump(test_gene_text, w)
        print "Saving TEST_VAR_TEXT..."
        with open(TEST_VAR_TEXT, 'wb') as w:
            pickle.dump(test_variation_text, w)
        print "Saving DOCUMENT_TRAIN_GENE_SENTENCE_WORDS..."
        with open(DOCUMENT_TRAIN_GENE_SENTENCE_WORDS, 'wb') as w:
            pickle.dump(document_train_gene_sentence_words, w)
        print "Saving DOCUMENT_TRAIN_VAR_SENTENCE_WORDS..."
        with open(DOCUMENT_TRAIN_VAR_SENTENCE_WORDS, 'wb') as w:
            pickle.dump(document_train_var_sentence_words, w)
        print "Saving DOCUMENT_TEST_GENE_SENTENCE_WORDS..."
        with open(DOCUMENT_TEST_GENE_SENTENCE_WORDS, 'wb') as w:
            pickle.dump(document_test_gene_sentence_words, w)
        print "Saving DOCUMENT_TEST_VAR_SENTENCE_WORDS..."
        with open(DOCUMENT_TEST_VAR_SENTENCE_WORDS, 'wb') as w:
            pickle.dump(document_test_var_sentence_words, w)
        if test2_data is not None:
            print "Saving TEST2_GENE_TEXT..."
            with open(TEST2_GENE_TEXT, 'wb') as w:
                pickle.dump(test2_gene_text, w)
            print "Saving TEST2_VAR_TEXT..."
            with open(TEST2_VAR_TEXT, 'wb') as w:
                pickle.dump(test2_variation_text, w)
            print "Saving DOCUMENT_TEST2_GENE_SENTENCE_WORDS..."
            with open(DOCUMENT_TEST2_GENE_SENTENCE_WORDS, 'wb') as w:
                pickle.dump(document_test2_gene_sentence_words, w)
            print "Saving DOCUMENT_TEST2_VAR_SENTENCE_WORDS..."
            with open(DOCUMENT_TEST2_VAR_SENTENCE_WORDS, 'wb') as w:
                pickle.dump(document_test2_var_sentence_words, w)
        print "Saving WORD_INDEX..."
        with open(WORD_INDEX, 'wb') as w:
            pickle.dump(word_index, w)

    return (train_gene_text, train_variation_text, 
        test_gene_text, test_variation_text, 
        test2_gene_text, test2_variation_text, word_index, 
        document_train_gene_sentence_words, document_train_var_sentence_words, 
        document_test_gene_sentence_words, document_test_var_sentence_words, 
        document_test2_gene_sentence_words, document_test2_var_sentence_words,
        word_index)

class MultiValModelCheckpoint(ModelCheckpoint):
    def __init__(self, filepath, test_gene_text, test_variation_text, y_true, test2_gene_text, test2_variation_text, y2_true, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        super(MultiValModelCheckpoint, self).__init__(filepath, monitor, verbose, save_best_only, save_weights_only, mode, period)
        self.test_gene_text = test_gene_text
        self.test_variation_text = test_variation_text
        self.y_true = y_true
        self.test2_gene_text = test2_gene_text
        self.test2_variation_text = test2_variation_text
        self.y2_true = y2_true

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        
        y_pred = self.model.predict([self.test_gene_text, self.test_variation_text], batch_size=32, verbose=1)
        logs['test_loss'] = log_loss(self.y_true, y_pred)
        logs['test2_loss'] = 0
        if self.test2_gene_text is not None:
            y2_pred = self.model.predict([self.test2_gene_text, self.test2_variation_text], batch_size=32, verbose=1)
            logs['test2_loss'] = log_loss(self.y2_true, y2_pred)

        super(MultiValModelCheckpoint, self).on_epoch_end(epoch, logs)

def get_class_weight(y):
    class_weight = compute_class_weight('balanced', range(y.shape[1]), np.argmax(y, axis=1))
    class_weight = {i:w for i, w in enumerate(class_weight)}
    return class_weight

def prepare_data(train_data, test_data, test2_data=None, embeddings_bin_path=CUSTOM_MEDLINE_60iter_WORD_EMBEDDINGS_PATH, embeddings_dim=CUSTOM_MEDLINE_60iter_WORD_EMBEDDINGS_DIM, is_gensim_model=True, binary=False, balance_class_weight=True):
    print "Train Shape:", train_data.shape
    print "Test Shape:", test_data.shape
    if test2_data is not None:
        print "Test2 Shape:", test2_data.shape

    train_y_raw = get_labels(train_data, mode='raw') # original 9 classes
    print "Train y raw Shape:", train_y_raw.shape
    test_y_raw = get_labels(test_data, mode='raw') # original 9 classes
    print "Test y raw Shape:", test_y_raw.shape
    test2_y_raw = get_labels(test2_data, mode='raw') # original 9 classes
    print "Test2 y raw Shape:", test2_y_raw.shape

    train_y_condensed = get_labels(train_data, mode='class_condensed') # 5 condensed classes (e.g. lump "likely GOF" with "GOF" as 1 class)
    print "Train y condensed:", train_y_condensed.shape
    test_y_condensed = get_labels(test_data, mode='class_condensed') # 5 condensed classes (e.g. lump "likely GOF" with "GOF" as 1 class)
    print "Test y condensed:", test_y_condensed.shape
    test2_y_condensed = get_labels(test2_data, mode='class_condensed') # 5 condensed classes (e.g. lump "likely GOF" with "GOF" as 1 class)
    print "Test2 y condensed:", test2_y_condensed.shape

    train_y_likelihood = get_labels(train_data, mode='likelihood') # 3 likelihood classes (i.e. "likely", "sure", "inconclusive")
    print "Train y likelihood:", train_y_likelihood.shape
    test_y_likelihood = get_labels(test_data, mode='likelihood') # 3 likelihood classes (i.e. "likely", "sure", "inconclusive")
    print "Test y likelihood:", test_y_likelihood.shape
    test2_y_likelihood = get_labels(test2_data, mode='likelihood') # 3 likelihood classes (i.e. "likely", "sure", "inconclusive")
    print "Test2 y likelihood:", test2_y_likelihood.shape

    original_text_data = process_text_data(train_data, test_data, test2_data, embeddings_bin_path, is_gensim_model, binary)
    (train_gene_text, train_variation_text, 
        test_gene_text, test_variation_text, 
        test2_gene_text, test2_variation_text, word_index, 
        document_train_gene_sentence_words, document_train_var_sentence_words, 
        document_test_gene_sentence_words, document_test_var_sentence_words, 
        document_test2_gene_sentence_words, document_test2_var_sentence_words,
        word_index) = original_text_data

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    train_index, test_index = next(sss.split(train_data, train_y_raw))
    X_train_gene_text, X_train_var_text, y_train_raw, y_train_condensed, y_train_likelihood = train_gene_text[train_index], train_variation_text[train_index], train_y_raw[train_index], train_y_condensed[train_index], train_y_likelihood[train_index]
    X_val_gene_text, X_val_var_text, y_val_raw, y_val_condensed, y_val_likelihood = train_gene_text[test_index], train_variation_text[test_index], train_y_raw[test_index], train_y_condensed[test_index], train_y_likelihood[test_index]

    if balance_class_weight:
        y_train_raw_class_weight = get_class_weight(y_train_raw)
        y_train_condensed_class_weight = get_class_weight(y_train_condensed)
        y_train_likelihood_class_weight = get_class_weight(y_train_likelihood)
    else:
        y_train_raw_class_weight = None
        y_train_condensed_class_weight = None
        y_train_likelihood_class_weight = None

    return ((X_train_gene_text, X_train_var_text, y_train_raw, y_train_condensed, y_train_likelihood), 
            (y_train_raw_class_weight, y_train_condensed_class_weight, y_train_likelihood_class_weight), 
            (X_val_gene_text, X_val_var_text, y_val_raw, y_val_condensed, y_val_likelihood), 
            original_text_data, 
            (train_y_raw, test_y_raw, test2_y_raw, train_y_condensed, test_y_condensed, test2_y_condensed, train_y_likelihood, test_y_likelihood, test2_y_likelihood))

def train(Xy_train, Xy_class_weights, Xy_val, original_text_data, original_y, word_index, embeddings_bin_path, embeddings_dim, is_gensim_model, binary, raw_model_GRU_dim=50, GRU_dim=25, perform_train=True):
    X_train_gene_text, X_train_var_text, y_train_raw, y_train_condensed, y_train_likelihood = Xy_train
    y_train_raw_class_weight, y_train_condensed_class_weight, y_train_likelihood_class_weight = Xy_class_weights
    X_val_gene_text, X_val_var_text, y_val_raw, y_val_condensed, y_val_likelihood = Xy_val
    (train_gene_text, train_variation_text, 
        test_gene_text, test_variation_text, 
        test2_gene_text, test2_variation_text, word_index, 
        document_train_gene_sentence_words, document_train_var_sentence_words, 
        document_test_gene_sentence_words, document_test_var_sentence_words, 
        document_test2_gene_sentence_words, document_test2_var_sentence_words,
        word_index) = original_text_data
    train_y_raw, test_y_raw, test2_y_raw, train_y_condensed, test_y_condensed, test2_y_condensed, train_y_likelihood, test_y_likelihood, test2_y_likelihood = original_y

    print "Building Raw Classes Model Network..."
    input_layers = build_text_input_layers(MAX_SENTENCE_LENGTH, MAX_DOCUMENT_LENGTH, name_prefix='')
    gene_text_input, var_text_input = input_layers
    embedding_layer = build_embedding_layer(word_index, embeddings_bin_path, embeddings_dim, is_gensim_model, binary, False, MAX_SENTENCE_LENGTH, MAX_DOCUMENT_LENGTH, name_prefix='')
    raw_dual_attention, raw_softmax = build_intermediate_networks(input_layers, embedding_layer, train_y_raw.shape[1], raw_model_GRU_dim, True, MAX_SENTENCE_LENGTH, MAX_DOCUMENT_LENGTH, name_prefix='raw')
    raw_model = build_model(input_layers, [raw_softmax], compile_m=True, optimizer='rmsprop')
    raw_model.summary()
    if perform_train:
        raw_checkpointer = MultiValModelCheckpoint('raw_MEDLINE60iterw2v_GRU' + str(raw_model_GRU_dim) + '.{epoch:02d}-{loss:.4f}-{val_loss:.4f}-{test_loss:.4f}-{test2_loss:.4f}.hdf5', test_gene_text, test_variation_text, test_y_raw, test2_gene_text, test2_variation_text, test2_y_raw, verbose=1, save_best_only=False)
        raw_model.fit([X_train_gene_text, X_train_var_text], y_train_raw, batch_size=32, epochs=5, validation_data=([X_val_gene_text, X_val_var_text], y_val_raw), callbacks=[raw_checkpointer], class_weight=y_train_raw_class_weight, verbose=1)
    else:
        raw_model.load_weights('raw_MEDLINE60iterw2v_GRU50.04-0.7716-1.1017-1.1013-3.6090.hdf5', by_name=True)

    print "Build Categorized Networks..."
    input_layers, condensed_class_softmax, likelihood_softmax, final_softmax, condensed_class_model_layer_names, likelihood_model_layer_names = build_networks(word_index, embeddings_bin_path, embeddings_dim, is_gensim_model, binary, 
                    num_condensed_classes=train_y_condensed.shape[1], num_likelihood_classes=train_y_likelihood.shape[1], num_raw_classes=train_y_raw.shape[1], 
                    GRU_dim=GRU_dim,
                    max_sentence_length=MAX_SENTENCE_LENGTH, max_document_length=MAX_DOCUMENT_LENGTH, condensed_class_network_prefix='condensed_class', likelihood_network_prefix='likelihood')

    print "Training Condensed Class Model..."
    condensed_class_model = build_model(input_layers, [condensed_class_softmax], compile_m=True, optimizer='rmsprop')
    condensed_class_model.summary()
    if perform_train:
        condensed_class_checkpointer = MultiValModelCheckpoint('condensed_class_MEDLINE60_GRU' + str(GRU_dim) + '.{epoch:02d}-{loss:.4f}-{val_loss:.4f}-{test_loss:.4f}-{test2_loss:.4f}.hdf5', test_gene_text, test_variation_text, test_y_condensed, test2_gene_text, test2_variation_text, test2_y_condensed, verbose=1, save_best_only=False)
        condensed_class_model.fit([X_train_gene_text, X_train_var_text], y_train_condensed, batch_size=32, epochs=6, validation_data=([X_val_gene_text, X_val_var_text], y_val_condensed), callbacks=[condensed_class_checkpointer], class_weight=y_train_condensed_class_weight, verbose=1)
    else:
        condensed_class_model.load_weights('condensed_class_MEDLINE60_GRU25.06-0.3916-0.5538-0.6201-2.9057.hdf5', by_name=True)

    print "Training Likelihood Model..."
    likelihood_model = build_model(input_layers, [likelihood_softmax], compile_m=True, optimizer='rmsprop')
    likelihood_model.summary()
    if perform_train:
        likelihood_checkpointer = MultiValModelCheckpoint('likelihood_MEDLINE60_GRU' + str(GRU_dim) + '.{epoch:02d}-{loss:.4f}-{val_loss:.4f}-{test_loss:.4f}-{test2_loss:.4f}.hdf5', test_gene_text, test_variation_text, test_y_likelihood, test2_gene_text, test2_variation_text, test2_y_likelihood, verbose=1, save_best_only=False)
        likelihood_model.fit([X_train_gene_text, X_train_var_text], y_train_likelihood, batch_size=32, epochs=6, validation_data=([X_val_gene_text, X_val_var_text], y_val_likelihood), callbacks=[likelihood_checkpointer], class_weight=y_train_likelihood_class_weight, verbose=1)
    else:
        likelihood_model.load_weights('likelihood_MEDLINE60_GRU25.05-0.5529-0.6431-0.6999-1.0935.hdf5', by_name=True)

    # print "Training Final Model..."
    # final_model = build_model(input_layers, final_softmax, compile_m=False)
    # final_model.load_weights('condensed_class_GRU25.06-0.3916-0.5538-0.6201-2.9057.hdf5', by_name=True)
    # final_model.load_weights('likelihood_GRU25.05-0.5497-0.6642-0.7447-1.2439.hdf5', by_name=True)

    # for layer in get_all_layers(final_model):
    #     if layer.name in condensed_class_model_layer_names or layer.name in likelihood_model_layer_names:
    #         layer.trainable = False
    # compile_model(final_model, optimizer='rmsprop')
    # final_model.summary()
    # final_checkpointer = MultiValModelCheckpoint('final_GRU' + str(GRU_dim) + '.{epoch:02d}-{loss:.4f}-{val_loss:.4f}-{test_loss:.4f}-{test2_loss:.4f}.hdf5', test_gene_text, test_variation_text, test_y_raw, test2_gene_text, test2_variation_text, test2_y_raw, verbose=1, save_best_only=False)
    # final_model.fit([X_train_gene_text, X_train_var_text], y_train_raw, batch_size=32, epochs=15, validation_data=([X_val_gene_text, X_val_var_text], y_val_raw), callbacks=[final_checkpointer], class_weight=y_train_raw_class_weight, verbose=1)

    return raw_model, condensed_class_model, likelihood_model

def generate_weightless_document_result(document_sentence_words):
    document_result = []
    for i, sentence in enumerate(document_sentence_words):
        sentence_object = {"weight": None}
        if isinstance(sentence, str): # irrelevant sentence
            sentence_object["words"] = sentence
            sentence_object["isRelevant"] = False
            sentence_object["isUsed"] = False
        elif isinstance(sentence, tuple):
            sentence_object["isRelevant"] = True
            if sentence[1] is None: # relevant sentence skipped due to MAX_DOCUMENT_LENGTH cap
                sentence_object["words"] = sentence[0]
                sentence_object["isUsed"] = False
            else:
                sentence_object["words"] = [{"word": word, "isUsed": True, "weight": None} if isinstance(word, str) else {"word": word[0], "isUsed": False, "weight": None} for word in sentence[1]]
                sentence_object["isUsed"] = True
        document_result.append(sentence_object)
    return document_result

def populate_weights(document_result, sentence_weights, word_weights):
    i = 0
    for sentence_object in document_result:
        if sentence_object["isUsed"]:
            sentence_object["weight"] = round(float(sentence_weights[i]), 4) # convert to float in case numpy float dtype not JSON serializable, truncate to save space
            j = 0
            for word_object in sentence_object["words"]:
                if word_object["isUsed"]:
                    word_object["weight"] = round(float(word_weights[i][j]), 4)
                    j += 1
            i += 1

def generate_text_result_json(document_sentence_words, sentence_weights, word_weights):
    document_result = generate_weightless_document_result(document_sentence_words)
    populate_weights(document_result, sentence_weights, word_weights)
    return document_result

def generate_documents_results_json(model, model_results, model_name_prefix, original_data, y, gene_text, var_text, documents_gene_sentence_words, documents_var_sentence_words):
    model_results_meta_data, model_results_gene_text, model_results_var_text = model_results

    print "Getting Prediction..."
    prediction = model.predict([gene_text, var_text], batch_size=32, verbose=1)
    logloss = log_loss(y, prediction)
    accuracy = accuracy_score(np.argmax(y, axis=1), np.argmax(prediction, axis=1))

    print "Getting attention weights..."
    gene_sentence_weights = get_sentence_weights(model, model_name_prefix + '_gene_sentence_biRNN', model_name_prefix + '_gene_sentence_attention', [gene_text, var_text])
    var_sentence_weights = get_sentence_weights(model, model_name_prefix + '_var_sentence_biRNN', model_name_prefix + '_var_sentence_attention', [gene_text, var_text])
    gene_word_weights = get_word_weights(model, model_name_prefix + '_gene_sentence_words_input', model_name_prefix + '_gene_word_biRNN', model_name_prefix + '_gene_word_attention', gene_text)
    var_word_weights = get_word_weights(model, model_name_prefix + '_var_sentence_words_input', model_name_prefix + '_var_word_biRNN', model_name_prefix + '_var_word_attention', var_text)

    for i in xrange(len(documents_gene_sentence_words)):
        meta_data = {}
        meta_data['ID'] = int(original_data.index.values[i])
        meta_data['Gene'] = str(original_data.iloc[i]['Gene'])
        meta_data['Variation'] = str(original_data.iloc[i]['Variation'])
        meta_data['logloss'] = round(float(logloss), 4)
        meta_data['accuracy'] = round(float(accuracy), 4)
        meta_data["trueClass"] = [int(t) for t in y[i]]
        meta_data["prediction"] = [round(float(p), 4) for p in prediction[i]]
        model_results_meta_data.append(meta_data)

        model_results_gene_text.append(generate_text_result_json(documents_gene_sentence_words[i], gene_sentence_weights[i], gene_word_weights[i]))
        model_results_var_text.append(generate_text_result_json(documents_var_sentence_words[i], var_sentence_weights[i], var_word_weights[i]))

def get_word_weights(model, input_layer_name, output_layer_name, attention_layer_name, data_input):
    for layer in get_all_layers(model):
        if layer.name == input_layer_name:
            word_encoder_input_layer = layer
        if layer.name == output_layer_name:
            word_encoder_output_layer = layer
        if layer.name == attention_layer_name:
            word_attention_layer = layer

    word_encoder_model = Model(inputs=word_encoder_input_layer.input, outputs=word_encoder_output_layer.output)
    word_encoder_output = word_encoder_model.predict(data_input.reshape((data_input.shape[0] * data_input.shape[1], data_input.shape[2])), batch_size=32, verbose=1)

    a = get_attention_weights(word_encoder_output, word_attention_layer)
    return a.reshape(data_input.shape)

def get_sentence_weights(model, output_layer_name, attention_layer_name, data_input):
    sentence_encoder_model = Model(inputs=model.input, outputs=model.get_layer(output_layer_name).output)
    sentence_encoder_output = sentence_encoder_model.predict(data_input, batch_size=32, verbose=1)
    a = get_attention_weights(sentence_encoder_output, model.get_layer(attention_layer_name))
    return a

def generate_results_json(original_data, original_data2, original_text_data, original_y, raw_model, condensed_class_model, likelihood_model):
    (train_gene_text, train_variation_text, 
        test_gene_text, test_variation_text, 
        test2_gene_text, test2_variation_text, word_index, 
        document_train_gene_sentence_words, document_train_var_sentence_words, 
        document_test_gene_sentence_words, document_test_var_sentence_words, 
        document_test2_gene_sentence_words, document_test2_var_sentence_words,
        word_index) = original_text_data

    train_y_raw, test_y_raw, test2_y_raw, train_y_condensed, test_y_condensed, test2_y_condensed, train_y_likelihood, test_y_likelihood, test2_y_likelihood = original_y

    raw_model_results_meta_data = []
    raw_model_results_gene_text = []
    raw_model_results_var_text = []
    raw_model_results = (raw_model_results_meta_data, raw_model_results_gene_text, raw_model_results_var_text)

    condensed_model_results_meta_data = []
    condensed_model_results_gene_text = []
    condensed_model_results_var_text = []
    condensed_model_results = (condensed_model_results_meta_data, condensed_model_results_gene_text, condensed_model_results_var_text)

    likelihood_model_results_meta_data = []
    likelihood_model_results_gene_text = []
    likelihood_model_results_var_text = []
    likelihood_model_results = (likelihood_model_results_meta_data, likelihood_model_results_gene_text, likelihood_model_results_var_text)

    generate_documents_results_json(raw_model, raw_model_results, 'raw', original_data, test_y_raw, test_gene_text, test_variation_text, document_test_gene_sentence_words, document_test_var_sentence_words)
    generate_documents_results_json(condensed_class_model, condensed_model_results, 'condensed_class', original_data, test_y_condensed, test_gene_text, test_variation_text, document_test_gene_sentence_words, document_test_var_sentence_words)
    generate_documents_results_json(likelihood_model, likelihood_model_results, 'likelihood', original_data, test_y_likelihood, test_gene_text, test_variation_text, document_test_gene_sentence_words, document_test_var_sentence_words)

    with open('visualization/results/rawModel_ResultsStage1_metaData.json', 'w+') as outfile:
        json.dump(raw_model_results_meta_data, outfile, separators=(',', ':'))
    with open('visualization/results/rawModel_ResultsStage1_geneText.json', 'w+') as outfile:
        json.dump(raw_model_results_gene_text, outfile, separators=(',', ':'))
    with open('visualization/results/rawModel_ResultsStage1_varText.json', 'w+') as outfile:
        json.dump(raw_model_results_var_text, outfile, separators=(',', ':'))

    with open('visualization/results/condensedModel_ResultsStage1_metaData.json', 'w+') as outfile:
        json.dump(condensed_model_results_meta_data, outfile, separators=(',', ':'))
    with open('visualization/results/condensedModel_ResultsStage1_geneText.json', 'w+') as outfile:
        json.dump(condensed_model_results_gene_text, outfile, separators=(',', ':'))
    with open('visualization/results/condensedModel_ResultsStage1_varText.json', 'w+') as outfile:
        json.dump(condensed_model_results_var_text, outfile, separators=(',', ':'))

    with open('visualization/results/likelihoodModel_ResultsStage1_metaData.json', 'w+') as outfile:
        json.dump(likelihood_model_results_meta_data, outfile, separators=(',', ':'))
    with open('visualization/results/likelihoodModel_ResultsStage1_geneText.json', 'w+') as outfile:
        json.dump(likelihood_model_results_gene_text, outfile, separators=(',', ':'))
    with open('visualization/results/likelihoodModel_ResultsStage1_varText.json', 'w+') as outfile:
        json.dump(likelihood_model_results_var_text, outfile, separators=(',', ':'))

    raw_model_results2_meta_data = [] # list of meta data for each document: [{"ID": int, "Gene": str, "Variation": str, trueClass": list.<int>, "prediction": list.<float>, "logloss": float, "accuracy": float}, {...}, ...]
    raw_model_results2_gene_text = [] 
    # list representing the gene text:
    # [
    #     {
    #         "words":
    #         [
    #             {
    #                 "word": str,
    #                 "isUsed": bool
    #                 "weight": float
    #             }, // word
    #             {
    #                 ...
    #             }
    #         ]|str,
    #         "isRelevant": bool
    #         "isUsed": bool
    #         "weight": float|None
    #     }, // sentence
    #     {
    #         ...
    #     }, // sentence
    #     ...
    # ]
    raw_model_results2_var_text = [] # list representing the variation text, same format as that of the list representing the gene text.
    raw_model_results2 = (raw_model_results2_meta_data, raw_model_results2_gene_text, raw_model_results2_var_text)

    condensed_model_results2_meta_data = []
    condensed_model_results2_gene_text = []
    condensed_model_results2_var_text = []
    condensed_model_results2 = (condensed_model_results2_meta_data, condensed_model_results2_gene_text, condensed_model_results2_var_text)

    likelihood_model_results2_meta_data = []
    likelihood_model_results2_gene_text = []
    likelihood_model_results2_var_text = []
    likelihood_model_results2 = (likelihood_model_results2_meta_data, likelihood_model_results2_gene_text, likelihood_model_results2_var_text)

    generate_documents_results_json(raw_model, raw_model_results2, 'raw', original_data2, test2_y_raw, test2_gene_text, test2_variation_text, document_test2_gene_sentence_words, document_test2_var_sentence_words)
    generate_documents_results_json(condensed_class_model, condensed_model_results2, 'condensed_class', original_data2, test2_y_condensed, test2_gene_text, test2_variation_text, document_test2_gene_sentence_words, document_test2_var_sentence_words)
    generate_documents_results_json(likelihood_model, likelihood_model_results2, 'likelihood', original_data2, test2_y_likelihood, test2_gene_text, test2_variation_text, document_test2_gene_sentence_words, document_test2_var_sentence_words)

    with open('visualization/results/rawModel_ResultsStage2_metaData.json', 'w+') as outfile:
        json.dump(raw_model_results2_meta_data, outfile, separators=(',', ':'))
    with open('visualization/results/rawModel_ResultsStage2_geneText.json', 'w+') as outfile:
        json.dump(raw_model_results2_gene_text, outfile, separators=(',', ':'))
    with open('visualization/results/rawModel_ResultsStage2_varText.json', 'w+') as outfile:
        json.dump(raw_model_results2_var_text, outfile, separators=(',', ':'))

    with open('visualization/results/condensedModel_ResultsStage2_metaData.json', 'w+') as outfile:
        json.dump(condensed_model_results2_meta_data, outfile, separators=(',', ':'))
    with open('visualization/results/condensedModel_ResultsStage2_geneText.json', 'w+') as outfile:
        json.dump(condensed_model_results2_gene_text, outfile, separators=(',', ':'))
    with open('visualization/results/condensedModel_ResultsStage2_varText.json', 'w+') as outfile:
        json.dump(condensed_model_results2_var_text, outfile, separators=(',', ':'))

    with open('visualization/results/likelihoodModel_ResultsStage2_metaData.json', 'w+') as outfile:
        json.dump(likelihood_model_results2_meta_data, outfile, separators=(',', ':'))
    with open('visualization/results/likelihoodModel_ResultsStage2_geneText.json', 'w+') as outfile:
        json.dump(likelihood_model_results2_gene_text, outfile, separators=(',', ':'))
    with open('visualization/results/likelihoodModel_ResultsStage2_varText.json', 'w+') as outfile:
        json.dump(likelihood_model_results2_var_text, outfile, separators=(',', ':'))


if __name__ == '__main__':
    mode = sys.argv[1]

    print "Loading Data..."
    train_data_S1, test_data_S1, train_data_S1_train_and_test, test_data_S2, train_data_S1_and_S2 = load_data()

    embeddings_bin_path = CUSTOM_MEDLINE_60iter_WORD_EMBEDDINGS_PATH
    is_gensim_model = True
    binary = False
    embeddings_dim = CUSTOM_MEDLINE_60iter_WORD_EMBEDDINGS_DIM

    print "Processing Data..."
    Xy_train, Xy_class_weights, Xy_val, original_text_data, original_y = prepare_data(train_data_S1, test_data_S1, test_data_S2, embeddings_bin_path, embeddings_dim, is_gensim_model, binary, balance_class_weight=True)

    if mode == "train":
        print "Build and Train Networks..."
        raw_model, condensed_class_model, likelihood_model = train(Xy_train, Xy_class_weights, Xy_val, original_text_data, original_y, original_text_data[-1], embeddings_bin_path, embeddings_dim, is_gensim_model, binary, raw_model_GRU_dim=50, GRU_dim=25, perform_train=True)
    elif mode == "visualize":
        print "Loading Trained Networks..."
        raw_model, condensed_class_model, likelihood_model = train(Xy_train, Xy_class_weights, Xy_val, original_text_data, original_y, original_text_data[-1], embeddings_bin_path, embeddings_dim, is_gensim_model, binary, raw_model_GRU_dim=50, GRU_dim=25, perform_train=False)

        print "Generating results JSON..."
        # Generates JSON files in ./visualization/results
        # Host the ./visualization directory to view results in a browser (e.g. by running "python -m SimpleHTTPServer" from the ./visualization directory, and going to "http://localhost:8000/" in a browser)
        generate_results_json(test_data_S1, test_data_S2, original_text_data, original_y, raw_model, condensed_class_model, likelihood_model)