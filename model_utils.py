from constants import *

import numpy as np
np.random.seed(NP_RANDOM_SEED)
import random as rn
rn.seed(RANDOM_SEED)
from constants import *
from text_preprocessing_utils import *

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Embedding, LSTM, GRU, Bidirectional, TimeDistributed, Input, Concatenate

from attention import *

def build_networks(word_index, embeddings_bin_path, embeddings_dim, is_gensim_model, binary, 
                    num_condensed_classes = 5, num_likelihood_classes=3, num_raw_classes=9, 
                    GRU_dim=25,
                    max_sentence_length=MAX_SENTENCE_LENGTH, max_document_length=MAX_DOCUMENT_LENGTH, condensed_class_network_prefix='condensed_class', likelihood_network_prefix='likelihood'):
    input_layers = build_text_input_layers(max_sentence_length, max_document_length, name_prefix='')
    gene_text_input, var_text_input = input_layers
    embedding_layer = build_embedding_layer(word_index, embeddings_bin_path, embeddings_dim, is_gensim_model, binary, False, max_sentence_length, max_document_length, name_prefix='')

    # condensed class network
    condensed_class_dual_attention, condensed_class_softmax = build_intermediate_networks(input_layers, embedding_layer, num_condensed_classes, GRU_dim, True, max_sentence_length, max_document_length, name_prefix=condensed_class_network_prefix)

    # likelihood network
    likelihood_dual_attention, likelihood_softmax = build_intermediate_networks(input_layers, embedding_layer, num_likelihood_classes, GRU_dim, True, max_sentence_length, max_document_length, name_prefix=likelihood_network_prefix)

    # final network
    final_condensed_class_dual_attention, final_condensed_class_softmax = build_intermediate_networks(input_layers, embedding_layer, num_condensed_classes, GRU_dim, True, max_sentence_length, max_document_length, name_prefix=condensed_class_network_prefix)
    final_likelihood_dual_attention, final_likelihood_softmax = build_intermediate_networks(input_layers, embedding_layer, num_likelihood_classes, GRU_dim, True, max_sentence_length, max_document_length, name_prefix=likelihood_network_prefix)
    # concatenated_views = Concatenate(trainable=True, name='concatenated_views')([final_condensed_class_dual_attention, final_likelihood_dual_attention, final_condensed_class_softmax, final_likelihood_softmax])
    concatenated_views = Concatenate(trainable=True, name='concatenated_views')([final_condensed_class_dual_attention, final_likelihood_dual_attention])
    final_softmax = append_softmax_layer(num_raw_classes, concatenated_views, True, 'final')

    condensed_class_model_layer_names = set([layer.name for layer in get_all_layers(build_model(input_layers, [condensed_class_softmax], compile_m=False))])
    likelihood_model_layer_names = set([layer.name for layer in get_all_layers(build_model(input_layers, [likelihood_softmax], compile_m=False))])

    return [gene_text_input, var_text_input], condensed_class_softmax, likelihood_softmax, final_softmax, condensed_class_model_layer_names, likelihood_model_layer_names

def build_intermediate_networks(input_layers, embedding_layer, num_classes, GRU_dim, layers_trainable, max_sentence_length, max_document_length, name_prefix=''):
    dual_attention = build_dual_attention_network(input_layers, embedding_layer,
                                layers_trainable=layers_trainable, GRU_dim=GRU_dim, max_sentence_length=max_sentence_length, max_document_length=max_document_length, name_prefix=name_prefix)
    softmax = append_softmax_layer(num_classes, dual_attention, trainable=layers_trainable, name_prefix=name_prefix)
    return dual_attention, softmax


def build_text_input_layers(max_sentence_length=MAX_SENTENCE_LENGTH, max_document_length=MAX_DOCUMENT_LENGTH, name_prefix=''):
    gene_text_input = Input(shape=(max_document_length,max_sentence_length), dtype='int32', name=name_prefix + '_gene_input')
    var_text_input = Input(shape=(max_document_length,max_sentence_length), dtype='int32', name=name_prefix + '_var_input')
    return gene_text_input, var_text_input

def build_embedding_layer(word_index, embeddings_bin_path, embeddings_dim, is_gensim_model, binary, trainable=False, max_sentence_length=MAX_SENTENCE_LENGTH, max_document_length=MAX_DOCUMENT_LENGTH, name_prefix=''):
    embedding_matrix = generate_embedding_matrix(word_index, embeddings_bin_path, embeddings_dim, is_gensim_model, binary)
    embedding_layer = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1], weights=[embedding_matrix], input_length=max_sentence_length, mask_zero=True, trainable=trainable, name=name_prefix + '_embeddings')
    return embedding_layer

def build_dual_attention_network(input_layers, embedding_layer, 
                                word_index=None, embeddings_bin_path=None, embeddings_dim=None, is_gensim_model=None, binary=None, embedding_layer_trainable=False, 
                                layers_trainable=True, GRU_dim=25, max_sentence_length=MAX_SENTENCE_LENGTH, max_document_length=MAX_DOCUMENT_LENGTH, name_prefix=''):
    if input_layers is None:
        input_layers = build_text_input_layers(max_sentence_length, max_document_length, name_prefix)
    if embedding_layer is None:
        embedding_layer = build_embedding_layer(word_index, embeddings_bin_path, embeddings_dim, is_gensim_model, binary, embedding_layer_trainable, max_sentence_length, max_document_length, name_prefix)

    gene_text_input, var_text_input = input_layers
    gene_attention = hierarchical_attention(embedding_layer, gene_text_input, name_prefix + '_gene', layers_trainable, GRU_dim, max_sentence_length)
    var_attention = hierarchical_attention(embedding_layer, var_text_input, name_prefix + '_var', layers_trainable, GRU_dim, max_sentence_length)

    dual_attention = Concatenate(trainable=layers_trainable, name=name_prefix + '_concatenated_attention_vector')([gene_attention, var_attention])
    return dual_attention

def append_softmax_layer(num_classes, previous_layer, trainable=True, name_prefix=''):
    softmax = Dense(num_classes, activation='softmax', trainable=trainable, name=name_prefix + "_softmax")(previous_layer)
    return softmax

def compile_model(model, optimizer='rmsprop'):
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['acc'])

def build_model(inputs, outputs, compile_m=True, optimizer='rmsprop', name_prefix=''):
    model = Model(inputs=inputs, outputs=outputs, name=name_prefix + '_model')
    if compile_m:
        compile_model(model, optimizer)
    return model

def get_all_layers(model):
    layers = []
    for layer in model.layers:
        append_layer_tree(layer, layers)
    return layers

def append_layer_tree(layer, layers_list):
    layers_list.append(layer)
    if hasattr(layer, 'layers'):
        for sublayer in layer.layers:
            append_layer_tree(sublayer, layers_list)
    if hasattr(layer, 'layer'):
        append_layer_tree(layer.layer, layers_list)
    if hasattr(layer, 'forward_layer'):
        append_layer_tree(layer.forward_layer, layers_list)
    if hasattr(layer, 'backward_layer'):
        append_layer_tree(layer.backward_layer, layers_list)


# def freeze_layer(layer):
#     if hasattr(layer, 'layer'):
#         freeze_layer(layer.layer)
#     if hasattr(layer, 'forward_layer'):
#         freeze_layer(layer.forward_layer)
#     if hasattr(layer, 'backward_layer'):
#         freeze_layer(layer.backward_layer)
#     layer.trainable = False

def hierarchical_attention(embedding_layer, document_input, name_prefix='', layers_trainable=True, GRU_dim=25, max_sentence_length=MAX_SENTENCE_LENGTH):
    # Word Encoder + Word Attention subnetwork (operated over a sentence)
    sentence_words_input = Input(shape=(max_sentence_length,), dtype='int32', name=name_prefix + '_sentence_words_input')
    embedded_words_sequences = embedding_layer(sentence_words_input)
    word_encoder = Bidirectional(GRU(GRU_dim, return_sequences=True, trainable=layers_trainable, name=name_prefix + '_word_gru'), trainable=layers_trainable, name=name_prefix + '_word_biRNN', merge_mode='concat')(embedded_words_sequences)
    word_attention = AttentionWithContext(trainable=layers_trainable, name=name_prefix + '_word_attention')(word_encoder)
    sentence_words_model = Model(sentence_words_input, word_attention, name=name_prefix + '_sentence_words_model')
    # sentence_words_model.trainable = layers_trainable

    # Sentence Encoder + Sentence Attention outer network (operated over sentences in a document)
    document_sentences_input = TimeDistributed(sentence_words_model, trainable=layers_trainable, name=name_prefix + '_timedistributed_document_sentences_input')(document_input)
    sentence_encoder = Bidirectional(GRU(GRU_dim, return_sequences=True, trainable=layers_trainable, name=name_prefix + '_sentence_gru'), trainable=layers_trainable, name=name_prefix + '_sentence_biRNN', merge_mode='concat')(document_sentences_input)
    sentence_attention = AttentionWithContext(trainable=layers_trainable, name=name_prefix + '_sentence_attention')(sentence_encoder)

    return sentence_attention

def get_attention_weights(encoder_output, attention_layer):
    W, b, u = attention_layer.get_weights()
    W = K.variable(W)
    b = K.variable(b)
    u = K.variable(u)

    uit = dot_product(K.variable(encoder_output), W)
    uit += b
    uit = K.tanh(uit)
    ait = K.dot(uit, u)
    a = K.exp(ait)
    a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
    return a.eval()