from constants import *

import os
import subprocess

import numpy as np
np.random.seed(NP_RANDOM_SEED)
import random
random.seed(RANDOM_SEED)
import matplotlib.pyplot as plt

import re
import string
import cPickle as pickle

import codecs
import unicodedata

from nltk.tokenize.treebank import TreebankWordTokenizer
from nltk.tokenize import sent_tokenize

import gensim
from operator import itemgetter

TREEBANK_WORD_TOKENIZER = TreebankWordTokenizer()

# ==================== Sentence Tokenization Methods ====================

def get_key_strings(variation, key_strings):
    normalized_variation = variation.strip().lower()

    if len(normalized_variation.strip()) <= 1:
        return key_strings

    # Check for single token variations
    # No special breakdown for substitution variations; entire token should be matched before this method is called in is_sentence_relevant
    if ' ' not in normalized_variation:
        # Split by '_':
        if '_' in normalized_variation:
            for variation_chunk in normalized_variation.split('_'):
                key_strings = get_key_strings(variation_chunk, key_strings)
            return key_strings
        # Split by '-':
        if '-' in normalized_variation:
            for variation_chunk in normalized_variation.split('-'):
                key_strings = get_key_strings(variation_chunk, key_strings)
            return key_strings
        # Split by 'delins'
        if 'delins' in normalized_variation:
            for variation_chunk in normalized_variation.split('delins'):
                key_strings = get_key_strings(variation_chunk, key_strings)
            key_strings.append('delins')
            key_strings.append('delet')
            key_strings.append('insert')
            return key_strings

        if 'deletion' in normalized_variation:
            key_strings.append('delet')
            return key_strings

        if 'insertion' in normalized_variation:
            key_strings.append('insert')
            return key_strings

        if 'amplification' in normalized_variation:
            key_strings.append('amplif')
            return key_strings

        if 'truncation' in normalized_variation or normalized_variation == 'truncating':
            key_strings.append('trunc')
            return key_strings

        # Split by 'del'
        if 'del' in normalized_variation:
            for variation_chunk in normalized_variation.split('del'):
                key_strings = get_key_strings(variation_chunk, key_strings)
            key_strings.append('delet')
            return key_strings
        # Split by 'ins'
        if 'ins' in normalized_variation:
            for variation_chunk in normalized_variation.split('ins'):
                key_strings = get_key_strings(variation_chunk, key_strings)
            key_strings.append('insert')
            return key_strings
        # Split by 'dup'
        if 'dup' in normalized_variation:
            for variation_chunk in normalized_variation.split('dup'):
                key_strings = get_key_strings(variation_chunk, key_strings)
            key_strings.append('duplica')
            return key_strings
        # Split by ';' (alleles)
        if ';' in normalized_variation:
            for variation_chunk in normalized_variation.split(';'):
                key_strings = get_key_strings(variation_chunk, key_strings)
            key_strings.append('allele')
            return key_strings
        # Split by 'fs'
        if ';' in normalized_variation:
            for variation_chunk in normalized_variation.split('fs'):
                key_strings = get_key_strings(variation_chunk, key_strings)
            key_strings.append('fs')
            key_strings.append('frame')
            key_strings.append('shift')
            return key_strings
        # Split by 'ext'
        if ';' in normalized_variation:
            for variation_chunk in normalized_variation.split('ext'):
                key_strings = get_key_strings(variation_chunk, key_strings)
            key_strings.append('exten')
            return key_strings

        key_strings.append(normalized_variation)
        return key_strings

    # Check for 'exon #'
    exon_match = re.search('exon [0-9]+', normalized_variation)
    if exon_match is not None:
        exon = exon_match.group(0)
        key_strings.append(exon)
        for remaining_variation_chunk in normalized_variation.split(exon):
            key_strings = get_key_strings(remaining_variation_chunk, key_strings)
        return key_strings

    # Check for non-exon related multi-token variations
    for variation_chunk in normalized_variation.split(' '):
        key_strings = get_key_strings(variation_chunk, key_strings)

    return key_strings

def is_sentence_relevant(sentence, gene, variation):
    normalized_sentence = sentence.strip().lower()

    if gene is not None:
        if gene.lower() in normalized_sentence:
            return True, gene.lower()

    if variation is not None:
        if variation.lower() in normalized_sentence:
            return True, variation.lower()

        for key_string in get_key_strings(variation, []):
            if key_string in normalized_sentence:
                return True, key_string

    return False, None

def select_relevant_sentences(row, sentences, max_flank_size=2, max_document_length=MAX_DOCUMENT_LENGTH):
    """
    Let N = len(sentences)
        D = MAX_DOCUMENT_LENGTH
        f_emp = flank_size (empirical)
        f_int = the optimal integer flank_size
        f_max = max_flank_size
        f = the actual f we want
    To retrieve as many sentences as possible to reach D:
        N(1 + 2*f_emp) <= D
        f_emp <= ((D / N) - 1) / 2
    which means effectively, for integer f, we want:
        f_int = max(0, floor(f_emp))
              Implementation Detail: f_emp = (D - N)/(2N) (== f_int in python if positive) because numerator and denominator are both python int. If the division is a positive int, the result is the same as floor
    and taking f_max into account that is set by user, the final f we want is:
        f = min(f_max, f_int)
    """

    flank_size = min(max_flank_size, max(0, (MAX_DOCUMENT_LENGTH - len(sentences)) / (2 * len(sentences))))

    gene = row['Gene']
    variation = row['Variation']
    relevant_gene_sentences_ind = []
    gene_key_string_sentences_ind_keystr = {}
    relevant_variation_sentences_ind = []
    var_key_string_sentences_ind_keystr = {}
    for i, sentence in enumerate(sentences):
        is_gene_sent_relevant, gene_key_string = is_sentence_relevant(sentence, gene, None)
        is_var_sent_relevant, var_key_string = is_sentence_relevant(sentence, None, variation)
        if is_gene_sent_relevant:
            gene_key_string_sentences_ind_keystr[i] = gene_key_string
            for offset in xrange(-flank_size, flank_size + 1):
                relevant_gene_sentences_ind.append(min(max(0, i + offset), len(sentences) - 1))
        if is_var_sent_relevant:
            var_key_string_sentences_ind_keystr[i] = var_key_string
            for offset in xrange(-flank_size, flank_size + 1):
                relevant_variation_sentences_ind.append(min(max(0, i + offset), len(sentences) - 1))

    relevant_gene_texts = []
    document_relevant_gene_sentences_ind = sorted(list(set(relevant_gene_sentences_ind)))
    for i in document_relevant_gene_sentences_ind:
        key_str = None
        if i in gene_key_string_sentences_ind_keystr:
            key_str = gene_key_string_sentences_ind_keystr[i]
        relevant_gene_texts.append((sentences[i], key_str))

    relevant_variation_texts = []
    document_relevant_var_sentences_ind = sorted(list(set(relevant_variation_sentences_ind)))
    for i in document_relevant_var_sentences_ind:
        key_str = None
        if i in var_key_string_sentences_ind_keystr:
            key_str = var_key_string_sentences_ind_keystr[i]
        relevant_variation_texts.append((sentences[i], key_str))

    return relevant_gene_texts, relevant_variation_texts, document_relevant_gene_sentences_ind, document_relevant_var_sentences_ind

# modified forms of functions from https://github.com/spyysalo/unicode2ascii/blob/master/unicode2ascii.py
def wide_unichr(i):
    try:
        return unichr(i)
    except ValueError:
        return (r'\U' + hex(i)[2:].zfill(8)).decode('unicode-escape')

def read_entities_mapping(entities_path=ENTITIES_PATH):
    mapping = {}

    with open(ENTITIES_PATH, 'rb') as f:
        # read in the replacement data
        linere = re.compile(r'^([0-9A-Za-z]{4,})\t(.*)$')

        for i, l in enumerate(f):
            # ignore lines starting with "#" as comments
            if len(l) != 0 and l[0] == "#":
                continue

            m = linere.match(l)
            assert m, "Format error in %s line %s: '%s'" % (fn, i+1, l.replace("\n","").encode("utf-8"))
            c, r = m.groups()

            c = wide_unichr(int(c, 16))
            assert c not in mapping or mapping[c] == r, "ERROR: conflicting mappings for %.4X: '%s' and '%s'" % (ord(c), mapping[c], r)

            # exception: literal '\n' maps to newline
            if r == '\\n':
                r = '\n'

            mapping[c] = r

    return mapping

def unicode2ascii(text, unicode_ascii_mapping):
    ascii_text = ''
    for c in text:
        if ord(c) >= 128:
            if c in unicode_ascii_mapping:
                c = unicode_ascii_mapping[c]
            else:
                c = unicodedata.normalize('NFKD', c).encode('ASCII', 'ignore')
        ascii_text += c
    return ascii_text

def load_word_vectors(embeddings_bin_path, is_gensim_model=False, binary=True):
    if is_gensim_model:
        word_vectors = gensim.models.Word2Vec.load(embeddings_bin_path).wv
    else:
        word_vectors = gensim.models.KeyedVectors.load_word2vec_format(embeddings_bin_path, binary=binary)
    return word_vectors

def get_allowed_vocabulary(embeddings_bin_path=CUSTOM_WORD_EMBEDDINGS_PATH, is_gensim_model=False, binary=True):
    print "Loading pretrained word embeddings..."
    word_vectors = load_word_vectors(embeddings_bin_path, is_gensim_model, binary)
    return word_vectors.vocab


def word_tokenize(sent):
    return TREEBANK_WORD_TOKENIZER.tokenize(sent)

def _sentence_list_to_words_file(sentence_list, max_document_length, outputf, key_word_indicator=None, verbose=1):
    refined_sentence_list = [] # for visualization/reconstruction purpose
    with open(outputf, 'ab') as o:
        for i in xrange(min(max_document_length, len(sentence_list))):
            sentence, key_str = sentence_list[i]
            sentence = sentence.lower()
            sentence = sentence.replace('-', ' ').replace(',', ' ') # for things like 'anti-cancer' and '10,11'
            raw_words = word_tokenize(sentence)

            if key_word_indicator is not None:
                temp_raw_words = []
                for word in raw_words:
                    if key_str is not None and key_str in word:
                        temp_raw_words.append(key_word_indicator)
                    temp_raw_words.append(word)
                raw_words = temp_raw_words

            refined_words = [] # for visualization/reconstruction purpose
            for word in raw_words:
                word = word.lstrip('0123456789.,')
                if len(word) > 1: # rid things of like '1a' in 'Fig 1a', and plain numerical words. But also remove some one letter words like 'a' and punctuations but that's probably fine
                    o.write(word + '\n')
                    refined_words.append(word)
                else:
                    refined_words.append((word, None))
            o.write('\n') # new sentence
            refined_sentence_list.append(refined_words)
        o.write('<NEWDOCUMENT>\n') # new document
    return refined_sentence_list

def get_relevant_words(data, gene_relevant_words_file, var_relevant_words_file, gene_lemmatized_words_file, var_lemmatized_words_file, max_sentence_length=MAX_SENTENCE_LENGTH, max_document_length=MAX_DOCUMENT_LENGTH, permissible_vocabulary=None, max_relevance_flank_size=2, long_sent_treatment='truncate', verbose=1):
    unicode_ascii_mapping = read_entities_mapping(ENTITIES_PATH)

    # for visualization/reconstruction purpose
    document_gene_sentence_words = []
    document_var_sentence_words = []

    for index, row in data.iterrows():
        if verbose and index % 100 == 0:
            print index
        text = unicode2ascii(unicode(row['Text'], "utf-8").encode('ASCII', 'ignore'), unicode_ascii_mapping)
        sentences = sent_tokenize(text)
        relevant_gene_sentences, relevant_variation_sentences, document_relevant_gene_sentences_ind, document_relevant_var_sentences_ind = select_relevant_sentences(row, sentences, max_relevance_flank_size)

        refined_gene_sentence_list = _sentence_list_to_words_file(relevant_gene_sentences, max_document_length, gene_relevant_words_file, GENE_KEY_WORD, verbose)
        refined_var_sentence_list = _sentence_list_to_words_file(relevant_variation_sentences, max_document_length, var_relevant_words_file, VAR_KEY_WORD, verbose)

        # for visualization/reconstruction purpose
        gene_document = list(sentences)
        for i, ind in enumerate(document_relevant_gene_sentences_ind):
            if i < len(refined_gene_sentence_list):
                gene_document[ind] = (gene_document[ind], refined_gene_sentence_list[i])
            else:
                gene_document[ind] = (gene_document[ind], None) # relevant sentences that were skipped due to MAX_DOCUMENT_LENGTH cap
        document_gene_sentence_words.append(gene_document)
        var_document = list(sentences)
        for i, ind in enumerate(document_relevant_var_sentences_ind):
            if i < len(refined_var_sentence_list):
                var_document[ind] = (var_document[ind], refined_var_sentence_list[i])
            else:
                var_document[ind] = (var_document[ind], None) # relevant sentences that were skipped due to MAX_DOCUMENT_LENGTH cap
        document_var_sentence_words.append(var_document)

    if verbose:
        print "Biolemmatizing gene relevant words..."
    biolemmatize(gene_relevant_words_file, gene_lemmatized_words_file)
    if verbose:
        print "Biolemmatizing var relevant words..."
    biolemmatize(var_relevant_words_file, var_lemmatized_words_file)

    if verbose:
        print "Processing biolemmatized outputs...Gene"
    gene_relevant_words, viz_gene_relevant_words = biolemmatize_output_to_text_words_list(gene_lemmatized_words_file, max_sentence_length, permissible_vocabulary, GENE_KEY_WORD, long_sent_treatment)
    if verbose:
        print "Processing biolemmatized outputs...Var"
    variation_relevant_words, viz_variation_relevant_words = biolemmatize_output_to_text_words_list(var_lemmatized_words_file, max_sentence_length, permissible_vocabulary, VAR_KEY_WORD, long_sent_treatment)

    # For visualization purpose:
    for i, gene_document in enumerate(document_gene_sentence_words):
        j = 0
        for sentence in gene_document:
            if isinstance(sentence, tuple) and sentence[1] is not None:
                words_list = sentence[1]
                final_words_list = viz_gene_relevant_words[i][j]
                start_k = 0
                for final_word in final_words_list:
                    for k, word in enumerate(words_list):
                        if k >= start_k and isinstance(word, str):
                            if word == final_word:
                                start_k = k + 1
                                break
                            else:
                                words_list[k] = (words_list[k], None)
                for k in xrange(start_k, len(words_list)): # leftover words to leave out, due to truncation of MAX_SENTENCE_LENGTH words cap
                    if isinstance(words_list[k], str):
                        words_list[k] = (words_list[k], None)
                j += 1

    for i, var_document in enumerate(document_var_sentence_words):
        j = 0
        for sentence in var_document:
            if isinstance(sentence, tuple) and sentence[1] is not None:
                words_list = sentence[1]
                final_words_list = viz_variation_relevant_words[i][j]
                start_k = 0
                for final_word in final_words_list:
                    for k, word in enumerate(words_list):
                        if k >= start_k and isinstance(word, str):
                            if word == final_word:
                                start_k = k + 1
                                break
                            else:
                                words_list[k] = (words_list[k], None)
                for k in xrange(start_k, len(words_list)): # leftover words to leave out, due to truncation of MAX_SENTENCE_LENGTH words cap
                    if isinstance(words_list[k], str):
                        words_list[k] = (words_list[k], None)
                j += 1

    return gene_relevant_words, variation_relevant_words, document_gene_sentence_words, document_var_sentence_words

def biolemmatize(words_file, output_file):
    with open(os.devnull, 'w') as devnull:
        command = 'java -Xmx1G -jar ' + BIOLEMMATIZER_PATH + ' -a -i ' + words_file + ' -o ' + output_file
        subprocess.check_call(command, shell=True, stdout=devnull, stderr=subprocess.STDOUT)

def biolemmatize_output_to_text_words_list(boutput_file, max_sentence_length=MAX_SENTENCE_LENGTH, permissible_vocabulary=None, key_word_indicator=None, long_sent_treatment='truncate'):
    relevant_text_list = []
    relevant_sentence_list = []
    relevant_words_list = []

    # For visualization purpose
    viz_relevant_text_list = []
    viz_relevant_sentence_list = []
    viz_relevant_words_list = []

    omitting = False
    with open(boutput_file, 'rb') as bo:
        i = 0
        for result_line in bo:
            if result_line == '\n':
                relevant_sentence_list.append(relevant_words_list)
                relevant_words_list = []

                viz_relevant_sentence_list.append(viz_relevant_words_list)
                viz_relevant_words_list = []

                omitting = False
            elif result_line.split('\t')[0] == '<NEWDOCUMENT>':
                relevant_text_list.append(relevant_sentence_list)
                relevant_sentence_list = []

                viz_relevant_text_list.append(viz_relevant_sentence_list)
                viz_relevant_sentence_list = []

                i += 1
                if i % 100 == 0:
                    print i
            else:
                if not omitting:
                    if len(relevant_words_list) >= max_sentence_length:
                        if long_sent_treatment == 'truncate':
                            continue
                        elif long_sent_treatment == 'omit':
                            relevant_words_list = []
                            viz_relevant_words_list = []
                            omitting = True
                            continue
                    raw_word, result_line = result_line.split('\t')
                    # result_line = result_line.split('\t')[1]
                    if '||' not in result_line:
                        representative_lemma = result_line.rstrip().split(' ')[0]
                    else:
                        results_count = {}
                        for result in result_line.rstrip().split('||'):
                            lemmatized_word = result.split(' ')[0]
                            if lemmatized_word in results_count:
                                results_count[lemmatized_word] += 1
                            else:
                                results_count[lemmatized_word] = 1
                        results_descriptor = [(-results_count[result], len(result), result) for result in results_count]
                        representative_lemma = sorted(results_descriptor, key=itemgetter(0, 1, 2))[0][-1] # sorted by frequency (larger better), length of word (shorter better), alphabetical
                    if permissible_vocabulary is not None:
                        if representative_lemma in permissible_vocabulary or representative_lemma == key_word_indicator:
                            relevant_words_list.append(representative_lemma)
                            viz_relevant_words_list.append(raw_word)
                    else:
                        relevant_words_list.append(representative_lemma)
                        viz_relevant_words_list.append(raw_word)
    return relevant_text_list, viz_relevant_text_list  

def generate_word_index(relevant_words):
    word_index = {}
    i = 1
    for text in relevant_words:
        for sentence in text:
            for word in sentence:
                if word not in word_index:
                    word_index[word] = i
                    i += 1
    return word_index

def texts_to_sequences(word_index, relevant_words, max_sentence_length, max_document_length):
    sequence_matrix = np.zeros((len(relevant_words), max_document_length, max_sentence_length), dtype='int32')
    for i, text in enumerate(relevant_words):
        for j, sentence in enumerate(text):
            for k, word in enumerate(sentence):
                if word in word_index:
                    sequence_matrix[i, j, k] = word_index[word]
    return sequence_matrix

def generate_final_text_data(train_data, test_data, test2_data=None,
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
    max_sentence_length=MAX_SENTENCE_LENGTH, max_document_length=MAX_DOCUMENT_LENGTH, permissible_vocabulary=None, max_relevance_flank_size=2, long_sent_treatment='truncate', verbose=1):

    train_gene_relevant_words, train_variation_relevant_words, document_train_gene_sentence_words, document_train_var_sentence_words = get_relevant_words(train_data, train_gene_relevant_words_file, train_var_relevant_words_file, train_gene_lemmatized_words_file, train_var_lemmatized_words_file, max_sentence_length, max_document_length, permissible_vocabulary, max_relevance_flank_size, long_sent_treatment, verbose)

    test_gene_relevant_words, test_variation_relevant_words, document_test_gene_sentence_words, document_test_var_sentence_words = get_relevant_words(test_data, test_gene_relevant_words_file, test_var_relevant_words_file, test_gene_lemmatized_words_file, test_var_lemmatized_words_file, max_sentence_length, max_document_length, permissible_vocabulary, max_relevance_flank_size, long_sent_treatment, verbose)
    
    test2_gene_relevant_words = None
    test2_variation_relevant_words = None
    test2_gene_text = None
    test2_variation_text = None
    document_test2_gene_sentence_words = None
    document_test2_var_sentence_words = None
    if test2_data is not None:
        test2_gene_relevant_words, test2_variation_relevant_words, document_test2_gene_sentence_words, document_test2_var_sentence_words = get_relevant_words(test2_data, test2_gene_relevant_words_file, test2_var_relevant_words_file, test2_gene_lemmatized_words_file, test2_var_lemmatized_words_file, max_sentence_length, max_document_length, permissible_vocabulary, max_relevance_flank_size, long_sent_treatment, verbose)

    word_index = {vocab:i for i, vocab in enumerate(permissible_vocabulary)}
    word_index[GENE_KEY_WORD] = len(word_index)
    word_index[VAR_KEY_WORD] = len(word_index)

    train_gene_text = texts_to_sequences(word_index, train_gene_relevant_words, max_sentence_length, max_document_length)
    train_variation_text = texts_to_sequences(word_index, train_variation_relevant_words, max_sentence_length, max_document_length)
    test_gene_text = texts_to_sequences(word_index, test_gene_relevant_words, max_sentence_length, max_document_length)
    test_variation_text = texts_to_sequences(word_index, test_variation_relevant_words, max_sentence_length, max_document_length)

    if test2_data is not None:
        test2_gene_text = texts_to_sequences(word_index, test2_gene_relevant_words, max_sentence_length, max_document_length)
        test2_variation_text = texts_to_sequences(word_index, test2_variation_relevant_words, max_sentence_length, max_document_length)

    return train_gene_text, train_variation_text, test_gene_text, test_variation_text, test2_gene_text, test2_variation_text, word_index, document_train_gene_sentence_words, document_train_var_sentence_words, document_test_gene_sentence_words, document_test_var_sentence_words, document_test2_gene_sentence_words, document_test2_var_sentence_words

def generate_embedding_matrix(word_index, embeddings_bin_path=CUSTOM_WORD_EMBEDDINGS_PATH, embedding_dim=200, is_gensim_model=False, binary=True):
    print "Loading pretrained word embeddings..."
    embeddings_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    word_vectors = load_word_vectors(embeddings_bin_path, is_gensim_model, binary)
    print "Generating Embedding Matrix..."
    # gene_key_word_vec = np.random.uniform(-0.25,0.25,embedding_dim)
    # var_key_word_vec = np.random.uniform(-0.25,0.25,embedding_dim)
    gene_key_word_vec = word_vectors.word_vec('gene')
    var_key_word_vec = word_vectors.word_vec('mutation')
    for word, i in word_index.items():
        if word in word_vectors.vocab:
            embedding_vector = word_vectors.word_vec(word)
            # words not found in embedding index will be all-zeros.
            embeddings_matrix[i] = embedding_vector
        elif word == GENE_KEY_WORD:
            embeddings_matrix[i] = gene_key_word_vec
        elif word == VAR_KEY_WORD:
            embeddings_matrix[i] = var_key_word_vec
    return embeddings_matrix

# ==================== Visualization Methods =====================

def num_words_per_sentence_distribution(relevant_words, title='Sentence Lengths Distribution', fname='sentDist.png'):
    sentence_len = []
    for text in relevant_words:
        for sentence in text:
            sentence_len.append(len(sentence))

    n, bins, patches = plt.hist(sentence_len, 100)
    plt.xlabel('Sentence Length')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.axis([0, max(sentence_len), 0, max(n)])
    plt.savefig(fname)
    # plt.show()
    plt.close()

def num_words_per_document_distribution(relevant_words, title='Document Lengths Distribution', fname='docDist.png'):
    document_len = []
    for text in relevant_words:
        num_words = 0
        for sentence in text:
            num_words += len(sentence)
        document_len.append(num_words)

    n, bins, patches = plt.hist(document_len, 100)
    plt.xlabel('Document Length')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.axis([0, max(document_len), 0, max(n)])
    plt.savefig(fname)
    # plt.show()
    plt.close()

def num_sentences_per_document_distribution(relevant_words, title='Num Sentences per Document Distribution', fname='numSentDist.png'):
    num_sentences = [len(text) for text in relevant_words]

    n, bins, patches = plt.hist(num_sentences, 100)
    plt.xlabel('Number of Sentences')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.axis([0, max(num_sentences), 0, max(n)])
    plt.savefig(fname)
    # plt.show()
    plt.close()