from constants_evidence_filter import *
import glob
import os
import shutil

import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

import xml.etree.ElementTree as ET

import time
import re
import codecs
import unicodedata

from operator import itemgetter

from nltk import sent_tokenize
from nltk.tokenize.treebank import TreebankWordTokenizer

import cPickle as pickle

import pubmed_parser as pp

import gensim
import random
random.seed(42)

def is_request_successful(response):
    """Returns whether response has a successful status code"""
    return response.status_code == requests.codes.ok

def get(url, params=None):
    """Requests a url, retry every 5 seconds if unsuccessful"""
    while 1:
        try:
            r = requests.get(url, params=params)
            if is_request_successful(r):
                return r
            else:
                print r.status_code, r.url, params, "Trying again in 5..."
                time.sleep(5)
        except Exception:
            print "Trying again in 5..."
            time.sleep(5)

# ===================== Get evidences from OnkoKB (NOT LABELS!!!) ============================
def get_genes_set():
    """Retrieves the set of genes in OncoKB"""
    r = get('http://oncokb.org/api/v1/genes')
    json_response = r.json()
    return set([gene_entry['hugoSymbol'] for gene_entry in json_response])

def get_gene_evidences(gene):
    """Given a gene, query OncoKB for the list of pmids associated with the gene"""
    payload = {'hugoSymbol': gene, 'source': 'oncotree'}
    r = get('http://oncokb.org/api/v1/evidences/lookup', params=payload)
    json_response = r.json()
    return [article['pmid'] for entry in json_response for article in entry['articles']]

def get_gene_evidence_set(gene_set):
    """Given a set of genes, retrieve the set of pmids associated with the set of genes"""
    evidence_list = []
    for gene in gene_set:
        evidence_list.extend(get_gene_evidences(gene))
    return set(evidence_list)

def get_all_evidence_pmids():
    """Retrieve all unique pmids cited in OncoKB"""
    gene_set = get_genes_set()
    return get_gene_evidence_set(gene_set)

# ===================== Get information from PMIDs ============================
def get_MeSH_terms(nlmcatalog_id):
    """
    Given an NLM catalog id, query NCBI for associated MeSH terms.

    Args:
        nlmcatalog_id (str): Query NLM catalog id

    Returns:
        tuple: (list: List of major MeSH terms, 
                list: List of non-major MeSH terms)
    """
    MeSH_major = []
    MeSH_other = []
    if nlmcatalog_id is not None:
        payload = {'retmode': 'xml','db': 'nlmcatalog', 'id': nlmcatalog_id}
        r = get('https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi', params=payload)
        xml_response_root = ET.fromstring(r.content)
        for MeshHeadingList in xml_response_root.iter('MeshHeadingList'):
            for MeSH in MeshHeadingList.iter('DescriptorName'):
                if MeSH.get('MajorTopicYN') == 'Y':
                    MeSH_major.append(MeSH.text)
                else:
                    MeSH_other.append(MeSH.text)
    return MeSH_major, MeSH_other

def get_pmid_summary(pmid):
    """
    Given a PMID, query NCBI for its meta data.

    Args:
        pmid (str): Query PMID

    Returns:
        dict: {'Source': list,
                'MeSH_major': list,
                'MeSH_other': list,
                'FullJournalName': list}
    """
    result = {'Source': [], 'MeSH_major': [], 'MeSH_other': [], 'FullJournalName': []}
    payload = {'db': 'pubmed', 'id': pmid}
    r = get('https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi', params=payload)
    xml_response_root = ET.fromstring(r.content)
    for item in xml_response_root.iter('Item'):
        name = item.attrib['Name']
        if name == 'Source':
            result['Source'].append(item.text)
        if name == 'NlmUniqueID':
            MeSH_major, MeSH_other = get_MeSH_terms(item.text)
            result['MeSH_major'].extend(MeSH_major)
            result['MeSH_other'].extend(MeSH_other)
        if name == 'FullJournalName':
            result['FullJournalName'].append(item.text)
    return result

def get_all_pmid_summaries(pmid_set):
    """
    Given a set of unique PMIDs, return a summary object.

    Args:
        pmid_set (set): Set of unique PMIDs

    Returns:
        dict: {'Source': set,
                'MeSH_major': set,
                'MeSH_other': set,
                'FullJournalName': set}
    """
    summaries = {'Source': set(), 'MeSH_major': set(), 'MeSH_other': set(), 'FullJournalName': set()}
    for i, pmid in enumerate(pmid_set):
        if i % 10 == 0:
            print i, len(pmid_set)
        result = get_pmid_summary(pmid)
        summaries['Source'].update(result['Source'])
        summaries['MeSH_major'].update(result['MeSH_major'])
        summaries['MeSH_other'].update(result['MeSH_other'])
        summaries['FullJournalName'].update(result['FullJournalName'])
    return summaries

# ===================== Utils ==========================
def is_journal_relevant(journal, Source, FullJournalName):
    """Returns whether a query journal name is related to Source and FullJournalName"""
    return journal is not None and len(journal) > 0 and (journal in Source or journal in FullJournalName)

def reduce_text(text, unicode_ascii_mapping, sentence_per_line=True):
    """Sanitize a given a str of text. If sentence_per_line is True, each sentence is separated with newline character"""
    cleaned_lines = ''
    lines = text.split('\n') # paragraph ish
    for line in lines:
        line = line.strip()
        if len(line) > 0:
            if line[0:8] == 'Summary ':
                line = line[8:]
            if not (line.replace(' ', '').isupper() or line.istitle()):
                if sentence_per_line:
                    cleaned_lines += '\n'.join(sent_tokenize(line)) + '\n'
                else:                   
                    if line[-1] != '.':
                        cleaned_lines += line + '. '
                    else:
                        cleaned_lines += line + ' '
    return unicode2ascii(cleaned_lines, unicode_ascii_mapping)

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
    """Given a str, map and normalize non-ascii unicode characters to ascii characters"""
    ascii_text = ''
    for c in text:
        if ord(c) >= 128:
            if c in unicode_ascii_mapping:
                c = unicode_ascii_mapping[c]
            else:
                c = unicodedata.normalize('NFKD', c).encode('ASCII', 'ignore')
        ascii_text += c
    return ascii_text

# ===================== Parse Medline ============================
def is_medline_relevant(medline_dict, evidence_pmid_summaries):
    """Returns whether a medline article is of interest, i.e. if article is from journals similar to the ones typically cited on OncoKB."""
    journal = medline_dict['journal']
    medline_ta = medline_dict['medline_ta']

    if is_journal_relevant(journal, evidence_pmid_summaries['Source'], evidence_pmid_summaries['FullJournalName']):
        return True
    if is_journal_relevant(medline_ta, evidence_pmid_summaries['Source'], evidence_pmid_summaries['FullJournalName']):
        return True

    # UNCOMMENT BELOW IF YOU ALSO WANT MeSH TERMS TO BE USED FOR RELEVANCE CRITERIA. Note: pretty slow.
    # nlm_id = medline_dict['nlm_unique_id']
    # if len(nlm_id) > 0:
    #     MeSH_major, MeSH_other = get_MeSH_terms(nlm_id)
    #     for mesh in MeSH_major + MeSH_other:
    #         if mesh in evidence_pmid_summaries['MeSH_major'] or mesh in evidence_pmid_summaries['MeSH_other']:
    #             return True
    return False

def get_relevant_medline(evidence_pmid_summaries, relevant_meta_data=None):
    """Goes through all medline articles on disk, determine relevance, and store the locations of the relevant ones for later use."""
    if relevant_meta_data is None:
        relevant_meta_data = {}
    for j, filename in enumerate(glob.iglob(os.path.join(MEDLINE_ROOT, '*.xml'))):
        print j, filename
        dicts_out = pp.parse_medline_xml(os.path.join(MEDLINE_ROOT, filename))
        for i, medline_dict in enumerate(dicts_out):
            if not (filename in relevant_meta_data and i in relevant_meta_data[filename]):
                if (is_medline_relevant(medline_dict, evidence_pmid_summaries)):
                    if filename in relevant_meta_data:
                        relevant_meta_data[filename].append(i)
                    else:
                        relevant_meta_data[filename] = [i]
    return relevant_meta_data

# ===================== Parse PubMed OA ============================
def is_pubmedOA_relevant(pubmed_xml_dict, evidence_pmid_summaries):
    """Returns whether a pubmedOA article is of interest, i.e. if article is from journals similar to the ones typically cited on OncoKB."""
    journal = pubmed_xml_dict['journal']
    return is_journal_relevant(journal, evidence_pmid_summaries['Source'], evidence_pmid_summaries['FullJournalName'])

def get_relevant_pubmedOA(evidence_pmid_summaries):
    """Goes through all pubmedOA articles on disk, determine relevance, and store the locations of the relevant ones for later use."""
    relevant_meta_data = []
    for i, directory_name in enumerate(os.listdir(PUBMEDOA_ROOT)):
        print i, directory_name
        directory_path = os.path.join(PUBMEDOA_ROOT, directory_name)
        for root, dirs, files in os.walk(directory_path):
            if len(files) > 0 and files[0].endswith('.nxml'):
                representative_file = os.path.join(root, files[0])
                pubmed_xml_dict = pp.parse_pubmed_xml(representative_file)
                if is_pubmedOA_relevant(pubmed_xml_dict, evidence_pmid_summaries):
                    relevant_meta_data.append(directory_name)
                break
    return relevant_meta_data

def clear_pubmed_dir(dir_to_keep):
    """Remove all irrelevant pubmedOA articles to save disk space"""
    dir_to_keep = set(dir_to_keep)
    for directory_name in os.listdir(PUBMEDOA_ROOT):
        if directory_name not in dir_to_keep:
            print "Removing", directory_name
            shutil.rmtree(os.path.join(PUBMEDOA_ROOT, directory_name))


# ===================== Consolidation ============================
def get_medline_text(medline_dict, unicode_ascii_mapping, sentence_per_line=True):
    """Return sanitized medline abstract text, with newline character separating each sentences"""
    abstract = medline_dict['abstract']
    return reduce_text(abstract, unicode_ascii_mapping, sentence_per_line)


def get_pubmedOA_text(pubmed_xml_dict, pubmed_paragraphs_dict, unicode_ascii_mapping, sentence_per_line=True):
    """Return sanitized pubmedOA text, with newline character separating each sentences."""
    abstract = pubmed_xml_dict['abstract']
    final_text = reduce_text(abstract, unicode_ascii_mapping)

    for paragraph in pubmed_paragraphs_dict:
        section = paragraph['section'].lower()
        if section != 'appendix' and 'author' not in section:
            text = paragraph['text']
            reduced_text = reduce_text(text, unicode_ascii_mapping, sentence_per_line)
            final_text += reduced_text
    return final_text

def consolidate_relevant_text(relevant_medline, relevant_pubmedOA, unicode_ascii_mapping, sentence_per_line=True, output_path=CONSOLIDATED_SENTENCE_TEXT):
    """Goes through all relevant medline and pubmed articles on disk, sanitize the text, and consolidate results into a file, with one sentence per line"""
    with open(output_path, 'wb') as f:
        for j, filename in enumerate(relevant_medline):
            print "medline", j
            medline_dict = pp.parse_medline_xml(os.path.join(MEDLINE_ROOT, filename))
            for i in relevant_medline[filename]:
                medline_text = get_medline_text(medline_dict[i], unicode_ascii_mapping, sentence_per_line)
                f.write(medline_text)

        for k, directory in enumerate(relevant_pubmedOA):
            print "pubmed", k
            directory_path = os.path.join(PUBMEDOA_ROOT, directory)
            for root, dirs, files in os.walk(directory_path):
                for filename in files:
                    if filename.endswith('.nxml'):
                        file_path = os.path.join(directory_path, filename)
                        pubmed_xml_dict = pp.parse_pubmed_xml(file_path)
                        pubmed_paragraphs_dict = pp.parse_pubmed_paragraph(file_path, all_paragraph=True)
                        pubmed_text = get_pubmedOA_text(pubmed_xml_dict, pubmed_paragraphs_dict, unicode_ascii_mapping, sentence_per_line)
                        f.write(pubmed_text)

def consolidated_text_to_words(inputf=CONSOLIDATED_SENTENCE_TEXT, outputf=CONSOLIDATED_WORDS_TEXT):
    """From file containing sentences per line, word tokenize and output a file with 1 word per line, and a newline character separating sentences."""
    word_tokenize = TreebankWordTokenizer().tokenize
    with open(outputf, 'wb') as o:
        with open(inputf, 'rb') as f:
            for i, sentence in enumerate(f):
                if i % 10000 == 0:
                    print i
                sentence = sentence.lower()
                sentence = sentence.replace('-', ' ').replace(',', ' ') # for things like 'anti-cancer' and '10,11'
                raw_words = word_tokenize(sentence)
                for word in raw_words:
                    word = word.lstrip('0123456789.,')
                    if len(word) > 1: # rid things of like '1a' in 'Fig 1a', and plain numerical words. But also remove some one letter words like 'a' and punctuations but that's probably fine
                        o.write(word + '\n')
                o.write('\n')

def biolemmatizer_output_to_word_list(biolemmatizer_output=LEMMATIZED_WORDS_TEXT, final_output=FINAL_WORDS_SENTENCE_TEXT):
    """
    Parses Biolemmatizer output, and output a file containing lemmatized words, 1 word per line and a newline character separating each sentence.
    To obtain the Biolemmatizer output, feed a file containing 1 word per line, and a newline character separating sentences, to BioLematizer program
    """
    with open(final_output, 'wb') as fo:
        with open(biolemmatizer_output, 'rb') as bo:
            i = 0
            for result_line in bo:
                if len(result_line) > 1:
                    result_line = result_line.split('\t')[1]
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
                    fo.write(representative_lemma + '\n')
                else:
                    fo.write('\n')
                    i += 1
                    if i % 10000 == 0:
                        print i

# ===================== Word2Vec training ============================
class BiomedicalWordSentencesShuffle(object):
    def __init__(self, file_path, buffer_size=10000):
        self.file_path = file_path
        self.buffer_size = buffer_size

    def __iter__(self):
        """
        self.file_path points to a file container 1 word per line, and a newline character separating each sentence.
        buffer_size number of sentences are read, and suffled, and then this methods yields a sentence when called.
        """
        with open(self.file_path, 'rb') as f:
            j = 0
            i = 0
            sentence_list = []
            word_list = []
            for word in f:
                if len(word) > 1: # Can rstrip before to retain non-newline single character words, e.g. 'a', but there are no single character words due to the way we preprocessed
                    word_list.append(word.rstrip())
                else:
                    j += 1
                    sentence_list.append(word_list)
                    word_list = []
                    if j >= self.buffer_size:
                        random.shuffle(sentence_list)
                        for sentence in sentence_list:
                            i += 1
                            yield sentence
                            if i % 1000000 == 0:
                                print i
                        j = 0
                        sentence_list = []


def main():
    """
    This method gathers text, and trains a word2vec model on it.
    The text gathering is as follows:
        - Query OncoKB for a list of PMIDs cited (NOT THE LABELS)
        - From the list of PMIDs, figure out which journals they are from, e.g. Nature Cell Biology, Cell Cycle, etc.
        - Gather a bunch of articles published in those journals from Medline (Pubmed) and PMC (PubmedOA). The idea is that we're gathering texts that are most related to cancer.
        - Extract text from those articles, and preprocess them (tokenize into sentences, tokenize each sentences into words, and lowercase and lemmatize the words)
        - Train word2vec embeddings on the text.

    Output: medline_SHUFFLED_biomedical_embeddings_200_lit_params_win2_90iters gensim Word2Vec model.
    """
    # ================ Generate evidence summaries ===============
    pmid_set = get_all_evidence_pmids()
    summaries = get_all_pmid_summaries(pmid_set)

    print "Pickling..."
    with open('evidence_pmid_summaries.p', 'wb') as w:
        pickle.dump(summaries, w)

    print "Loading precalculated evidence pmid summaries..."
    with open('evidence_pmid_summaries.p', 'rb') as r:
        summaries = pickle.load(r)

    # ================ Identify relevant articles ================
    print "Identifying relevant Medline articles..."
    relevant_medline = get_relevant_medline(summaries)
    with open('relevant_medline_meta_data.p', 'wb') as w:
        pickle.dump(relevant_medline, w)

    # print "Identifying relevant PubMedOA articles..."
    # relevant_pubmedOA = get_relevant_pubmedOA(summaries)
    # with open('relevant_pubmedOA_meta_data.p', 'wb') as w:
    #     pickle.dump(relevant_pubmedOA, w)
    # Save disk space, remove uneeded pubmed files
    # clear_pubmed_dir(relevant_pubmedOA)

    print "Loading relevant articles..."
    with open('relevant_medline_meta_data.p', 'rb') as r:
        relevant_medline = pickle.load(r)
    # with open('relevant_pubmedOA_meta_data.p', 'rb') as r:
    #     relevant_pubmedOA = pickle.load(r)

    relevant_pubmedOA = [] # Skipping PubMedOA articles and only use Medline ones (due to time constraints and computational power limitations). If desired, uncomment lines 382-387 or 392-393, and comment out this line to include both sets of articles.
    unicode_ascii_mapping = read_entities_mapping(ENTITIES_PATH)
    consolidate_relevant_text(relevant_medline, relevant_pubmedOA, unicode_ascii_mapping, sentence_per_line=True, output_path=CONSOLIDATED_SENTENCE_TEXT)

    # ================ Process sentences ================
    print "Process consolidated sentences and write to file..."
    consolidated_text_to_words(CONSOLIDATED_SENTENCE_TEXT, CONSOLIDATED_WORDS_TEXT) 
    biolemmatizer_output_to_word_list(LEMMATIZED_WORDS_TEXT, FINAL_WORDS_SENTENCE_TEXT)

    # ================ word2vec ================

    print "Initializing sentence generator"
    b = BiomedicalWordSentencesShuffle(FINAL_WORDS_SENTENCE_TEXT, buffer_size=2000000)

    print "Training word2vec..."
    # Paramters used are those from Chiu et al. "How to Train Good Word Embeddings for Biomedical NLP"
    model = gensim.models.Word2Vec(sentences=b, sg=1, size=200, alpha=0.05, sample=0.0001, window=2, min_count=5, negative=10, iter=60, workers=4, seed=1)

    print "Saving..."
    model.save('medline_SHUFFLED_biomedical_embeddings_200_lit_params_win2_60iters')



if __name__ == "__main__":
    main()