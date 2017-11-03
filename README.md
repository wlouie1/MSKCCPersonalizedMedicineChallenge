# MSKCC Personalized Medicine Challenge Submission

This is my solution to the [Personalized Medicine: Redefining Cancer Treatment](https://www.kaggle.com/c/msk-redefining-cancer-treatment) challenge on Kaggle. I have made some modifications to the original model that generated the final submission since the end of the competition.
The code here would generate a model (trained on 80% of the Stage 1 training data) with the following scores:
###### Raw Labels Model (9 classes)
* Validation data (20% of training data):
	* Log Loss: 1.1017
* Stage 1 test data:
	* Log Loss: 1.1013 (Accuracy: 61.1%)
* Stage 2 test data (possibly unreliable, see [this](https://www.kaggle.com/c/msk-redefining-cancer-treatment/discussion/40676) and [this](https://www.kaggle.com/c/msk-redefining-cancer-treatment/discussion/42129)):
	* Log Loss: 3.609 (Accuracy: 13.6%)

The main model outputs predictions for the given training data's 9 classes, which are (these labels are not supposed to be known to us during the competition):

1. Likely Loss-of-function
2. Likely Gain-of-function
3. Neutral
4. Loss-of-function
5. Likely Neutral
6. Inconclusive
7. Gain-of-function
8. Likely Switch-of-function
9. Switch-of-function

Which can be condensed into:
1. Loss-of-function
2. Gain-of-function
3. Neutral
4. Inconclusive
5. Switch-of-function

And likelihood:
1. Likely
2. Sure
3. Inconclusive

The code also generates 2 other models, trained on the condensed class labels and likelihood labels:
###### Condensed Labels Model (5 classes)
* Validation data (20% of training data):
	* Log Loss: 0.5538
* Stage 1 test data:
	* Log Loss: 0.6201 (Accuracy: 76.1%)
* Stage 2 test data:
	* Log Loss: 2.9057 (Accuracy: 20%)

###### Likelihood Labels Model (3 classes)
* Validation data (20% of training data):
	* Log Loss: 0.6431
* Stage 1 test data:
	* Log Loss: 0.6999 (Accuracy: 66.9%)
* Stage 2 test data:
	* Log Loss: 1.0935 (Accuracy: 34.4%)

The scores are not especially impressive (among other reasons, the model is trained on only 80% of the original training data, and hyperparameters are not tuned), but the approach is versatile and intrepretable. The solution also ONLY uses the provided text data in the training set, so there are a lot of potential improvements to be made in using the provided Gene/Variation data, and external data. Please see [here]() for more details on the approach, visualizations, and suggestions for further improvements.

## Usage
Please see [here]() for more details on the approach and visualizations.

#### Include Data Files
Populate the [`data`](data/) directory with the competition data files from both stages:
```
stage_2_private_solution.csv
stage1_solution_filtered.csv
stage2_sample_submission.csv
stage2_test_text.csv
stage2_test_variants.csv
test_text
test_variants
training_text
training_variants
```
This solution uses custom trained word2vec vectors. To produce them, populate the [`MEDLINE`](wordEmbeddings/PubmedOA_MEDLINE_XML/MEDLINE) directory with [MEDLINE](https://www.nlm.nih.gov/bsd/licensee/) articles. See [`README`](wordEmbeddings/PubmedOA_MEDLINE_XML/README) for more detailed instructions. Note that the word vectors used in this solution are trained on only MEDLINE abstracts; populate the [`PubmedOA`](wordEmbeddings/PubmedOA_MEDLINE_XML/PubmedOA) directory (also see the [`README`](wordEmbeddings/PubmedOA_MEDLINE_XML/README)) and uncomment some lines in [`evidence_filter.py`](wordEmbeddings/evidence_filter.py) if you also want to include articles from the [PubMed Open-Access (OA) subset](http://www.ncbi.nlm.nih.gov/pmc/tools/ftp/).

#### Train word2vec Model
From the [`wordEmbeddings`](wordEmbeddings/) directory, run [`evidence_filter.py`](wordEmbeddings/evidence_filter.py)
```python
python evidence_filter.py
```
What that does is:
1. Query OncoKB for a list of PMIDs cited (NOT THE LABELS).
2. From the list of PMIDs, figure out which journals they are from, e.g. Nature Cell Biology, Cell Cycle, etc. Gathering this list of journal names is the only extent at which OncoKB is used in this solution.
3. Go through all the MEDLINE articles in [`MEDLINE`](wordEmbeddings/PubmedOA_MEDLINE_XML/MEDLINE), and pick the ones that are published in those journals. The idea is that only articles that are most related to cancer are retained.
4. Extract abstract texts from the filtered MEDLINE articles.
5. Preprocess the text (tokenize into sentences, tokenize each sentence into words, lowercase and lemmatize the words using [BioLemmatizer 1.2](http://biolemmatizer.sourceforge.net/)).
6. Train a word2vec model on the resulting text using [gensim](https://radimrehurek.com/gensim/models/word2vec.html) and hyperparameters from the paper, [How to Train Good Word Embeddings for Biomedical NLP (Chiu et al.)](https://aclweb.org/anthology/W/W16/W16-2922.pdf). Due to the relatively small training set, the model is trained with 60 iterations.

Output: A saved gensim word2vec model `medline_SHUFFLED_biomedical_embeddings_200_lit_params_win2_60iters` in the [`wordEmbeddings`](wordEmbeddings/) directory.

#### Train Models
From the root directory, run [`main.py`](wordEmbeddings/main.py) with `train` flag to train and save the weights of the models:
```python
python main.py train
```

#### Visualize and Interpret Models
To visualize the model's sentence and word weights assignment to the Stage 1 and Stage 2 test data, run [`main.py`](wordEmbeddings/main.py) with `visualize` flag:

```python
python main.py visualize
```
This will generate JSON files in the [`results`](visualization/results/) directory. Host the outer [`visualization`](visualization/) directory to view the results in a browser. For example, to host and view in a browser locally using python, run the following from the [`visualization`](visualization/) directory:
```python
python -m SimpleHTTPServer
```
And then open up http://localhost:8000/ in a browser. Note that the page may take a few minutes to fully load.

## Dependencies
Python 2.7.13
* requests==2.12.4
* pandas==0.19.2
* matplotlib==2.0.0
* Keras==2.0.6 (Theano 0.9.0 backend used)
* nltk==3.2.2
* gensim==2.3.0
* numpy==1.13.1
* scikit_learn==0.19.1

[BioLemmatizer 1.2](http://biolemmatizer.sourceforge.net/) (in [`utils`](utils/) directory)

## Credits

[attention.py](attention.py) and [entities.dat](entities.dat) are borrowed from [cbaziotis's gist](https://gist.github.com/cbaziotis/7ef97ccf71cbc14366835198c09809d2) and [spyysalo's biomedical unicode2ascii project](https://github.com/spyysalo/unicode2ascii/blob/master/entities.dat) respectively.
