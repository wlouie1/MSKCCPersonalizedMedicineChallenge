# MSKCC Personalized Medicine Challenge Submission

This is my approach to the [Personalized Medicine: Redefining Cancer Treatment](https://www.kaggle.com/c/msk-redefining-cancer-treatment) challenge on Kaggle. I have made some modifications to the original model that generated the final submission since the end of the competition.

The competition only asks for predictions for 9 classes, but one can also think of them as a combination of 5 condensed classes and 3 likelihood classes:

| Raw Classes (9) | Condensed Classes (5) | Likelihood Classes (3) |
| ------------ | ------------- | ------------- |
| Likely Loss-of-function | Loss-of-function | Likely |
| Likely Gain-of-function | Gain-of-function | Likely |
| Neutral | Neutral | Sure |
| Loss-of-function | Loss-of-function | Sure |
| Likely Neutral | Neutral | Likely |
| Inconclusive | Inconclusive | Inconclusive |
| Gain-of-function | Gain-of-function | Sure |
| Likely Switch-of-function | Switch-of-function | Likely |
| Switch-of-function | Switch-of-function | Sure |

The code here generates 3 models with the same arhitecture, each trained (on 80% of the Stage 1 training data) separately to predict the Raw, Condensed, and Likelihood classes, with the following results on the Stage 1 and Stage 2 test data (possibly unreliable, see [this](https://www.kaggle.com/c/msk-redefining-cancer-treatment/discussion/40676) and [this](https://www.kaggle.com/c/msk-redefining-cancer-treatment/discussion/42129)):

###### Raw Labels Model (9 classes)
| | Validation data (20% of train) | Stage 1 Test data | Stage 2 test data |
| ------------ | ------------ | ------------- | ------------- |
| Log Loss | 1.1017 | 1.1013 | 3.609 |
| Accuracy | - | 61.1% | 13.6% |

###### Condensed Labels Model (5 classes)
| | Validation data (20% of train) | Stage 1 Test data | Stage 2 test data |
| ------------ | ------------ | ------------- | ------------- |
| Log Loss | 0.5538 | 0.6201 | 2.9057 |
| Accuracy | - | 76.1% | 20% |

###### Likelihood Labels Model (3 classes)
| | Validation data (20% of train) | Stage 1 Test data | Stage 2 test data |
| ------------ | ------------ | ------------- | ------------- |
| Log Loss | 0.6431 | 0.6999 | 1.0935 |
| Accuracy | - | 66.9% | 34.4% |

The scores are not especially impressive (among other reasons, the model is trained on only 80% of the original training data, and hyperparameters are not tuned), but the approach is versatile and intrepretable. The solution also ONLY uses the provided text data in the training set, so there are a lot of potential improvements to be made in using the provided Gene/Variation data, and external data. Please see [Usage and Approach](#usage-and-approach) for more details on the approach, visualizations, and suggestions for further improvements.

## Usage and Approach

### Include Data Files
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

### Train word2vec Model
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

**Output**: A saved gensim word2vec model `medline_SHUFFLED_biomedical_embeddings_200_lit_params_win2_60iters` in the [`wordEmbeddings`](wordEmbeddings/) directory.

### Train Models
From the root directory, run [`main.py`](wordEmbeddings/main.py) with `train` flag to train and save the weights of the models:
```python
python main.py train
```
What that does is:
1. 

### Visualize and Interpret Models
The visualizations of the models' sentence and word weights assignment to the Stage 1 and Stage 2 test data are available here:
https://wlouie1.github.io/MSKCCPersonalizedMedicineChallenge/visualization

Note that the page may take a couple minutes to fully load, so **be patient**. Once it does, it should look something like this:

On the navigation bar on the top, there is a drop down at the top right to allow you to select the `ID` of the test data instance. The corresponding Gene/Variation information is on the top left. There are also two modes you can toggle: `Filtered and Sorted` (default) and `Full Text View`. Shown in the main body of the page are the models results. In the `Filtered and Sorted` mode of the page, the sentences are ranked highest attention weight to lowest attention weight (what the model attributes as most important to least important), with the opacity of the red color corresponding to the weight values. In each sentence, words are highlighted in blue, and the opacity of the blue corresponds to their attention weight values (normalized by its sentence weight)--the darker the blue highlight, the more "significant" the word is to the model. The `Full Text View` shows the original given text, not just the text subset used to train the models. 

In the screenshot example above, `ID` 651 of the Stage 1 Test data is selected (DIS3/R780K), and the models correctly predict it to be Loss-of-function. The Raw Labels model and the Condensed Labels model both attribute phrases such as `mutations markedly reduced hdis3 exoribonucleolytic activity` and `catalytic mb mutant` as important in classifying the gene variation as Loss-of-function. Not all of them assign important sentences and words that make sense, but you can see that many of them seem to be in the general right direction if you click around and explore other result instances in the test data.

To build the visualizations locally, run [`main.py`](wordEmbeddings/main.py) with `visualize` flag:

```python
python main.py visualize
```
This will generate JSON files in the [`results`](visualization/results/) directory. Host the outer [`visualization`](visualization/) directory to view the results in a browser. For example, to host and view in a browser locally using python, run the following from the [`visualization`](visualization/) directory:
```python
python -m SimpleHTTPServer
```
And then open up http://localhost:8000/ in a browser. Again, the page may take a couple minutes to fully load.

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
