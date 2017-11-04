import gensim
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from ggplot import *

TSNE_MAX_VOCAB_SIZE = 1000
TSNE_PLOT_MAX_VOCAB_SIZE = 250

def main():
	wvmodel = gensim.models.Word2Vec.load('medline_SHUFFLED_biomedical_embeddings_200_lit_params_win2_60iters')
	vocab_size = sum(1 for word in wvmodel.wv.vocab)
	print "Vocab size:", vocab_size

	sorted_vocab = sorted([(vocab_obj.count, word) for word, vocab_obj in wvmodel.wv.vocab.items()], reverse=True)
	words_of_interest = [word for count, word in sorted_vocab[0:TSNE_MAX_VOCAB_SIZE]]

	word_vectors = wvmodel.wv[words_of_interest]

	print "TSNEing..."
	tsne = TSNE(n_components=2, perplexity=40, n_iter=1000, random_state=0, init='pca', verbose=1)
	tsne_out = tsne.fit_transform(word_vectors)

	print tsne_out.shape

	df = pd.concat([pd.DataFrame(tsne_out), pd.Series(words_of_interest)], axis=1)
	df.columns = ['x', 'y', 'word']

	print ggplot(df.iloc[0:TSNE_PLOT_MAX_VOCAB_SIZE], aes(x='x', y='y', label='word')) + geom_text(size=10, alpha=0.6)


if __name__ == "__main__":
	main()