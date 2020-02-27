![textvec logo](/examples/images/logo.png?raw=true)
## WHAT: Supervised text vectorization tool

Textvec is a text vectorization tool, with the aim to implement all the "classic" text vectorization NLP methods in Python. The main idea of this project is to show alternatives for an excellent TFIDF method which is highly overused for supervised tasks. All interfaces are similar to [scikit-learn](https://github.com/scikit-learn/scikit-learn) so you should be able to test the performance of this supervised methods just with a few changes.

Textvec is compatible with: __Python 2.7-3.7__.

------------------

## WHY: Comparison with TFIDF
As you can read in the different articles<sup>1,2</sup> almost on every dataset supervised methods outperform unsupervised.
But most text classification examples on the internet ignores that fact.

|          |      IMDB_bin      |   RT_bin   |  Airlines Sentiment_bin  | Airlines Sentiment_multiclass | 20news_multiclass |
|----------|--------------------|------------|--------------------------|-------------------------------|-------------------|
| TF       |       0.8984       |   0.7571   |          0.9194          |            0.8084             |       0.8206      |
| TFIDF    |       0.9052       |   0.7717   |        __0.9259__        |            0.8118             |     __0.8575__    |
| TFPF     |       0.8813       |   0.7403   |          0.9212          |              NA               |         NA        |
| TFRF     |       0.8797       |   0.7412   |          0.9194          |              NA               |         NA        |
| TFICF    |       0.8984       |   0.7642   |          0.9199          |          __0.8125__           |       0.8292      |
| TFBINICF |       0.8984       |   0.7571   |          0.9194          |              NA               |         NA        |
| TFCHI2   |       0.8898       |   0.7398   |          0.9108          |              NA               |         NA        |
| TFGR     |       0.8850       |   0.7065   |          0.8956          |              NA               |         NA        |
| TFRRF    |       0.8879       |   0.7506   |          0.9194          |              NA               |         NA        |
| TFOR     |     __0.9092__     | __0.7806__ |          0.9207          |              NA               |         NA        |

Here is a comparison for binary classification on imdb sentiment data set. Labels sorted by accuracy score and the heatmap shows the correlation between different approaches. As you can see some methods are good for to ensemble models or perform features selection.

![Binary comparison](/examples/images/imdb_bin.png?raw=true)

For more dataset benchmarks (rotten tomatoes, airline sentiment) see [Binary classification quality comparison](/examples/binary_comparison.ipynb)

------------------

## Install:
Usage:
```
pip install textvec
```

Source code:
```
git clone https://github.com/textvec/textvec
cd textvec
pip install .
```

------------------

## HOW: Examples
The usage is similar to scikit-learn:
``` python
from sklearn.feature_extraction.text import CountVectorizer
from textvec.vectorizers import TfBinIcfVectorizer

cvec = CountVectorizer().fit(train_data.text)

tficf_vec = TfBinIcfVectorizer(sublinear_tf=True)
tficf_vec.fit(cvec.transform(text), y)
```
For more detailed examples see [Basic example](/examples/basic_usage.ipynb) and other notebooks in [Examples](/examples)

### Currently implemented methods:

- TfIcfVectorizer
- TforVectorizer
- TfgrVectorizer
- TfigVectorizer
- Tfchi2Vectorizer
- TfrfVectorizer
- TfrrfVectorizer
- TfBinIcfVectorizer
- TfpfVectorizer
- SifVectorizer
- TfbnsVectorizer

Most of the vectorization techniques you can find in articles<sup>1,2,3</sup>. If you see any method with wrong name or reference please commit!

------------------

## TODO
- [ ] Docs

------------------

## REFERENCE
- [1] [Deqing Wang and Hui Zhang] [Inverse-Category-Frequency based Supervised Term Weighting Schemes for Text Categorization](https://arxiv.org/pdf/1012.2609.pdf)
- [2] [M. Lan, C. L. Tan, J. Su, and Y. Lu] [Supervised and traditional term weighting methods for automatic text categorization](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.151.3665&rep=rep1&type=pdf)
- [3] [Sanjeev Arora, Yingyu Liang and Tengyu Ma] [A Simple But Tough-To-Beat Baseline For Sentence Embeddings](https://openreview.net/pdf?id=SyK00v5xx)
- [4] Thanks [aysent](https://aysent.github.io/2015/10/21/supervised-term-weighting.html#motivation-for-text-classification-tasks) for an inspiration
