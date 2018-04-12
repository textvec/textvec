# Textvec
## WHAT: Supervised text vectorization tool

Textvec is a text vectorization tool, with the aim to implement all the "classic" text vectorization NLP methods in Python. The main idea of this project is to show alternatives for an excellent TFIDF method which is highly overused for supervised tasks. An interfaces are similar to [scikit-learn](https://github.com/scikit-learn/scikit-learn) so you should be able to test the perfomance of this supervised methods just with a few changes.

Textvec is compatible with: __Python 2.7-3.6__.

------------------

## WHY: Comparison with TFIDF
As you can read in the different articles<sup>1,2</sup> almost on every dataset supervised methods outperfom unsupervised.
But most text classification examples on the internet ignores that fact.

|          |      IMDB_bin      |   RT_bin   |  Airlines Sentiment_bin  | Airlines Sentiment_multiclass | 20news_multiclass |
|----------|--------------------|------------|--------------------------|-------------------------------|-------------------|
| TFOR     |     __0.9088__     | __0.7820__ |          0.9173          |              NA               |         NA        |
| TFICF    |       0.8992       |   0.7661   |          0.9220          |          __0.8067__           |     __0.8552__    |
| TFBINICF |       0.8978       |   0.7628   |        __0.9238__        |              NA               |         NA        |
| TFRF     |       0.8977       |   0.7609   |          0.9207          |              NA               |         NA        |
| TFIDF    |       0.8923       |   0.7539   |          0.8939          |            0.7763             |       0.8335      |
| TFPF     |       0.8949       |   0.7464   |          0.9164          |              NA               |         NA        |
| TF       |       0.8786       |   0.7286   |          0.9017          |            0.7865             |       0.7796      |
| TFIR     |       0.8361       |   0.7159   |          0.9017          |              NA               |         NA        |
| TFCHI2   |       0.8734       |   0.6990   |          0.8900          |              NA               |         NA        |
| TFGR     |       0.8581       |   0.6793   |          0.8883          |              NA               |         NA        |

Here is a comparison for binary classification on imdb sentiment data set. Labels sorted by accuracy score and the heatmap shows the correlation between different aproaches. As you can see some methods are good for to ensemble the models or perform features selection.

![Binary comparison](https://github.com/zveryansky/textvec/blob/master/examples/images/imdb_bin.png)

For more dataset benchmarks (rottent tomatos, airline sentiment) see [Binary classification quality comparison](https://github.com/zveryansky/textvec/blob/master/examples/binary_classification_quality_comparison.ipynb)

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
For more detailed examples see [Basic example](https://github.com/zveryansky/textvec/blob/master/examples/basic_usage.ipynb) and other notebooks in [Examples](https://github.com/zveryansky/textvec/blob/master/examples)

### Currently impletented methods:

- TfIcfVectorizer
- TforVectorizer
- TfgrVectorizer
- TfigVectorizer
- Tfchi2Vectorizer
- TfrfVectorizer
- TfrrfVectorizer
- TfBinIcfVectorizer
- TfpfVectorizer

Most of the vectorization techniques you can find in articles<sup>1,2</sup>. If you see any method with wrong name or reference pls commit!

------------------

## TODO
- [ ] Add methods from https://arxiv.org/pdf/1305.0638.pdf
- [ ] Remove dependence of sklearn
- [ ] Tests

------------------

## REFERENCE
- [1] https://arxiv.org/pdf/1012.2609.pdf
- [2] [M. Lan, C. L. Tan, J. Su, and Y. Lu] Supervised and traditional term weighting methods for automatic text categorization