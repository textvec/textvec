import collections
import itertools

import numpy as np
import scipy.sparse as sp
from scipy.stats import norm
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.base import TransformerMixin, BaseEstimator
from gensim.models.keyedvectors import BaseKeyedVectors


def ensure_sparse_format(array, dtype=np.float64):
    if sp.issparse(array):
        if array.dtype != dtype:
            array = array.astype(dtype)
    else:
        array = sp.csr_matrix(array, dtype=dtype)
    return array


class TfIcfVectorizer(TransformerMixin, BaseEstimator):
    """Supervised method (supports multiclass) to transform 
    a count matrix to a normalized Tficf representation
    Tf means term-frequency while ICF means inverse category frequency.
    Parameters
    ----------
    norm : 'l1', 'l2', 'max' or None, optional
        Norm used to normalize term vectors. None for no normalization.
    sublinear_tf : boolean, default=False
        Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf).
    References
    ----------
    .. [0] `https://arxiv.org/pdf/1012.2609.pdf`
    """

    def __init__(self, norm=None, sublinear_tf=False):
        self.norm = norm
        self.sublinear_tf = sublinear_tf

    def fit(self, X, y):
        n_samples, n_features = X.shape
        samples = []
        self.number_of_classes = len(np.unique(y))
        for val in range(self.number_of_classes):
            class_mask = sp.spdiags(y == val, 0, n_samples, n_samples)
            samples.append(np.bincount(
                (class_mask * X).indices, minlength=n_features))
        samples = np.array(samples)
        self.corpus_occurence = np.sum(samples != 0, axis=0)
        self.k = np.log2(1 + (self.number_of_classes / self.corpus_occurence))
        self._n_features = n_features
        return self

    def transform(self, X, min_freq=1):
        if self.sublinear_tf:
            X = ensure_sparse_format(X)
            np.log(X.data, X.data)
            X.data += 1
        f = self._n_features
        X = X * sp.spdiags(self.k, 0, f, f)
        if self.norm:
            X = normalize(X, self.norm, copy=False)
        return X


class BaseBinaryFitter(TransformerMixin):
    """Base class for supervised methods (supports only binary classification).
    Should not be used as by itself.
    ----------
    norm : 'l1', 'l2', 'max' or None, optional
        Norm used to normalize term vectors. None for no normalization.
    smooth_df : boolean or int, default=True
        Smooth df weights by adding one to document frequencies, as if an
        extra document was seen containing every term in the collection
        exactly once. Prevents zero divisions.
    sublinear_tf : boolean, default=False
        Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf).
    References
    ----------
    """

    def __init__(self, norm='l2', smooth_df=True, sublinear_tf=False):
        self.norm = norm
        self.smooth_df = smooth_df
        self.sublinear_tf = sublinear_tf

    def fit(self, X, y):

        n_samples, n_features = X.shape

        pos_samples = sp.spdiags(y, 0, n_samples, n_samples)
        neg_samples = sp.spdiags(1 - y, 0, n_samples, n_samples)

        X_pos = pos_samples * X
        X_neg = neg_samples * X

        tp = np.bincount(X_pos.indices, minlength=n_features)
        fp = np.sum(y) - tp
        tn = np.bincount(X_neg.indices, minlength=n_features)
        fn = np.sum(1 - y) - tn

        self._n_samples = n_samples
        self._n_features = n_features

        self._tp = tp
        self._fp = fp
        self._fn = fn
        self._tn = tn
        self._p = np.sum(y)
        self._n = np.sum(1 - y)

        if self.smooth_df:
            self._n_samples += int(self.smooth_df)
            self._tp += int(self.smooth_df)
            self._fp += int(self.smooth_df)
            self._fn += int(self.smooth_df)
            self._tn += int(self.smooth_df)
        return self


class TfbnsVectorizer(BaseBinaryFitter, BaseEstimator):
    """Supervised method (supports ONLY binary classification)
    transform a count matrix to a normalized TfBNS representation
    Tf means term-frequency while OR means odds ratio.
    Parameters
    ----------
    norm : 'l1', 'l2', 'max' or None, optional
        Norm used to normalize term vectors. None for no normalization.
    smooth_df : boolean or int, default=True
        Smooth df weights by adding one to document frequencies, as if an
        extra document was seen containing every term in the collection
        exactly once. Prevents zero divisions.
    sublinear_tf : boolean, default=False
        Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf).
    References
    ----------
    .. [George Forman] https://www.researchgate.net/publ \
    ication/221613942_BNS_feature_scaling_An_improved_representation_over_TF-ID \
    F_for_SVM_text_classification

    """
    def transform(self, X):
        tp = self._tp
        fp = self._fp
        fn = self._fn
        tn = self._tn

        f = self._n_features
        tpr = tp / self._p
        fpr = fp / self._n
        min_bound, max_bound = 0.0005, 1 - 0.0005
        tpr[tpr < min_bound]  = min_bound
        tpr[tpr > max_bound]  = max_bound
        fpr[fpr < min_bound]  = min_bound
        fpr[fpr > max_bound]  = max_bound
        k = np.abs(norm.ppf(tpr) - norm.ppf(fpr))
        X = X * sp.spdiags(k, 0, f, f)
        if self.norm:
            X = normalize(X, self.norm, copy=False)
        return X


class TfebnsVectorizer(TransformerMixin, BaseEstimator):
    """Doesn't work for multiclass as promised in paper. Needs investigation
    Parameters
    ----------
    norm : 'l1', 'l2', 'max' or None, optional
        Norm used to normalize term vectors. None for no normalization.
    sublinear_tf : boolean, default=False
        Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf).
    References
    ----------
    .. [0] `https://www.researchgate.net/publication/333231361_\
    Weighting_Words_Using_Bi-Normal_Separation_for_Text_Classifi \
    cation_Tasks_with_Multiple_Classes`
    """
    def __init__(self, norm=None, sublinear_tf=False):
        self.norm = norm
        self.sublinear_tf = sublinear_tf

    def fit(self, X, y):
        n_samples, n_features = X.shape
        samples = []
        for i, val in enumerate(y.unique()):
            class_y = y == val
            class_p = np.sum(class_y)
            class_n = np.sum(1 - class_y)
            pos_samples = sp.spdiags(class_y, 0, n_samples, n_samples)
            neg_samples = sp.spdiags(1 - class_y, 0, n_samples, n_samples)

            class_X_pos = pos_samples * X
            class_X_neg = neg_samples * X
            tp = np.bincount(class_X_pos.indices, minlength=n_features)
            fp = class_p - tp
            tn = np.bincount(class_X_neg.indices, minlength=n_features)
            fn = class_n - tn
            tpr = tp / class_p
            fpr = fp / class_n
            min_bound, max_bound = 0.0005, 1 - 0.0005
            tpr[tpr < min_bound]  = min_bound
            tpr[tpr > max_bound]  = max_bound
            fpr[fpr < min_bound]  = min_bound
            fpr[fpr > max_bound]  = max_bound
            samples.append(norm.ppf(tpr) - norm.ppf(fpr))
        samples = np.array(samples)
        self.k = np.max(samples, axis=0)
        self._n_features = n_features
        return self

    def transform(self, X):
        if self.sublinear_tf:
            X = ensure_sparse_format(X)
            np.log(X.data, X.data)
            X.data += 1
        f = self._n_features
        X = X * sp.spdiags(self.k, 0, f, f)
        if self.norm:
            X = normalize(X, self.norm, copy=False)
        return X


class TforVectorizer(BaseBinaryFitter, BaseEstimator):
    """Supervised method (supports ONLY binary classification) 
    transform a count matrix to a normalized Tfor representation
    Tf means term-frequency while OR means odds ratio.
    Parameters
    ----------
    norm : 'l1', 'l2', 'max' or None, optional
        Norm used to normalize term vectors. None for no normalization.
    smooth_df : boolean or int, default=True
        Smooth df weights by adding one to document frequencies, as if an
        extra document was seen containing every term in the collection
        exactly once. Prevents zero divisions.
    sublinear_tf : boolean, default=False
        Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf).
    References
    ----------
    .. [M. Lan, C. L. Tan, J. Su, and Y. Lu] `Supervised and traditional 
                term weighting methods for automatic text categorization`
    """

    def transform(self, X, confidence=False):
        """
        Parameters
        ----------
        confidence : boolean, default=False
            Return bool vector states that feature if in 95% confidence interval.
        """
        if self.sublinear_tf:
            X = ensure_sparse_format(X)
            np.log(X.data, X.data)
            X.data += 1

        tp = self._tp
        fp = self._fp
        fn = self._fn
        tn = self._tn

        f = self._n_features

        k = np.log((tp * tn) / (fp * fn))
        X = X * sp.spdiags(k, 0, f, f)

        if self.norm:
            X = normalize(X, self.norm, copy=False)

        if confidence:
            up = np.exp(k + 1.96 * np.sqrt(1 / tp + 1 / fp + 1 / fn + 1 / tn))
            low = np.exp(k - 1.96 * np.sqrt(1 / tp + 1 / fp + 1 / fn + 1 / tn))
            return X, (up < 1.0) | (low > 1.0)
        return X


class TfgrVectorizer(BaseBinaryFitter, BaseEstimator):
    """Supervised method (supports ONLY binary classification) 
    transform a count matrix to a normalized Tfor representation
    Tf means term-frequency while GR means gain ratio.
    Parameters
    ----------
    norm : 'l1', 'l2', 'max' or None, optional
        Norm used to normalize term vectors. None for no normalization.
    smooth_df : boolean or int, default=True
        Smooth df weights by adding one to document frequencies, as if an
        extra document was seen containing every term in the collection
        exactly once. Prevents zero divisions.
    sublinear_tf : boolean, default=False
        Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf).
    References
    ----------
    .. [0] `https://en.wikipedia.org/wiki/Information_gain_ratio`
    """

    def transform(self, X):
        if self.sublinear_tf:
            X = ensure_sparse_format(X)
            np.log(X.data, X.data)
            X.data += 1

        tp = self._tp
        fp = self._fp
        fn = self._fn
        tn = self._tn

        n = self._n_samples

        f = self._n_features

        k = -((tp + fp) / n) * np.log((tp + fp) / n)
        k -= ((fn + tn) / n) * np.log((fn + tn) / n)
        k += (tp / n) * np.log(tp / (tp + fn))
        k += (fn / n) * np.log(fn / (tp + fn))
        k += (fp / n) * np.log(fp / (fp + tn))
        k += (tn / n) * np.log(tn / (fp + tn))

        d = -((tp + fp) / n) * np.log((tp + fp) / n)
        d -= ((fn + tn) / n) * np.log((fn + tn) / n)

        k *= d

        X = X * sp.spdiags(k, 0, f, f)
        if self.norm:
            X = normalize(X, self.norm, copy=False)

        return X


class TfigVectorizer(BaseBinaryFitter, BaseEstimator):
    """Supervised method (supports ONLY binary classification) 
    transform a count matrix to a normalized Tfor representation
    Tf means term-frequency while IG means information gain.
    Parameters
    ----------
    norm : 'l1', 'l2', 'max' or None, optional
        Norm used to normalize term vectors. None for no normalization.
    smooth_df : boolean or int, default=True
        Smooth df weights by adding one to document frequencies, as if an
        extra document was seen containing every term in the collection
        exactly once. Prevents zero divisions.
    sublinear_tf : boolean, default=False
        Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf).
    References
    ----------
    .. [M. Lan, C. L. Tan, J. Su, and Y. Lu] `Supervised and traditional 
                term weighting methods for automatic text categorization`
    """

    def transform(self, X):
        if self.sublinear_tf:
            X = ensure_sparse_format(X)
            np.log(X.data, X.data)
            X.data += 1

        tp = self._tp
        fp = self._fp
        fn = self._fn
        tn = self._tn

        n = self._n_samples

        f = self._n_features

        k = -((tp + fp) / n) * np.log((tp + fp) / n)
        k -= ((fn + tn) / n) * np.log((fn + tn) / n)
        k += (tp / n) * np.log(tp / (tp + fn))
        k += (fn / n) * np.log(fn / (tp + fn))
        k += (fp / n) * np.log(fp / (fp + tn))
        k += (tn / n) * np.log(tn / (fp + tn))

        X = X * sp.spdiags(k, 0, f, f)
        if self.norm:
            X = normalize(X, self.norm, copy=False)

        return X


class Tfchi2Vectorizer(BaseBinaryFitter, BaseEstimator):
    """Supervised method (supports ONLY binary classification) 
    transform a count matrix to a normalized Tfor representation
    Tf means term-frequency while CHI2 means Chi-Square.
    Parameters
    ----------
    norm : 'l1', 'l2', 'max' or None, optional
        Norm used to normalize term vectors. None for no normalization.
    smooth_df : boolean or int, default=True
        Smooth df weights by adding one to document frequencies, as if an
        extra document was seen containing every term in the collection
        exactly once. Prevents zero divisions.
    sublinear_tf : boolean, default=False
        Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf).
    References
    ----------
    .. [M. Lan, C. L. Tan, J. Su, and Y. Lu] `Supervised and traditional 
                term weighting methods for automatic text categorization`
    """

    def transform(self, X):
        if self.sublinear_tf:
            X = ensure_sparse_format(X)
            np.log(X.data, X.data)
            X.data += 1

        tp = self._tp
        fp = self._fp
        fn = self._fn
        tn = self._tn

        n = self._n_samples

        f = self._n_features

        k = n * (tp * tn - fp * fn)**2
        v = (tp + fp) * (fn + tn) * (tp + fn) * (fp + tn)
        k *= v

        X = X * sp.spdiags(k, 0, f, f)
        if self.norm:
            X = normalize(X, self.norm, copy=False)

        return X


class TfrfVectorizer(BaseBinaryFitter, BaseEstimator):
    """Supervised method (supports ONLY binary classification) 
    transform a count matrix to a normalized Tfrf representation
    Tf means term-frequency while RF means relevance frequency.
    Parameters
    ----------
    norm : 'l1', 'l2', 'max' or None, optional
        Norm used to normalize term vectors. None for no normalization.
    smooth_df : boolean or int, default=True
        Smooth df weights by adding one to document frequencies, as if an
        extra document was seen containing every term in the collection
        exactly once. Prevents zero divisions.
    sublinear_tf : boolean, default=False
        Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf).
    References
    ----------
    .. [M. Lan, C. L. Tan, J. Su, and Y. Lu] `Supervised and traditional 
                term weighting methods for automatic text categorization`
    """

    def transform(self, X):
        if self.sublinear_tf:
            X = ensure_sparse_format(X)
            np.log(X.data, X.data)
            X.data += 1

        tp = self._tp
        fn = self._fn

        f = self._n_features

        k = np.log2(2 + tp / fn)

        X = X * sp.spdiags(k, 0, f, f)
        if self.norm:
            X = normalize(X, self.norm, copy=False)

        return X


class TfrrfVectorizer(BaseBinaryFitter, BaseEstimator):
    """Supervised method (supports ONLY binary classification) 
    transform a count matrix to a normalized Tfirf representation
    Tf means term-frequency while RRF means reversed relevance frequency.
    Parameters
    ----------
    norm : 'l1', 'l2', 'max' or None, optional
        Norm used to normalize term vectors. None for no normalization.
    smooth_df : boolean or int, default=True
        Smooth df weights by adding one to document frequencies, as if an
        extra document was seen containing every term in the collection
        exactly once. Prevents zero divisions.
    sublinear_tf : boolean, default=False
        Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf).
    References
    ----------
    """

    def transform(self, X):
        if self.sublinear_tf:
            X = ensure_sparse_format(X)
            np.log(X.data, X.data)
            X.data += 1

        tp = self._tp
        tn = self._tn

        f = self._n_features

        k = np.log2(2 + tp / tn)

        X = X * sp.spdiags(k, 0, f, f)
        if self.norm:
            X = normalize(X, self.norm, copy=False)

        return X


class TfBinIcfVectorizer(BaseBinaryFitter, BaseEstimator):
    """Supervised method (supports ONLY binary classification) 
    transform a count matrix to a normalized Tficf representation
    Tf means term-frequency while ICF means inverse category frequency.
    Parameters
    ----------
    norm : 'l1', 'l2', 'max' or None, optional
        Norm used to normalize term vectors. None for no normalization.
    sublinear_tf : boolean, default=False
        Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf).
    References
    ----------
    .. [0] `https://arxiv.org/pdf/1012.2609.pdf`
    """

    def transform(self, X, min_freq=1):
        if self.sublinear_tf:
            X = ensure_sparse_format(X)
            np.log(X.data, X.data)
            X.data += 1
        tp = self._tp
        fn = self._fn

        f = self._n_features

        k = np.log2(
            1 + (2 / (2 - ((tp >= min_freq) | (fn >= min_freq)).astype(int))))

        X = X * sp.spdiags(k, 0, f, f)
        if self.norm:
            X = normalize(X, self.norm, copy=False)

        return X


class TfpfVectorizer(BaseBinaryFitter, BaseEstimator):
    """Supervised method (supports ONLY binary classification) 
    transform a count matrix to a normalized Tficf representation
    Tf means term-frequency while PF power frequency.
    Parameters
    ----------
    norm : 'l1', 'l2', 'max' or None, optional
        Norm used to normalize term vectors. None for no normalization.
    sublinear_tf : boolean, default=False
        Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf).
    References
    ----------
    """

    def transform(self, X, min_freq=1):
        if self.sublinear_tf:
            X = ensure_sparse_format(X)
            np.log(X.data, X.data)
            X.data += 1
        tp = self._tp
        fn = self._fn

        f = self._n_features

        k = np.log(
            2 + tp / fn ** (2 / (2 - ((tp >= min_freq) | (fn >= min_freq)).astype(int))))

        X = X * sp.spdiags(k, 0, f, f)
        if self.norm:
            X = normalize(X, self.norm, copy=False)

        return X


# TODO: add frequency source
class SifVectorizer:
    """Unsupervised method
    compute smooth inverse frequency (SIF)

    Parameters
    ----------
    model : :class:`~gensim.models.keyedvectors.BaseKeyedVectors`
        Object contains the word vectors and the vocabulary.

    alpha : float, default=0.001
        Parameter which is used to weigh each individual word
        based on its probability.

    npc : int, default=1
        Number of principal components to remove from
        sentence embedding.

    norm : 'l1', 'l2', 'max' or None, optional
        Norm used to normalize term vectors. None for no normalization.

    Examples
    --------
    >>> from textvec.vectorizers import SifVectorizer
    >>> import gensim.downloader as api
    >>> model = api.load("glove-twitter-25")
    >>> X = [["first", "sentence"], ["second", "sentence"]]
    >>> sif = SifVectorizer(model, alpha=0.001, npc=1)
    >>> sif.fit(X)
    >>> sif.transform(X)
    array([[-0.2028063 , -0.07884892,  0.30937403, -0.4058012 , -0.02779805,
            -0.14715618,  0.09867747,  0.2490029 ,  0.22715728,  0.02029565,
            -0.0324943 , -0.14876653, -0.19695622, -0.349479  , -0.00145111,
            -0.17245306,  0.14833301, -0.15239874,  0.1624661 ,  0.08161873,
             0.13065818, -0.06360044, -0.39932743,  0.02312368,  0.26987103],
           [ 0.20280789,  0.0788492 , -0.30937368,  0.40580118,  0.02779823,
             0.14715572, -0.09867608, -0.24900348, -0.22715725, -0.02029571,
             0.0324944 ,  0.14876756,  0.19695152,  0.34947947,  0.00145159,
             0.1724532 , -0.14833233,  0.15239872, -0.1624666 , -0.08161937,
            -0.13065891,  0.06360071,  0.39932764, -0.02312382, -0.26987168]],
          dtype=float32)

    References
    ----------
    Arora S, Liang Y, Ma T (2017)
    A Simple but Tough-to-Beat Baseline for Sentence Embeddings
    https://openreview.net/pdf?id=SyK00v5xx

    """

    def __init__(self, model, alpha=1e-3, npc=1, norm="l2"):
        if isinstance(model, BaseKeyedVectors):
            self.model = model
        else:
            raise RuntimeError("Model must be child of BaseKeyedVectors class")

        assert alpha > 0
        assert npc >= 0

        self.model = model
        self.dim = model.vector_size
        self.alpha = alpha
        self.npc = npc
        self.norm = norm

        self.vocab = None
        self.oov_sif_weight = None

    def _compute_pc(self, X, npc):
        """Compute the first n principal components
        for given sentence embeddings.

        Parameters
        ----------
        X : numpy.ndarray
            The sentence embedding.

        npc : int
            The number of principal components to compute.

        Returns
        -------
        numpy.ndarray
            The first `npc` principal components of sentence embedding.

        """
        svd = TruncatedSVD(n_components=npc, n_iter=7, random_state=0)
        svd.fit(X)
        return svd.components_

    def _remove_pc(self, X, npc):
        """Remove the projection from the averaged sentence embedding.

        Parameters
        ----------
        X : numpy.ndarray
            The sentence embedding.

        npc : int
            The number of principal components to compute.

        Returns
        -------
        numpy.ndarray
            The sentence embedding after removing the projection.

        """
        pc = self._compute_pc(X, npc)
        if npc == 1:
            return X - X.dot(pc.transpose()) * pc
        else:
            return X - X.dot(pc.transpose()).dot(pc)

    def fit(self, X, y=None):
        """Learn a sif weights dictionary of all tokens
        in the tokenized sentences.

        Parameters
        ----------
        X : iterable
            An iterable which yields iterable of str.

        y : Ignored

        Returns
        -------
        self

        """
        vocab = collections.Counter(itertools.chain.from_iterable(X))
        corpus_size = len(vocab)
        self.vocab = {
            self.model.vocab[k].index: self.alpha / (self.alpha + v / corpus_size)
            for k, v in vocab.items()
            if k in self.model
        }
        self.oov_sif_weight = min(vocab.values())
        return self

    def _get_weighted_average(self, sentences):
        """Calculate average SIF embedding for each sentence.

        Parameters
        ----------
        sentences : iterable
            An iterable which yields iterable of str.

        Returns
        -------
        numpy.ndarray
            The sentence embedding matrix.

        """
        wv_vocab = self.model.vocab
        wv_vectors = self.model.vectors

        sent_embeddings = np.zeros((len(sentences), self.dim), dtype=np.float32)
        for i, sent in enumerate(sentences):
            sent_indices = [wv_vocab[w].index for w in sent if w in self.model]
            weights = np.array([
                self.vocab.get(idx, self.oov_sif_weight)
                for idx in sent_indices
            ])[:, None]
            if sent_indices:
                sent_emb = np.mean(wv_vectors[sent_indices] * weights, axis=0)
                sent_embeddings[i] = sent_emb
        return sent_embeddings

    def transform(self, X):
        """Transform sentences to SIF weighted sentence embedding matrix.

        Parameters
        ----------
        X : iterable
            An iterable which yields iterable of str.

        Returns
        -------
        numpy.ndarray
            The sentence embedding matrix.

        """
        if not self.vocab:
            raise RuntimeError("This SifVectorizer instance is not fitted yet.")
        embeddings = self._get_weighted_average(X)
        if self.npc > 0:
            embeddings = self._remove_pc(embeddings, self.npc)
        if self.norm:
            embeddings = normalize(embeddings, self.norm, copy=False)
        return embeddings
