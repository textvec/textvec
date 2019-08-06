from collections import defaultdict

import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from gensim.models.keyedvectors import BaseKeyedVectors


class TfIcfVectorizer():
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

    def transform(self, X, min_freq=1):
        if self.sublinear_tf:
            X[X.nonzero()] = 1.0 + np.log(X[X.nonzero()])
        f = self._n_features
        X = X * sp.spdiags(self.k, 0, f, f)
        if self.norm:
            X = normalize(X, self.norm)
        return X


class BaseBinaryFitter():
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
        fn = np.bincount(X_neg.indices, minlength=n_features)
        tn = np.sum(1 - y) - fn

        self._n_samples = n_samples
        self._n_features = n_features

        self._tp = tp
        self._fp = fp
        self._fn = fn
        self._tn = tn

        if self.smooth_df:
            self._n_samples += int(self.smooth_df)
            self._tp += int(self.smooth_df)
            self._fp += int(self.smooth_df)
            self._fn += int(self.smooth_df)
            self._tn += int(self.smooth_df)


class TforVectorizer(BaseBinaryFitter):
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
            X[X.nonzero()] = 1.0 + np.log(X[X.nonzero()])

        tp = self._tp
        fp = self._fp
        fn = self._fn
        tn = self._tn

        f = self._n_features

        k = np.log((tp * tn) / (fp * fn))
        X = X * sp.spdiags(k, 0, f, f)

        if self.norm:
            X = normalize(X, self.norm)

        if confidence:
            up = np.exp(k + 1.96 * np.sqrt(1 / tp + 1 / fp + 1 / fn + 1 / tn))
            low = np.exp(k - 1.96 * np.sqrt(1 / tp + 1 / fp + 1 / fn + 1 / tn))
            return X, (up < 1.0) | (low > 1.0)
        return X


class TfgrVectorizer(BaseBinaryFitter):
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
            X[X.nonzero()] = 1.0 + np.log(X[X.nonzero()])

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
            X = normalize(X, self.norm)

        return X


class TfigVectorizer(BaseBinaryFitter):
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
            X[X.nonzero()] = 1.0 + np.log(X[X.nonzero()])

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
            X = normalize(X, self.norm)

        return X


class Tfchi2Vectorizer(BaseBinaryFitter):
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
            X[X.nonzero()] = 1.0 + np.log(X[X.nonzero()])

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
            X = normalize(X, self.norm)

        return X


class TfrfVectorizer(BaseBinaryFitter):
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
            X[X.nonzero()] = 1.0 + np.log(X[X.nonzero()])

        tp = self._tp
        fn = self._fn

        f = self._n_features

        k = np.log2(2 + tp / fn)

        X = X * sp.spdiags(k, 0, f, f)
        if self.norm:
            X = normalize(X, self.norm)

        return X


class TfrrfVectorizer(BaseBinaryFitter):
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
            X[X.nonzero()] = 1.0 + np.log(X[X.nonzero()])

        tp = self._tp
        tn = self._tn

        f = self._n_features

        k = np.log2(2 + tp / tn)

        X = X * sp.spdiags(k, 0, f, f)
        if self.norm:
            X = normalize(X, self.norm)

        return X


class TfBinIcfVectorizer(BaseBinaryFitter):
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
    .. [0] Arora S, Liang Y, Ma T (2017)
        A Simple but Tough-to-Beat Baseline for Sentence Embeddings.
        `https://arxiv.org/pdf/1012.2609.pdf`
    """

    def transform(self, X, min_freq=1):
        if self.sublinear_tf:
            X[X.nonzero()] = 1.0 + np.log(X[X.nonzero()])
        tp = self._tp
        fn = self._fn

        f = self._n_features

        k = np.log2(
            1 + (2 / (2 - ((tp >= min_freq) | (fn >= min_freq)).astype(int))))

        X = X * sp.spdiags(k, 0, f, f)
        if self.norm:
            X = normalize(X, self.norm)

        return X


class TfpfVectorizer(BaseBinaryFitter):
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
            X[X.nonzero()] = 1.0 + np.log(X[X.nonzero()])
        tp = self._tp
        fn = self._fn

        f = self._n_features

        k = np.log(
            2 + tp / fn ** (2 / (2 - ((tp >= min_freq) | (fn >= min_freq)).astype(int))))

        X = X * sp.spdiags(k, 0, f, f)
        if self.norm:
            X = normalize(X, self.norm)

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
        based on its probability p(w).
        If alpha = 1, train simply computes the averaged
        sentence representation.

    npc : int, default=1
        Number of principal components to remove from
        sentence embedding.

    norm : 'l1', 'l2', 'max' or None, optional
        Norm used to normalize term vectors. None for no normalization.

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
        self._word2freq = None

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
        vocab = defaultdict(int)
        for doc in X:
            for word in doc:
                vocab[word] += 1
        vocab = defaultdict(
            int, {k: v / len(vocab) for k, v in vocab.items()}
        )
        self._word2freq = vocab
        return self

    # TODO: optimize
    def _get_weighted_average(self, docs):
        sentences_vec = []
        for doc in docs:
            sif = []
            word_vectors = []
            for word in doc:
                if word in self.model:
                    sif.append(
                        self.alpha / (self.alpha + self._word2freq[word])
                    )
                    word_vectors.append(self.model[word])
            weighted_average = np.dot(sif, word_vectors) / len(sif) \
                if sif else np.zeros(self.dim)
            sentences_vec.append(weighted_average)
        return np.array(sentences_vec)

    def transform(self, X):
        embeddings = self._get_weighted_average(X)
        if self.npc > 0:
            embeddings = self._remove_pc(embeddings, self.npc)
        if self.norm:
            embeddings = normalize(embeddings, self.norm)
        return embeddings
