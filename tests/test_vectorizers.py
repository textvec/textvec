import math

import pytest
from scipy.spatial.distance import cosine
from scipy.sparse.linalg import norm as sp_norm

from textvec.vectorizers import *


tokenized_sentences = [["dog", "cat"], ["foo", "bar"], ["one", "two", "foo"]]


vectorizers = [
    TfIcfVectorizer,
    TforVectorizer,
    TfgrVectorizer,
    TfigVectorizer,
    Tfchi2Vectorizer,
    TfrfVectorizer,
    TfrrfVectorizer,
    TfBinIcfVectorizer,
    TfpfVectorizer
]


def test_keyed_vectors(keyed_vectors):
    similarity = cosine(keyed_vectors["cat"], keyed_vectors["dog"])
    assert math.isclose(similarity, 0.04)


@pytest.mark.parametrize("vectorizer", vectorizers)
class TestVectorizers:
    def test_shape(self, count_matrix_dataset, vectorizer):
        _, x, y = count_matrix_dataset
        v = vectorizer()
        v.fit(x, y)
        matrix = v.transform(x)
        assert matrix.shape == (3, 6)
        assert x.shape == matrix.shape

    def test_norm(self, count_matrix_dataset, vectorizer):
        _, x, y = count_matrix_dataset
        v = vectorizer(norm="l2")
        v.fit(x, y)
        matrix = v.transform(x)
        norm = sp_norm(matrix, axis=1)
        np.testing.assert_allclose(np.mean(norm), 1.)

    def test_oov(self, count_matrix_dataset, vectorizer):
        cv, x, y = count_matrix_dataset
        v = vectorizer()
        v.fit(x, y)
        sentences = ["out of", "vocab words"]
        transformed = cv.transform(sentences)
        matrix = v.transform(transformed).toarray()
        np.testing.assert_allclose(matrix, np.zeros_like(matrix))


class TestSif:
    def test_shape(self, keyed_vectors):
        sif = SifVectorizer(keyed_vectors).fit(tokenized_sentences)
        matrix = sif.transform(tokenized_sentences)
        assert matrix.shape == (3, 5)

    def test_not_fitted(self, keyed_vectors):
        sif = SifVectorizer(keyed_vectors)
        with pytest.raises(RuntimeError):
            sif.transform([["foo", "bar"]])

    def test_norm(self, keyed_vectors):
        sif = SifVectorizer(keyed_vectors, norm="l2").fit(tokenized_sentences)
        matrix = sif.transform(tokenized_sentences)
        norm = np.linalg.norm(matrix, axis=1)
        np.testing.assert_allclose(np.mean(norm), 1.)

    def test_vocab(self, keyed_vectors):
        sif = SifVectorizer(keyed_vectors).fit(tokenized_sentences)
        assert len(sif.vocab) == 6

    def test_oov(self, keyed_vectors):
        sif = SifVectorizer(keyed_vectors, npc=0).fit(tokenized_sentences)
        matrix = sif.transform([["out", "of"], ["vocab", "words"]])
        expected = np.zeros_like(matrix)
        np.testing.assert_array_equal(matrix, expected)
