import pytest
import numpy as np
from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import CountVectorizer


@pytest.fixture(scope="module")
def keyed_vectors():
    model = KeyedVectors(5)
    words = ["cat", "dog", "foo", "bar", "one", "two"]
    vectors = np.array([[0, 0, 0, 0, 1],
                        [0, 0, 0, 0.28, 0.96],
                        [1, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0],
                        [0.28, 0.96, 0, 0, 0],
                        [0.6, 0.8, 0, 0, 0]])
    model.add(words, vectors)
    return model


@pytest.fixture(scope="module")
def count_matrix_dataset():
    cv = CountVectorizer()
    sentences = ["dog cat", "foo bar", "one two foo"]
    targets = np.array([1, 0, 0])
    matrix = cv.fit_transform(sentences)
    return cv, matrix, targets
