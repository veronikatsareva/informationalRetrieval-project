import pandas as pd
import numpy as np
import pickle
import json
import spacy
import string
import tqdm
from scipy import sparse
from gensim.models import Word2Vec
from gensim.models import FastText
from bm25_vectorizer import BM25Vectorizer


def corporaPreprocess():
    """
    This function preprocess the corpus of texts that are stored in .csv file.
    First, I delete the rows with empty desctiptions and uninformative columns
    for this task. The new dataset is saved in data/data.csv.
    Next, each text is tokenized and lemmatized with spacy.
    The preprocessed texts with their metadata are stored in metadata.json.
    :returns: 0 in case of successful execution
    """
    df = pd.read_csv("data/Netflix TV Shows and Movies.csv")
    df = df.dropna(subset=["description"]).reset_index(drop=True)
    corpora = df[["title", "description", "release_year", "imdb_score"]].copy()
    corpora.to_csv("data/data.csv")

    data = {}

    processer = spacy.load("en_core_web_sm")

    for index, row in tqdm.tqdm(corpora.iterrows()):
        processedText = processer(row["description"])
        tokens = [
            token.text.lower()
            for token in processedText
            if token.text not in string.punctuation
        ]
        lemmas = [
            token.lemma_.lower()
            for token in processedText
            if token.lemma_ not in string.punctuation
        ]
        data[index] = {
            "title": row["title"],
            "description": row["description"],
            "release_year": row["release_year"],
            "imdb_score": row["imdb_score"],
            "tokens": tokens,
            "lemmas": lemmas,
        }

    with open("data/metadata.json", "w") as file:
        json.dump(data, file)

    return 0


def bm25Vectorization():
    """
    This function vectorizes the corpus with BM25.
    The vectorizer and term-document matrix are saved in /models
    as binary files.
    :returns: 0 in case of successful execution
    """
    with open("data/metadata.json", "r") as file:
        data = json.load(file)

    texts = [" ".join(data[idx]["lemmas"]) for idx in data]

    vectorizer = BM25Vectorizer()
    matrix = vectorizer.fit_transform(texts)

    with open("models/bm25_vectorizer.pkl", "wb") as file:
        pickle.dump(vectorizer, file)

    sparse.save_npz("models/bm25_matrix.npz", matrix)

    return 0


def word2Vectorization():
    """
    This function vectorizes the corpus with word2vec.
    The model and matrix are saved in /models
    as binary files.
    :returns: 0 in case of successful execution
    """
    with open("data/metadata.json", "r") as file:
        data = json.load(file)

    texts = [data[idx]["tokens"] for idx in data]
    model = Word2Vec(sentences=texts, vector_size=100, window=5, min_count=1, workers=4)
    model.save("models/word2vec.model")

    matrix = np.array(
        [
            np.array([model.wv[word] for word in data[idx]["tokens"]]).mean(axis=0)
            for idx in tqdm.tqdm(data)
        ]
    )

    np.save("models/word2vec_matrix.npy", matrix)

    return 0


def fastTextVectorization():
    """
    This function vectorizes the corpus with fasttext.
    The model and matrix are saved in /models
    as binary files.
    :returns: 0 in case of successful execution
    """
    with open("data/metadata.json", "r") as file:
        data = json.load(file)

    texts = [data[idx]["tokens"] for idx in data]
    model = FastText(vector_size=4, window=3, min_count=1, sentences=texts, epochs=10)

    model.save("models/fasttext.model")

    matrix = np.array(
        [
            np.array([model.wv[word] for word in data[idx]["tokens"]])
            .mean(axis=0)
            .tolist()
            for idx in tqdm.tqdm(data)
        ]
    )

    np.save("models/fasttext_matrix.npy", matrix)

    return 0


# Uncomment any of the lines in order to execute the target vectorization
# and / or data preprocessing.

# corporaPreprocess()
# bm25Vectorization()
# word2Vectorization()
# fastTextVectorization()
