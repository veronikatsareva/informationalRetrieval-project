import json
import numpy as np
import pickle
from scipy import sparse
import spacy
from gensim.models import Word2Vec, FastText
from sklearn.metrics.pairwise import cosine_similarity
import argparse
from spellchecker import SpellChecker


def processQuery(query, indexType):
    """
    This function preprocesses query accordind to the target
    indexType (bm25, word2vec or fasttext).
    :argument query: a string with a query
    :argument indexType: a string with a target indexType (bm25, word2vec or fasttext)
    :returns: spellchecked query, vectorized query, matrix of the corpus
    """
    processer = spacy.load("en_core_web_sm")

    spell = SpellChecker()
    tokenizedQuery = [q.text for q in processer(query) if not q.is_punct]
    misspelled = spell.unknown(tokenizedQuery)
    checkedQuery = [
        word if word not in misspelled else spell.correction(word)
        for word in tokenizedQuery
    ]

    if indexType == "bm25":
        lemmatizedQuery = [q.lemma_ for q in processer(" ".join(checkedQuery))]

        with open("models/bm25_vectorizer.pkl", "rb") as file:
            vectorizer = pickle.load(file)

        vectorizedQuery = vectorizer.transform([" ".join(lemmatizedQuery)])

        matrix = sparse.load_npz("models/bm25_matrix.npz")

        return checkedQuery, vectorizedQuery, matrix

    if indexType == "word2vec":
        model = Word2Vec.load("models/word2vec.model")

        vectorizedQuery = np.array(
            [
                model.wv[word] if word in model.wv else np.zeros((1, 100))
                for word in checkedQuery
            ]
        ).mean(axis=0)

        matrix = np.load("models/word2vec_matrix.npy")

        return vectorizedQuery, matrix

    if indexType == "fasttext":
        model = FastText.load("models/fasttext.model")
        vectorizedQuery = np.array([model.wv[word] for word in checkedQuery]).mean(
            axis=0
        )

        matrix = np.load("models/fasttext_matrix.npy")

        return checkedQuery, vectorizedQuery, matrix


def search(query, indexType, rankNum):
    """
    This function is an implementation of the search.
    :argument query: a string with a query
    :argument indexType: a string with a target indexType (bm25, word2vec or fasttext)
    :argument rankNum: an integer number, the amount of the records that must be in results
    :returns: spellchecked query, relevant documents
    """
    checkedQuery, query, matrix = processQuery(query, indexType)
    results = []

    if indexType == "bm25":
        results = (matrix @ query.T).toarray().flatten()
    else:
        results = cosine_similarity(matrix, query.reshape(1, -1)).flatten()

    rank = np.argsort(results)[::-1]

    with open("data/metadata.json", "r") as file:
        data = json.load(file)

    texts = [data[str(idx)] for idx in rank if rank[idx] > 0][: int(rankNum)]
    request = " ".join(checkedQuery)

    return request, texts


def main():
    """
    This function is an implementation of the search with CLI interface.
    :returns: 0 in case of successful execution
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--query", help="Enter a text as a query.")
    parser.add_argument(
        "--idx", help="Enter a type of index: bm25, word2vec or fasttext."
    )
    parser.add_argument(
        "--top", help="The number of top documents that will be returned."
    )
    args = parser.parse_args()

    content = search(args.q, args.idx, args.N)

    for res in content:
        print(f"Title: {res['title']}")
        print(f"Release year: {res['release_year']}")
        print(f"Rating: {res['imdb_score']}")
        print(f"Plot: {res['description']}")
        print()

    return 0


if __name__ == "__main__":
    main()
