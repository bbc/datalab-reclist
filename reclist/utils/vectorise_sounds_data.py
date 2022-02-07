import json

import numpy as np

from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder


def generate_metadata_vectors(
    enriched_items_path='/Users/piscoa01/Downloads/bbc-datalab-sounds-pesquet_dataset_2021-09_item_metadata_210930_all_items_enriched.ndjson',
    pca=False):
    """
    Generates vectors from BBC Sounds resource metadata.
    @param enriched_items_path: str, path of an ndjson file containing item metadata.
    Only genres and masterbrand are used at the moment.
    @return: Dict resource_id:metadata_vector

    """

    with open(enriched_items_path, 'r') as f:
        enriched_items = [json.loads(line) for line in f]
    # extract genres
    for item in enriched_items:
        try:
            item['genres_labels'] = ' '.join(
                [g.get('string', '').replace(' ', '-').lower() for g in item.get('genres_inherited_first', [])])
        except TypeError:
            # print(item.get('genres_inherited_first'))
            item['genres_labels'] = 'NoGenreInfoAvailable'

    # We might switch to tf-idf, instead of countvectoriser
    # vectorizer = TfidfVectorizer(strip_accents='unicode')
    # tfidf_genres = vectorizer.fit_transform([item['genres_labels'] for item in enriched_items])

    countvectoriser = CountVectorizer(strip_accents='unicode')
    vectorised_genres = countvectoriser.fit_transform([item['genres_labels'] for item in enriched_items]).toarray()

    # vectorise masterbrands
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit([[item['master_brand']] for item in enriched_items])
    one_hot_masterbrands = enc.transform([[item['master_brand']] for item in enriched_items]).toarray()

    # add parameter to include brand?

    # hstack vectors
    metadata_vectors = np.hstack((vectorised_genres, one_hot_masterbrands))

    if pca:
        pca = PCA(n_components=0.95)
        pca.fit(metadata_vectors)
        reduced_metadata_vectors = pca.transform(metadata_vectors)
        return {enriched_items[ix]['resource_id']: reduced_metadata_vectors[ix] for ix, value in
                enumerate(enriched_items)}
    else:
        return {enriched_items[ix]['resource_id']: metadata_vectors[ix] for ix, value in enumerate(enriched_items)}
