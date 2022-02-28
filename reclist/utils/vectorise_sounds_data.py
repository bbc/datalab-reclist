import numpy as np

from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder


def extract_bbc_sounds_genres(enriched_items, results_as_list=True):
    """Extracts genre labels from genres_inherited_first dicts."""
    # extract genres
    for item in enriched_items:
        try:
            item['genres_labels'] = [g.get('string', '').replace(' ', '-').lower() for g in
                                     item.get('genres_inherited_first', [])] if results_as_list else ' '.join(
                [g.get('string', '').replace(' ', '-').lower() for g in item.get('genres_inherited_first', [])])
        except TypeError:
            # print(item.get('genres_inherited_first'))
            item['genres_labels'] = ['NoGenreInfoAvailable'] if results_as_list else 'NoGenreInfoAvailable'

    return enriched_items


def generate_genre_dict(enriched_items):
    return {item['resource_id']: item['genres_labels'] for item in extract_bbc_sounds_genres(enriched_items)}


def generate_metadata_vectors(enriched_items, pca=False):
    """
    Generates vectors from BBC Sounds resource metadata.
    @param enriched_items: list of dicts, containing item metadata
    @param pca: bool, whether to apply PCA to content vectors
    Only genres and masterbrand are used at the moment.
    @return: Dict resource_id:metadata_vector

    """

    # extract genres
    enriched_items = extract_bbc_sounds_genres(enriched_items, results_as_list=False)

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
