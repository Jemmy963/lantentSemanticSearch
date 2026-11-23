from sklearn.decomposition import TruncatedSVD


def apply_svd(tfidf_matrix):
    # Reduce the TF-IDF matrix while preserving its most useful patterns.
    number_of_documents, number_of_terms = tfidf_matrix.shape

    components = min(100, number_of_documents - 1, number_of_terms - 1)
    if components < 1:
        raise ValueError("upload at least two documents with different terms.")

    svd = TruncatedSVD(n_components=components, random_state=42)
    reduced_matrix = svd.fit_transform(tfidf_matrix)

    return svd, reduced_matrix
