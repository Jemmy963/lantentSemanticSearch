from matrix import clean_text
from sklearn.metrics.pairwise import cosine_similarity

def search_documents(
        query,
        vectoriser,
        svd,
        document_vectors,
        filenames
):
    #find and rank documents according to user's query.

    cleaned_query = clean_text(query)

    query_tfidf = vectoriser.transform([cleaned_query])
    query_vector = svd.transform(query_tfidf)

    scores = cosine_similarity(query_vector,document_vectors)[0]

    ranked_results = sorted(zip(filenames,scores),key=lambda item: item[1],reverse=True)

    return ranked_results