import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DATA_PATH = "parfumo_text_profile.csv"


def load_data():
    df = pd.read_csv(DATA_PATH)
    df["text_profile"] = df["text_profile"].fillna("")
    return df


def build_tfidf_model(df):
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2)
    )

    tfidf_matrix = vectorizer.fit_transform(df["text_profile"])
    return vectorizer, tfidf_matrix


def recommend_tfidf(query, df, vectorizer, tfidf_matrix, top_k=3):
    query_vector = vectorizer.transform([query])
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()

    top_indices = similarity_scores.argsort()[::-1][:top_k]

    results = df.iloc[top_indices].copy()
    results["similarity_score"] = similarity_scores[top_indices]

    columns_to_show = [
        "Name",
        "Brand",
        "Main_Accords",
        "Top_Notes",
        "Middle_Notes",
        "Base_Notes",
        "similarity_score"
    ]

    return results[columns_to_show]


def main():
    df = load_data()
    vectorizer, tfidf_matrix = build_tfidf_model(df)

    print("Model TF-IDF berhasil dibuat.")
    print(f"Jumlah parfum: {len(df)}")

    query = input("\nMasukkan deskripsi parfum yang diinginkan: ")

    recommendations = recommend_tfidf(
        query=query,
        df=df,
        vectorizer=vectorizer,
        tfidf_matrix=tfidf_matrix,
        top_k=3
    )

    print("\nTop-3 Rekomendasi TF-IDF:")
    print(recommendations.to_string(index=False))


if __name__ == "__main__":
    main()