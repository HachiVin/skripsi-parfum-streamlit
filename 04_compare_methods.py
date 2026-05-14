import os
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


DATA_PATH = "parfumo_text_profile.csv"
EMBEDDING_PATH = "sentence_transformer_embeddings.npy"
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


def load_data():
    df = pd.read_csv(DATA_PATH)

    columns = [
        "Name",
        "Brand",
        "Main_Accords",
        "Top_Notes",
        "Middle_Notes",
        "Base_Notes",
        "text_profile"
    ]

    for col in columns:
        if col in df.columns:
            df[col] = df[col].fillna("")

    return df


def build_tfidf(df):
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2)
    )

    tfidf_matrix = vectorizer.fit_transform(df["text_profile"])
    return vectorizer, tfidf_matrix


def load_sentence_transformer(df):
    model = SentenceTransformer(MODEL_NAME)

    if os.path.exists(EMBEDDING_PATH):
        embeddings = np.load(EMBEDDING_PATH)
    else:
        embeddings = model.encode(
            df["text_profile"].tolist(),
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True
        )
        np.save(EMBEDDING_PATH, embeddings)

    return model, embeddings


def recommend_tfidf(query, df, vectorizer, tfidf_matrix, top_k=3):
    query_vector = vectorizer.transform([query])
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()

    top_indices = similarity_scores.argsort()[::-1][:top_k]

    results = df.iloc[top_indices].copy()
    results["method"] = "TF-IDF"
    results["rank"] = range(1, top_k + 1)
    results["similarity_score"] = similarity_scores[top_indices]

    return results


def recommend_sentence_transformer(query, df, model, embeddings, top_k=3):
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )

    similarity_scores = np.dot(query_embedding, embeddings.T).flatten()

    top_indices = similarity_scores.argsort()[::-1][:top_k]

    results = df.iloc[top_indices].copy()
    results["method"] = "Sentence-Transformer"
    results["rank"] = range(1, top_k + 1)
    results["similarity_score"] = similarity_scores[top_indices]

    return results


def show_results(title, results):
    print(f"\n{title}")
    print("-" * 100)

    columns_to_show = [
        "rank",
        "Name",
        "Brand",
        "Main_Accords",
        "Top_Notes",
        "Middle_Notes",
        "Base_Notes",
        "similarity_score"
    ]

    print(results[columns_to_show].to_string(index=False))


def main():
    df = load_data()

    print("Membangun model TF-IDF...")
    vectorizer, tfidf_matrix = build_tfidf(df)

    print("Memuat model Sentence-Transformer...")
    st_model, st_embeddings = load_sentence_transformer(df)

    print("\nSemua model berhasil dimuat.")
    print(f"Jumlah parfum: {len(df)}")

    query = input("\nMasukkan deskripsi parfum yang diinginkan: ")

    tfidf_results = recommend_tfidf(
        query=query,
        df=df,
        vectorizer=vectorizer,
        tfidf_matrix=tfidf_matrix,
        top_k=3
    )

    st_results = recommend_sentence_transformer(
        query=query,
        df=df,
        model=st_model,
        embeddings=st_embeddings,
        top_k=3
    )

    show_results("Top-3 Rekomendasi TF-IDF", tfidf_results)
    show_results("Top-3 Rekomendasi Sentence-Transformer", st_results)

    comparison_results = pd.concat([tfidf_results, st_results], ignore_index=True)

    output_columns = [
        "method",
        "rank",
        "Name",
        "Brand",
        "Main_Accords",
        "Top_Notes",
        "Middle_Notes",
        "Base_Notes",
        "similarity_score"
    ]

    comparison_results[output_columns].to_csv(
        "comparison_result.csv",
        index=False,
        encoding="utf-8"
    )

    print("\nHasil perbandingan berhasil disimpan ke file: comparison_result.csv")


if __name__ == "__main__":
    main()