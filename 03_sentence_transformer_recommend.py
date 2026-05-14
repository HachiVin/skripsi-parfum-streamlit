import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


DATA_PATH = "parfumo_text_profile.csv"
EMBEDDING_PATH = "sentence_transformer_embeddings.npy"

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


def load_data():
    df = pd.read_csv(DATA_PATH)

    text_columns = [
        "Name",
        "Brand",
        "Main_Accords",
        "Top_Notes",
        "Middle_Notes",
        "Base_Notes",
        "text_profile"
    ]

    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].fillna("")

    return df


def build_or_load_embeddings(df, model):
    if os.path.exists(EMBEDDING_PATH):
        print("Memuat embedding yang sudah ada...")
        embeddings = np.load(EMBEDDING_PATH)
    else:
        print("Membuat embedding Sentence-Transformer...")
        print("Proses ini bisa memakan waktu beberapa menit pada pertama kali dijalankan.")

        embeddings = model.encode(
            df["text_profile"].tolist(),
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True
        )

        np.save(EMBEDDING_PATH, embeddings)
        print(f"Embedding berhasil disimpan ke {EMBEDDING_PATH}")

    return embeddings


def recommend_sentence_transformer(query, df, model, embeddings, top_k=3):
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )

    similarity_scores = np.dot(query_embedding, embeddings.T).flatten()

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

    print("Memuat model Sentence-Transformer...")
    model = SentenceTransformer(MODEL_NAME)

    embeddings = build_or_load_embeddings(df, model)

    print("\nModel Sentence-Transformer berhasil dibuat.")
    print(f"Jumlah parfum: {len(df)}")

    query = input("\nMasukkan deskripsi parfum yang diinginkan: ")

    recommendations = recommend_sentence_transformer(
        query=query,
        df=df,
        model=model,
        embeddings=embeddings,
        top_k=3
    )

    print("\nTop-3 Rekomendasi Sentence-Transformer:")
    print(recommendations.to_string(index=False))


if __name__ == "__main__":
    main()
    