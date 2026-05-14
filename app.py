import os
import re
import uuid
from datetime import datetime

import gspread
import numpy as np
import pandas as pd
import streamlit as st

from google.oauth2.service_account import Credentials
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


# =========================
# KONFIGURASI
# =========================

DATA_TRAIN_PATH = "parfumo_train.csv"
DATA_TEST_PATH = "parfumo_test.csv"
EMBEDDING_PATH = "sentence_transformer_test_embeddings.npy"

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
TOP_K = 3


# =========================
# KONFIGURASI STREAMLIT
# =========================

st.set_page_config(
    page_title="Rekomendasi Parfum",
    page_icon="🌸",
    layout="wide"
)


# =========================
# CSS MOBILE FRIENDLY
# =========================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 28px;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        font-size: 15px;
        color: #B8B8B8;
        margin-bottom: 1rem;
    }

    .recommendation-card {
        border: 1px solid rgba(180, 180, 180, 0.25);
        border-radius: 14px;
        padding: 16px;
        margin-bottom: 10px;
        background-color: rgba(255, 255, 255, 0.03);
    }

    .rank-badge {
        font-size: 14px;
        font-weight: 700;
        margin-bottom: 6px;
        color: #7EC8FF;
    }

    .perfume-name {
        font-size: 20px;
        font-weight: 700;
        margin-bottom: 4px;
    }

    .brand-name {
        font-size: 14px;
        color: #CCCCCC;
        margin-bottom: 12px;
    }

    .field-label {
        font-weight: 700;
    }

    .score-wrapper {
        margin-bottom: 28px;
        padding-bottom: 12px;
        border-bottom: 1px solid rgba(180, 180, 180, 0.2);
    }

    @media (max-width: 768px) {
        .main-title {
            font-size: 24px;
        }

        .perfume-name {
            font-size: 18px;
        }

        .recommendation-card {
            padding: 14px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================
# KAMUS TERJEMAHAN SEDERHANA
# =========================

QUERY_TRANSLATION = {
    "manis": "sweet",
    "segar": "fresh",
    "lembut": "soft",
    "soft": "soft",
    "kalem": "soft calm gentle",
    "cewe": "female feminine",
    "cewek": "female feminine",
    "wanita": "female feminine",
    "perempuan": "female feminine",
    "pria": "male masculine",
    "cowo": "male masculine",
    "cowok": "male masculine",
    "malam": "night",
    "siang": "day",
    "pagi": "morning",
    "kantor": "office",
    "pekerjaan": "work office",
    "kerja": "work office",
    "indoor": "indoor office",
    "formal": "formal",
    "santai": "casual",
    "mewah": "luxury elegant",
    "elegan": "elegant",
    "aroma": "scent",
    "wangi": "fragrance",
    "parfum": "perfume",
    "buah": "fruity fruit",
    "buah-buahan": "fruity fruit",
    "berry": "berry fruity",
    "beri": "berry fruity",
    "vanila": "vanilla",
    "mawar": "rose",
    "melati": "jasmine",
    "jeruk": "citrus orange lemon",
    "kayu": "woody wood",
    "bedak": "powdery",
    "tembakau": "tobacco",
    "kulit": "leather",
    "kopi": "coffee",
    "coklat": "chocolate",
    "kelapa": "coconut",
}


# =========================
# GOOGLE SHEETS
# =========================

@st.cache_resource
def get_google_sheet():
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets"
    ]

    credentials = Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=scopes
    )

    client = gspread.authorize(credentials)
    spreadsheet = client.open_by_key(st.secrets["SPREADSHEET_ID"])
    worksheet = spreadsheet.sheet1

    return worksheet


def save_feedback_to_gsheet(
    respondent_id,
    gender,
    age_range,
    experience_level,
    desired_query,
    tfidf_results,
    st_results,
    scores
):
    worksheet = get_google_sheet()

    all_results = pd.concat([tfidf_results, st_results], ignore_index=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    rows = []

    for idx, row in all_results.iterrows():
        rows.append([
            timestamp,
            respondent_id,
            gender,
            age_range,
            experience_level,
            desired_query,
            "",  # excluded_input dikosongkan karena filtering dihapus
            "",  # excluded_notes dikosongkan karena filtering dihapus
            row["method"],
            int(row["rank"]),
            str(row["Name"]),
            str(row["Brand"]),
            str(row["Main_Accords"]),
            str(row["Top_Notes"]),
            str(row["Middle_Notes"]),
            str(row["Base_Notes"]),
            float(row["similarity_score"]),
            int(scores[idx])
        ])

    worksheet.append_rows(rows, value_input_option="USER_ENTERED")


# =========================
# LOAD DATA DAN MODEL
# =========================

@st.cache_data
def load_split_data():
    train_df = pd.read_csv(DATA_TRAIN_PATH)
    test_df = pd.read_csv(DATA_TEST_PATH)

    columns = [
        "Name",
        "Brand",
        "Main_Accords",
        "Top_Notes",
        "Middle_Notes",
        "Base_Notes",
        "text_profile"
    ]

    for df in [train_df, test_df]:
        for col in columns:
            if col in df.columns:
                df[col] = df[col].fillna("").astype(str)

    return train_df, test_df


@st.cache_resource
def build_tfidf(train_text_profiles, test_text_profiles):
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2)
    )

    vectorizer.fit(train_text_profiles)
    test_tfidf_matrix = vectorizer.transform(test_text_profiles)

    return vectorizer, test_tfidf_matrix


@st.cache_resource
def load_sentence_model():
    return SentenceTransformer(MODEL_NAME)


@st.cache_data
def load_or_create_test_embeddings(test_text_profiles):
    model = SentenceTransformer(MODEL_NAME)

    recompute = True

    if os.path.exists(EMBEDDING_PATH):
        embeddings = np.load(EMBEDDING_PATH)

        if embeddings.shape[0] == len(test_text_profiles):
            recompute = False

    if recompute:
        embeddings = model.encode(
            list(test_text_profiles),
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True
        )
        np.save(EMBEDDING_PATH, embeddings)

    return embeddings


# =========================
# UTILITAS
# =========================

def generate_respondent_id():
    today = datetime.now().strftime("%Y%m%d")
    unique_code = uuid.uuid4().hex[:6].upper()
    return f"R-{today}-{unique_code}"


def normalize_query(query):
    query_lower = query.lower()
    additional_terms = []

    for indo_word, english_word in QUERY_TRANSLATION.items():
        pattern = rf"\b{re.escape(indo_word.lower())}\b"
        if re.search(pattern, query_lower):
            additional_terms.append(english_word)

    if additional_terms:
        query = query + " " + " ".join(additional_terms)

    return query


def prepare_query_for_modeling(desired_query):
    processed_query = normalize_query(desired_query.lower().strip())
    return processed_query


# =========================
# REKOMENDASI
# =========================

def recommend_tfidf(desired_query, test_df, vectorizer, test_tfidf_matrix, top_k=3):
    processed_query = prepare_query_for_modeling(desired_query)

    query_vector = vectorizer.transform([processed_query])
    similarity_scores = cosine_similarity(query_vector, test_tfidf_matrix).flatten()

    top_indices = similarity_scores.argsort()[::-1][:top_k]

    results = test_df.iloc[top_indices].copy()
    results["method"] = "TF-IDF"
    results["rank"] = range(1, top_k + 1)
    results["similarity_score"] = similarity_scores[top_indices]

    return results


def recommend_sentence_transformer(desired_query, test_df, model, test_embeddings, top_k=3):
    processed_query = prepare_query_for_modeling(desired_query)

    query_embedding = model.encode(
        [processed_query],
        normalize_embeddings=True
    )

    similarity_scores = np.dot(query_embedding, test_embeddings.T).flatten()

    top_indices = similarity_scores.argsort()[::-1][:top_k]

    results = test_df.iloc[top_indices].copy()
    results["method"] = "Sentence-Transformer"
    results["rank"] = range(1, top_k + 1)
    results["similarity_score"] = similarity_scores[top_indices]

    return results


# =========================
# TAMPILAN KARTU DAN PENILAIAN
# =========================

def clean_value(value):
    if value is None:
        return "-"
    value = str(value).strip()
    if value == "" or value.lower() == "nan":
        return "-"
    return value


def render_recommendation_card(row):
    rank = clean_value(row["rank"])
    name = clean_value(row["Name"])
    brand = clean_value(row["Brand"])
    main_accords = clean_value(row["Main_Accords"])
    top_notes = clean_value(row["Top_Notes"])
    middle_notes = clean_value(row["Middle_Notes"])
    base_notes = clean_value(row["Base_Notes"])
    similarity_score = round(float(row["similarity_score"]), 4)

    st.markdown(
        f"""
        <div class="recommendation-card">
            <div class="rank-badge">Rank {rank}</div>
            <div class="perfume-name">{name}</div>
            <div class="brand-name">Brand: {brand}</div>
            <p><span class="field-label">Main Accords:</span> {main_accords}</p>
            <p><span class="field-label">Top Notes:</span> {top_notes}</p>
            <p><span class="field-label">Middle Notes:</span> {middle_notes}</p>
            <p><span class="field-label">Base Notes:</span> {base_notes}</p>
            <p><span class="field-label">Similarity Score:</span> {similarity_score}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_recommendation_with_score(row, key):
    render_recommendation_card(row)

    score = st.selectbox(
        "Nilai relevansi parfum ini:",
        [0, 1, 2],
        key=key,
        help="0 = tidak relevan, 1 = cukup relevan, 2 = sangat relevan"
    )

    st.markdown('<div class="score-wrapper"></div>', unsafe_allow_html=True)

    return score


# =========================
# MAIN APP
# =========================

def main():
    if "respondent_id" not in st.session_state:
        st.session_state["respondent_id"] = generate_respondent_id()

    st.markdown(
        '<div class="main-title">Rekomendasi Parfum Berbasis Deskripsi Pengguna</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="subtitle">Perbandingan metode TF-IDF dan Sentence-Transformer dengan visualisasi Streamlit.</div>',
        unsafe_allow_html=True
    )

    st.info(
        "Penilaian dilakukan berdasarkan informasi parfum yang ditampilkan, seperti "
        "**main accords**, **top notes**, **middle notes**, dan **base notes**. "
        "Anda tidak harus pernah mencoba parfum tersebut. Berikan nilai berdasarkan "
        "seberapa sesuai karakter parfum dengan deskripsi yang Anda tulis."
    )

    with st.expander("Panduan singkat pengisian"):
        st.write(
            """
            1. Isi data responden.
            2. Tulis deskripsi parfum yang Anda inginkan.
            3. Klik tombol **Tampilkan Rekomendasi**.
            4. Baca informasi parfum yang muncul.
            5. Berikan nilai relevansi pada setiap parfum yang direkomendasikan.
            """
        )

        st.write(
            """
            **Contoh deskripsi:**  
            Saya ingin parfum wanita yang manis, soft, kalem, dan cocok untuk pekerjaan indoor.
            """
        )

    with st.spinner("Memuat data training, data testing, dan model..."):
        train_df, test_df = load_split_data()

        vectorizer, test_tfidf_matrix = build_tfidf(
            tuple(train_df["text_profile"].tolist()),
            tuple(test_df["text_profile"].tolist())
        )

        sentence_model = load_sentence_model()
        test_embeddings = load_or_create_test_embeddings(
            tuple(test_df["text_profile"].tolist())
        )

    st.success(
        f"Data berhasil dimuat. Training: {len(train_df)} parfum | Testing kandidat rekomendasi: {len(test_df)} parfum"
    )

    st.subheader("Data Responden")

    respondent_id = st.session_state["respondent_id"]
    st.text_input("Kode Responden Otomatis", value=respondent_id, disabled=True)

    gender = st.selectbox(
        "Gender",
        ["Tidak ingin menyebutkan", "Laki-laki", "Perempuan"]
    )

    age_range = st.selectbox(
        "Rentang Usia",
        ["< 17", "17-20", "21-25", "26-30", "> 30"]
    )

    experience_level = st.selectbox(
        "Pengalaman Menggunakan Parfum",
        [
            "Pemula",
            "Pengguna biasa",
            "Cukup memahami parfum",
            "Sangat memahami parfum"
        ]
    )

    st.subheader("Input Preferensi Parfum")

    desired_query = st.text_area(
        "Tuliskan deskripsi parfum yang Anda inginkan:",
        placeholder=(
            "Contoh: saya ingin parfum wanita yang manis, soft, kalem, "
            "dan cocok untuk pekerjaan indoor"
        )
    )

    run_button = st.button("Tampilkan Rekomendasi")

    if run_button:
        if not desired_query.strip():
            st.warning("Silakan isi deskripsi parfum yang diinginkan terlebih dahulu.")
            return

        with st.expander("Detail pemrosesan input"):
            processed_query = prepare_query_for_modeling(desired_query)

            st.write("Deskripsi yang diinginkan:")
            st.write(desired_query)

            st.write("Query yang dipakai model:")
            st.write(processed_query)

            st.write("Catatan:")
            st.write("Rekomendasi diambil dari data testing, sedangkan TF-IDF di-fit menggunakan data training.")

        tfidf_results = recommend_tfidf(
            desired_query=desired_query,
            test_df=test_df,
            vectorizer=vectorizer,
            test_tfidf_matrix=test_tfidf_matrix,
            top_k=TOP_K
        )

        st_results = recommend_sentence_transformer(
            desired_query=desired_query,
            test_df=test_df,
            model=sentence_model,
            test_embeddings=test_embeddings,
            top_k=TOP_K
        )

        st.session_state["desired_query"] = desired_query
        st.session_state["tfidf_results"] = tfidf_results
        st.session_state["st_results"] = st_results
        st.session_state["recommendation_token"] = uuid.uuid4().hex[:8]

    if "tfidf_results" in st.session_state and "st_results" in st.session_state:
        st.subheader("Hasil Rekomendasi Top-3 dan Penilaian Relevansi")

        st.write(
            "Baca informasi setiap parfum, lalu langsung beri nilai relevansi di bawah parfum tersebut."
        )

        st.write(
            "**Panduan penilaian:**  \n"
            "- **0 = Tidak relevan**, jika karakter parfum tidak sesuai dengan deskripsi.  \n"
            "- **1 = Cukup relevan**, jika karakter parfum hanya sesuai sebagian.  \n"
            "- **2 = Sangat relevan**, jika karakter parfum sangat sesuai dengan deskripsi."
        )

        token = st.session_state.get("recommendation_token", "default")

        with st.form("feedback_form"):
            tab_tfidf, tab_st = st.tabs(["TF-IDF", "Sentence-Transformer"])

            tfidf_scores = []
            st_scores = []

            with tab_tfidf:
                st.markdown("### TF-IDF")
                for idx, row in st.session_state["tfidf_results"].reset_index(drop=True).iterrows():
                    score = render_recommendation_with_score(
                        row,
                        key=f"{token}_tfidf_score_{idx}"
                    )
                    tfidf_scores.append(score)

            with tab_st:
                st.markdown("### Sentence-Transformer")
                for idx, row in st.session_state["st_results"].reset_index(drop=True).iterrows():
                    score = render_recommendation_with_score(
                        row,
                        key=f"{token}_st_score_{idx}"
                    )
                    st_scores.append(score)

            scores = tfidf_scores + st_scores

            submit = st.form_submit_button("Simpan Penilaian")

            if submit:
                try:
                    save_feedback_to_gsheet(
                        respondent_id=st.session_state["respondent_id"],
                        gender=gender,
                        age_range=age_range,
                        experience_level=experience_level,
                        desired_query=st.session_state["desired_query"],
                        tfidf_results=st.session_state["tfidf_results"],
                        st_results=st.session_state["st_results"],
                        scores=scores
                    )

                    st.success(
                        "Penilaian berhasil disimpan ke Google Sheets. Terima kasih sudah menjadi responden."
                    )

                    st.session_state["respondent_id"] = generate_respondent_id()

                except Exception as error:
                    st.error("Gagal menyimpan data ke Google Sheets.")
                    st.write(error)


if __name__ == "__main__":
    main()