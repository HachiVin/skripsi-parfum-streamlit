# Perfume recommendations from free-text descriptions: TF-IDF vs Sentence-Transformer

Undergraduate thesis (S1 Teknik Informatika, Universitas Duta Bangsa Surakarta, 2026), also
published in the proceedings of an international conference. A user describes the perfume
they want in everyday Indonesian ("aroma segar untuk cuaca panas", a fresh scent for hot
weather). The system matches that description against 42,115 perfumes described in English
and returns the top 3 from two methods side by side, so their quality can be compared.

**Live demo:** [ISI: URL Streamlit Cloud]

**Publication:** Aryanto, K. D. R. R., Nastiti, F. E., & Irawan, R. D. (2026). Comparison of
TF-IDF and Sentence-Transformer NLP Methods for a Perfume Recommendation System Based on User
Descriptions with Streamlit Visualization. *Proceeding of the IC-ITECHS: International
Conference on Information Technology and Security*, 6(1), 149–160.
https://doi.org/10.32664/fn1g4768 ([article page](https://jurnal.ubhinus.ac.id/IC-ITECHS/article/view/2467))

![Top-3 recommendations of both methods side by side, each with a 0-2 relevance rating](docs/side_by_side.jpg)

## The problem

Many people buy perfume online because a product is trending, then find that it does not
suit them. E-commerce filters work on brand and price. Most buyers do not know note names
like bergamot or vetiver; they describe what they want in their own words, often in
Indonesian, while most perfume data is in English.

The question this project answers: for such descriptions, does a multilingual semantic model
(Sentence-Transformer) recommend better than a keyword method (TF-IDF), and by how much?

## Results

30 respondents, 180 relevance ratings (0 = not relevant, 1 = somewhat, 2 = very relevant),
27 distinct queries. Each respondent rated the top 3 of both methods for their own query.

| Method | Precision@3 | nDCG@3 | Mean rating (0–2) |
|---|---|---|---|
| TF-IDF | 0.833 | 0.850 | 1.311 |
| **Sentence-Transformer** | **0.867** | **0.912** | **1.456** |

![Precision@3 and nDCG@3 of both methods](docs/evaluation.png)

- An item counts as relevant for Precision@3 when it was rated 1 or 2. nDCG@3 uses the full
  0–2 rating, so it also rewards putting the best item first.
- The gap is larger on nDCG@3 (0.062) than on Precision@3 (0.034). Both methods usually find
  something relevant; Sentence-Transformer is better at ranking the most relevant perfume at
  the top.
- Per respondent: 11 scored Sentence-Transformer higher, 8 scored TF-IDF higher, 11 tied.

**Stability when the catalogue shrinks.** The 42,115 perfumes were split 80:20 (33,692 and
8,423) and every respondent query was run against each part. The table shows the mean cosine
similarity of the top 3.

| Method | 80% part | 20% part | Drop | Relative drop |
|---|---|---|---|---|
| TF-IDF | 0.1936 | 0.1221 | 0.0714 | about 37% |
| Sentence-Transformer | 0.6010 | 0.5884 | 0.0126 | about 2% |

No model is trained on either part, so this is not a train/test evaluation. It shows how much
each method's match quality depends on the size and vocabulary of the catalogue. The two
methods' scores are on different scales, which is why the relative drop is the fairer
comparison.

## How it works

```mermaid
flowchart LR
    D[("Parfumo dataset<br/>59,325 perfumes")] --> P["01_prepare_data.py<br/>drop 17,210 without scent data<br/>build text_profile"]
    P --> T["TF-IDF<br/>unigrams + bigrams<br/>42,115 × 116,402 sparse"]
    P --> S["Sentence-Transformer<br/>paraphrase-multilingual-MiniLM-L12-v2<br/>42,115 × 384 dense"]
    Q["Indonesian query"] --> T
    Q --> S
    T --> R["Top-3 by<br/>cosine similarity"]
    S --> R
    R --> A["Streamlit app<br/>respondent rates each result 0-2"]
    A --> G[("Google Sheets<br/>ratings")]
```

1. **Data.** [Parfumo Fragrance Dataset](https://github.com/rfordatascience/tidytuesday/tree/main/data/2024/2024-12-10)
   from TidyTuesday (December 2024). Name, brand, main accords and top, middle and base notes
   are kept. Perfumes with no scent information at all (17,210) are removed, leaving 42,115.
2. **One text per perfume.** The six fields are joined into a labelled `text_profile`, for
   example `Name: Tidal Pool Brand: CB I Hate Perfume Main accords: Top notes: Bergamot ...`.
   Both methods receive exactly the same text, so the comparison is about the representation
   only.
3. **TF-IDF.** scikit-learn `TfidfVectorizer` with English stop words and `ngram_range=(1, 2)`,
   so two-word phrases like "woody amber" count.
4. **Sentence-Transformer.** The pre-trained multilingual model
   `paraphrase-multilingual-MiniLM-L12-v2`, without fine-tuning. Embeddings are normalised and
   saved to `sentence_transformer_embeddings.npy`, so the app loads them instead of encoding
   42,115 texts at start-up.
5. **Query expansion.** Before matching, the app appends English equivalents for 41 common
   Indonesian words found in the query, for example manis → sweet, segar → fresh, cewek →
   female feminine (`QUERY_TRANSLATION` in `app.py`). There is no translation service or
   model; both methods get the same expanded query.
6. **Ranking.** Cosine similarity between the query and every perfume; the top 3 of each
   method are shown.
7. **User study.** The Streamlit app asks for gender, age range and perfume experience (no
   names), takes a free-text description, shows both top 3 lists and saves the ratings to
   Google Sheets with an anonymous respondent code.

## Run it

Python 3.11 (the version of the dev container).

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app expects `parfumo_text_profile.csv` and `sentence_transformer_embeddings.npy`, both in
the repository. If the embeddings file is missing, it is created on first run, which takes a
few minutes.

Saving ratings needs a Google service account with edit access to the target spreadsheet.
Put its key in `.streamlit/secrets.toml` locally, or in *Settings → Secrets* on Streamlit
Cloud. The file is git-ignored; never commit it.

```toml
SPREADSHEET_ID = "your-spreadsheet-id"

[gcp_service_account]
type = "service_account"
project_id = "..."
private_key_id = "..."
private_key = "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
client_email = "...@....iam.gserviceaccount.com"
client_id = "..."
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
client_x509_cert_url = "..."
universe_domain = "googleapis.com"
```

The scripts reproduce the steps without the app:

| Script | What it does |
|---|---|
| `01_prepare_data.py` | Downloads the Parfumo data, filters it and writes `parfumo_text_profile.csv` |
| `02_tfidf_recommend.py` | Top 3 for a query typed in the terminal, TF-IDF |
| `03_sentence_transformer_recommend.py` | Same with Sentence-Transformer; builds the embeddings file if missing |
| `04_compare_methods.py` | Both methods for one query, saved to `comparison_result.csv` |

## Decisions and trade-offs

- **Same text, same similarity, two representations.** Earlier work on perfume recommendation
  kept one representation and compared similarity measures (cosine vs Jaccard). This study
  does the opposite, so any difference comes from how the text is represented.
- **A pre-trained multilingual model, without fine-tuning.** The result shows what an
  off-the-shelf model already does across Indonesian and English; fine-tuning on Indonesian
  perfume descriptions is left as future work.
- **Real users instead of synthetic queries.** Relevance of a perfume to a description is a
  human judgement, so the evaluation is built on ratings from 30 people writing their own
  queries, not on queries invented by the author.
- **Precomputed embeddings.** Encoding 42,115 profiles takes minutes, so it is done once and
  saved; the Streamlit Cloud app only loads the `.npy` file and encodes the query.

## Limitations

- **Small study.** 30 respondents and 27 queries, and no statistical significance test. The
  numbers show a consistent direction, not a proven general difference.
- **Ratings were not blind.** The app labels each list with its method and always shows
  TF-IDF first, which may have influenced some ratings.
- **The query expansion glossary helps keyword matching.** TF-IDF can only match words that
  exist in the English data, so the 41 Indonesian–English pairs are likely to help TF-IDF
  most.
  Sentence-Transformer handles Indonesian without them. The published paper describes the
  setup as translation-free, meaning no translation service or model.
- **No negation.** Neither method understands "not". In one test query, "saya tidak suka
  aroma vanilla" (I do not like vanilla), TF-IDF ranked *Heaven Scent Vanilla* first.
- **Gaps in the data.** Many perfumes have only main accords or only some note lists, which
  limits what either method can match.

Suggested next steps from the thesis: fine-tune the model on Indonesian perfume descriptions,
recruit more respondents, and combine both methods in a hybrid ranking.

## Repository layout

```
app.py                                  Streamlit app used for the user study
01_prepare_data.py … 04_compare_methods.py   data preparation and command-line versions
parfumo_text_profile.csv                42,115 cleaned perfume profiles
sentence_transformer_embeddings.npy     precomputed 384-dimension embeddings
comparison_result.csv                   example output of 04_compare_methods.py
feedback_responden.csv                  example rows of the rating format
docs/                                   images for this README
requirements.txt, .devcontainer/        environment
```

The app interface is in Indonesian, for the Indonesian respondents.

## Citation

```bibtex
@inproceedings{aryanto2026perfume,
  title     = {Comparison of TF-IDF and Sentence-Transformer NLP Methods for a Perfume
               Recommendation System Based on User Descriptions with Streamlit Visualization},
  author    = {Aryanto, Kevin De Rafael Rio and Nastiti, Faulinda Ely and Irawan, Ridwan Dwi},
  booktitle = {Proceeding of the IC-ITECHS: International Conference on Information
               Technology and Security},
  volume    = {6},
  number    = {1},
  pages     = {149--160},
  year      = {2026},
  doi       = {10.32664/fn1g4768}
}
```