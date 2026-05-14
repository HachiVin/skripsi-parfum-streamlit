import pandas as pd
from sklearn.model_selection import train_test_split


DATASET_URL = "https://raw.githubusercontent.com/rfordatascience/tidytuesday/main/data/2024/2024-12-10/parfumo_data_clean.csv"

OUTPUT_ALL = "parfumo_text_profile.csv"
OUTPUT_TRAIN = "parfumo_train.csv"
OUTPUT_TEST = "parfumo_test.csv"

TEST_SIZE = 0.2
RANDOM_STATE = 42


def main():
    print("Membaca dataset Parfumo...")
    df = pd.read_csv(DATASET_URL)

    print("\nUkuran dataset awal:")
    print(df.shape)

    selected_columns = [
        "Name",
        "Brand",
        "Main_Accords",
        "Top_Notes",
        "Middle_Notes",
        "Base_Notes",
    ]

    missing_columns = [col for col in selected_columns if col not in df.columns]
    if missing_columns:
        print("\nKolom berikut tidak ditemukan:")
        print(missing_columns)
        return

    df_selected = df[selected_columns].copy()
    df_selected = df_selected.fillna("")

    scent_columns = [
        "Main_Accords",
        "Top_Notes",
        "Middle_Notes",
        "Base_Notes",
    ]

    before_filter = len(df_selected)

    df_selected = df_selected[
        df_selected[scent_columns].apply(
            lambda row: any(str(value).strip() != "" for value in row),
            axis=1
        )
    ].copy()

    after_filter = len(df_selected)
    removed_data = before_filter - after_filter

    print("\nJumlah data sebelum filter aroma:")
    print(before_filter)

    print("\nJumlah data setelah filter aroma:")
    print(after_filter)

    print("\nJumlah data yang dihapus karena tidak memiliki informasi aroma:")
    print(removed_data)

    df_selected = df_selected.reset_index(drop=True)
    df_selected.insert(0, "perfume_id", range(1, len(df_selected) + 1))

    df_selected["text_profile"] = (
        "Name: " + df_selected["Name"].astype(str) + " " +
        "Brand: " + df_selected["Brand"].astype(str) + " " +
        "Main accords: " + df_selected["Main_Accords"].astype(str) + " " +
        "Top notes: " + df_selected["Top_Notes"].astype(str) + " " +
        "Middle notes: " + df_selected["Middle_Notes"].astype(str) + " " +
        "Base notes: " + df_selected["Base_Notes"].astype(str)
    )

    train_df, test_df = train_test_split(
        df_selected,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True
    )

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    df_selected.to_csv(OUTPUT_ALL, index=False, encoding="utf-8-sig")
    train_df.to_csv(OUTPUT_TRAIN, index=False, encoding="utf-8-sig")
    test_df.to_csv(OUTPUT_TEST, index=False, encoding="utf-8-sig")

    print("\nBerhasil membuat file:")
    print(f"1. {OUTPUT_ALL}")
    print(f"2. {OUTPUT_TRAIN}")
    print(f"3. {OUTPUT_TEST}")

    print("\nJumlah data training:")
    print(len(train_df))

    print("\nJumlah data testing:")
    print(len(test_df))

    print("\nContoh data testing:")
    print(test_df[["perfume_id", "Name", "Brand", "Main_Accords"]].head())


if __name__ == "__main__":
    main()