import pandas as pd


DATASET_URL = "https://raw.githubusercontent.com/rfordatascience/tidytuesday/main/data/2024/2024-12-10/parfumo_data_clean.csv"

OUTPUT_FILE = "parfumo_text_profile.csv"


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

    print("\nJumlah data sebelum filter aroma kosong:")
    print(before_filter)

    print("\nJumlah data setelah filter aroma kosong:")
    print(after_filter)

    print("\nJumlah data yang dihapus karena tidak memiliki informasi aroma sama sekali:")
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

    df_selected.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print(f"\nBerhasil membuat file: {OUTPUT_FILE}")
    print("\nContoh data hasil preprocessing:")
    print(
        df_selected[
            [
                "perfume_id",
                "Name",
                "Brand",
                "Main_Accords",
                "Top_Notes",
                "Middle_Notes",
                "Base_Notes",
            ]
        ].head()
    )


if __name__ == "__main__":
    main()