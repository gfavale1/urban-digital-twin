from pathlib import Path

import geopandas as gpd
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
ISTAT_RAW = ROOT / "data" / "raw" / "istat"


def inspect_shapefiles():
    print("\n=== SHAPEFILES ===")

    shapefiles = list(ISTAT_RAW.rglob("*.shp"))

    for path in shapefiles:
        print(f"\nFile: {path.relative_to(ROOT)}")

        gdf = gpd.read_file(path)

        print(f"Righe: {len(gdf)}")
        print(f"CRS: {gdf.crs}")
        print("Colonne:")
        print(gdf.columns.tolist())

        print("\nPrime righe:")
        print(gdf.head(3).drop(columns="geometry").to_string())


def read_istat_csv(path):
    """
    I file CSV ISTAT delle basi territoriali possono essere UTF-16.
    Proviamo gli encoding più comuni senza interrompere l'ispezione.
    """

    attempts = [
        ("utf-8-sig", None),
        ("utf-16", None),
        ("utf-16-le", None),
        ("cp1252", None),
    ]

    last_error = None

    for encoding, sep in attempts:
        try:
            return pd.read_csv(
                path,
                sep=sep,
                engine="python",
                encoding=encoding
            )
        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        f"Impossibile leggere {path}: {last_error}"
    )


def inspect_csv():
    print("\n=== CSV ===")

    csv_files = list(ISTAT_RAW.rglob("*.csv"))

    for path in csv_files:
        print(f"\nFile: {path.relative_to(ROOT)}")

        try:
            df = read_istat_csv(path)

            print(f"Righe: {len(df)}")
            print(f"Colonne: {len(df.columns)}")

            print("Nomi colonne:")
            print(df.columns.tolist())

            print("\nPrime righe:")
            print(df.head(3).to_string())

        except Exception as exc:
            print(f"ERRORE LETTURA: {exc}")


def inspect_excel():
    print("\n=== EXCEL ===")

    excel_files = list(ISTAT_RAW.rglob("*.xlsx"))

    for path in excel_files:
        # Per ora ci interessano soprattutto Basilicata e tracciato.
        if "R17_Basilicata" not in path.name and "TRACCIATO" not in path.name:
            continue

        print(f"\nFile: {path.relative_to(ROOT)}")

        workbook = pd.ExcelFile(path)

        print("Fogli:")
        print(workbook.sheet_names)

        for sheet in workbook.sheet_names:
            print(f"\n--- Foglio: {sheet} ---")

            df = pd.read_excel(
                path,
                sheet_name=sheet,
                nrows=5
            )

            print(f"Colonne: {len(df.columns)}")
            print("Nomi colonne:")
            print(df.columns.tolist())

            print("\nPrime righe:")
            print(df.head(3).to_string())


if __name__ == "__main__":
    inspect_shapefiles()
    inspect_csv()
    inspect_excel()