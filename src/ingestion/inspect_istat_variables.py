from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

CENSUS_FILE = (
    ROOT
    / "data"
    / "raw"
    / "istat"
    / "censimento_2023"
    / "Dati_regionali_2023"
    / "R17_Basilicata_2023_sezioni.xlsx"
)

TRACE_FILE = (
    ROOT
    / "data"
    / "raw"
    / "istat"
    / "censimento_2023"
    / "Dati_regionali_2023"
    / "TRACCIATO FILE REGIONALI.xlsx"
)


def inspect_matera():
    df = pd.read_excel(CENSUS_FILE)

    matera = df[
        df["COMUNE"]
        .astype(str)
        .str.strip()
        .str.casefold()
        == "matera"
    ].copy()

    print("\n=== MATERA ===")
    print(f"Sezioni trovate: {len(matera)}")

    print("PROCOM:")
    print(matera["PROCOM"].unique())

    print("SEZ21_ID univoci:")
    print(matera["SEZ21_ID"].nunique())

    print("\nPrime 5 sezioni:")
    print(
        matera[
            [
                "CODREG",
                "CODPRO",
                "CODCOM",
                "COMUNE",
                "PROCOM",
                "SEZ21_ID",
            ]
        ]
        .head()
        .to_string(index=False)
    )


def inspect_dictionary():
    trace = pd.read_excel(TRACE_FILE)

    # Salviamo anche il dizionario completo:
    output = ROOT / "docs" / "istat_variables_2023.csv"
    output.parent.mkdir(parents=True, exist_ok=True)

    trace.to_csv(
        output,
        index=False,
        encoding="utf-8"
    )

    keywords = [
        "popolazione",
        "età",
        "anni",
        "famigli",
        "stranier",
        "occupat",
        "disoccup",
        "abitaz",
        "automobil",
        "istruzione",
    ]

    pattern = "|".join(keywords)

    relevant = trace[
        trace["DEFINIZIONE"]
        .astype(str)
        .str.contains(
            pattern,
            case=False,
            regex=True,
            na=False
        )
    ]

    print("\n=== VARIABILI POTENZIALMENTE UTILI ===")

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_colwidth", None)

    print(
        relevant[
            ["NOME_CAMPO", "DEFINIZIONE"]
        ].to_string(index=False)
    )

    print(
        f"\nDizionario completo salvato in: "
        f"{output.relative_to(ROOT)}"
    )


if __name__ == "__main__":
    inspect_matera()
    inspect_dictionary()