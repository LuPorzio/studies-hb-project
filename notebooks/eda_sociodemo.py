import marimo

__generated_with = "0.18.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import altair as alt
    from pathlib import Path
    return Path, alt, mo, pd


@app.cell
def _(mo):
    mo.md(r"""
    # EDA – socio_demo_IT.dta

    Obiettivo:
    - capire struttura e qualità del file socio-demografico
    - controllare missing / codifiche
    - fare pulizia minima
    - esportare un file pulito in `data/processed/`
    """)
    return


@app.cell
def _(Path, pd):
    # --- 1. Load data ---
    data_path = Path("../data/socio_demo_IT.dta")
    socio_df = pd.read_stata(data_path)

    socio_df.head()
    return data_path, socio_df


@app.cell
def _(socio_df):
    # --- 2. Basic structure ---
    socio_df.shape, socio_df.columns
    return


@app.cell
def _(socio_df):
    socio_df.info()
    return


@app.cell
def _(socio_df):
    #CONTROLLO
    socio_df["w1_A01"].value_counts(dropna=False)
    return


@app.cell
def _(socio_df):
    #controllo
    socio_df["degree"].value_counts(dropna=False)
    return


@app.cell
def _(socio_df):
    #controllo 
    socio_df["cohort"].value_counts(dropna=False)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Missing values
    Controlliamo quante informazioni mancano e in quali colonne.
    """)
    return


@app.cell
def _(pd, socio_df):
    missing_counts = socio_df.isna().sum().sort_values(ascending=False)
    missing_pct = (missing_counts / len(socio_df) * 100).round(2)

    missing_df = pd.DataFrame(
        {"missing_n": missing_counts, "missing_pct": missing_pct}
    )
    missing_df
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Duplicati e chiavi
    Verifichiamo se `userid` è un identificatore unico.
    """)
    return


@app.cell
def _(socio_df):
    # duplicates of userid
    socio_df["userid"].duplicated().sum()
    return


@app.cell
def _(socio_df):
    # show any duplicated userids if present
    socio_df.loc[socio_df["userid"].duplicated(keep=False), ["userid", "token"]].sort_values("userid")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Pulizia minima
    - standardizziamo i nomi colonne in snake_case
    - cast di alcune variabili
    - creiamo versioni più pulite di categorie
    """)
    return


@app.cell
def _(socio_df):
    import re

    def to_snake(s):
        s = s.strip()
        s = re.sub(r"\s+", "_", s)
        s = re.sub(r"[^\w_]", "", s)
        return s.lower()

    socio_df_clean = socio_df.copy()
    socio_df_clean.columns = [to_snake(c) for c in socio_df_clean.columns]

    socio_df_clean.head()
    return (socio_df_clean,)


@app.cell
def _(socio_df_clean):
    # --- 3. Basic casting ---
    # userid as int
    socio_df_clean["userid"] = socio_df_clean["userid"].astype(int)

    # gender appears in w1_a01 already as Male/Female
    if "w1_a01" in socio_df_clean.columns:
        socio_df_clean["gender"] = socio_df_clean["w1_a01"].astype("category")
    # Nota: manteniamo sia 'w1_a01' (variabile originale) sia 'gender' (versione pulita).
    # Useremo solo 'gender' nei merge e nei modelli finali; 'w1_a01' resta per coerenza e tracciabilità.

    # cohort / degree etc as category if present
    for col in ["cohort", "degree", "pilot", "department", "nationality"]:
        if col in socio_df_clean.columns:
            socio_df_clean[col] = socio_df_clean[col].astype("category")

    socio_df_clean.dtypes
    return


@app.cell
def _(socio_df_clean):
    # --- cohort_group (3 fasce età) ---
    def recode_cohort(x):
        x = str(x)
        if x in ["17-18", "19", "20"]:
            return "young"
        elif x in ["21", "22", "23", "24"]:
            return "middle"
        else:
            return "older"

    socio_df_clean["cohort_group"] = (
        socio_df_clean["cohort"].astype(str).apply(recode_cohort).astype("category")
    )

    socio_df_clean["cohort_group"].value_counts(dropna=False)
    return


@app.cell
def _(socio_df_clean):
    socio_df_clean.dtypes[["userid","gender","degree","cohort","cohort_group","department"]]
    return


@app.cell
def _(socio_df_clean):
    socio_df_clean[["userid","gender","degree","cohort","cohort_group"]].head()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Distribuzioni delle variabili socio-demografiche principali
    (Genere, nazionalità, dipartimento, degree, cohort)
    """)
    return


@app.cell
def _(socio_df_clean):
    socio_df_clean["gender"].value_counts(dropna=False) if "gender" in socio_df_clean.columns else None
    return


@app.cell
def _(socio_df_clean):
    socio_df_clean["nationality"].value_counts(dropna=False).head(20) if "nationality" in socio_df_clean.columns else None
    return


@app.cell
def _(socio_df_clean):
    socio_df_clean["department"].value_counts(dropna=False).head(20) if "department" in socio_df_clean.columns else None
    return


@app.cell
def _(socio_df_clean):
    socio_df_clean["degree"].value_counts(dropna=False) if "degree" in socio_df_clean.columns else None
    return


@app.cell
def _(socio_df_clean):
    socio_df_clean["cohort"].value_counts(dropna=False).sort_index() if "cohort" in socio_df_clean.columns else None
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Variabili psicometriche (Big5, valori M*, intelligenze multiple, ecc.)
    Calcoliamo statistiche descrittive e controlliamo range/outlier.
    """)
    return


@app.cell
def _(socio_df_clean):
    # numeric columns
    num_cols = socio_df_clean.select_dtypes(include="number").columns.tolist()
    num_cols
    return (num_cols,)


@app.cell
def _(num_cols, socio_df_clean):
    socio_df_clean[num_cols].describe().T
    return


@app.cell
def _(mo):
    mo.md(r"""
    Se vogliamo, possiamo fare qualche istogramma rapido delle scale principali.
    (Scegliamo poche colonne per non fare 100 grafici.)
    """)
    return


@app.cell
def _(alt, socio_df_clean):

    # choose some key scales if present
    key_scales = [c for c in ["extraversion", "agreeableness", "conscientiousness",
                                "neuroticism", "openness"] if c in socio_df_clean.columns]

    charts = []
    for c in key_scales:
        chart = (
            alt.Chart(socio_df_clean)
            .mark_bar()
            .encode(
                x=alt.X(c, bin=alt.Bin(maxbins=12), title=c),
                y=alt.Y("count()", title="n")
            )
            .properties(height=220, width="container", title=f"Distribuzione {c}")
        )
        charts.append(chart)

    alt.vconcat(*charts) if charts else None
    
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Output pulito
    Salviamo una versione pulita del socio-demo per il merge futuro.
    """)
    return


@app.cell
def _(data_path, socio_df_clean):
    processed_path = data_path.parent / "processed"
    processed_path.mkdir(exist_ok=True)

    out_file = processed_path / "socio_demo_cleaned.parquet"
    socio_df_clean.to_parquet(out_file, index=False)

    out_file
    return


if __name__ == "__main__":
    app.run()
