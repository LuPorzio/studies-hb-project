import marimo

__generated_with = "0.19.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import altair as alt
    from pathlib import Path
    import matplotlib.pyplot as plt
    import seaborn as sns
    return Path, alt, mo, pd, plt, sns


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
def _(mo):
    mo.md(r"""
 
    """)
    return


@app.cell
def _(Path, pd):
    # --- 1. Load data ---
    data_path = Path("../data/socio_demo_IT.dta")
    socio_df = pd.read_stata(data_path)

    socio_df.head(2999999)
    return data_path, socio_df


@app.cell
def _(socio_df):
    len(socio_df["userid"].unique())
    return


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
    socio_df_clean["nationality"].value_counts(dropna=False).head(20) if "nationality" in socio_df_clean.columns else None
    return


@app.cell
def _(socio_df_clean):
    socio_df_clean["department"].value_counts(dropna=False).head(20) if "department" in socio_df_clean.columns else None
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
def _(alt, socio_df_clean_reduced):

    # choose some key scales if present
    key_scales = [c for c in ["extraversion", "agreeableness", "conscientiousness",
                                "neuroticism", "openness"] if c in socio_df_clean_reduced.columns]

    charts = []
    for c in key_scales:
        chart = (
            alt.Chart(socio_df_clean_reduced)
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
    # Socio-Demographic plots on the full aggregated dataset (event_df) with only our selected sample
    """)
    return


@app.cell
def _(socio_df_clean):
    ids_filtered_sample = [
        0,  1,  3,  4,   5,   6,   8,   9,  12,  13,  15,  18,  19,
            20,  24,  26,  28,  30,  32,  33,  34,  41,  44,  45,  48,  52,
            55,  57,  58,  59,  60,  61,  65,  66,  70,  73,  75,  76,  79,
            80,  82,  83,  87,  89,  91,  92,  97,  98,  99, 100, 105, 106,
           107, 109, 111, 112, 113, 114, 118, 119, 124, 126, 128, 130, 131,
           132, 134, 136, 141, 144, 146, 148, 151, 153, 155, 158, 160, 161,
           162, 163, 165, 166, 167, 169, 173, 176, 177, 182, 185, 188, 191,
           194, 195, 196, 197, 198, 199, 200, 202, 203, 204, 206, 208, 209,
           210, 212, 215, 216, 223, 224, 225, 229, 233, 239, 243, 245, 250,
           251, 252, 253, 254, 255, 256, 258, 259, 262
           ]

    socio_df_clean_reduced = socio_df_clean[socio_df_clean["userid"].isin(ids_filtered_sample)]
    return (socio_df_clean_reduced,)


@app.cell
def _(mo):
    mo.md(r"""
    # Gender Distribution by Department: Check if certain departments have gender imbalances
    """)
    return


@app.cell
def _(plt, sns, socio_df_clean_reduced):
    def _(socio_df_clean_reduced, plt, sns):
        # Gender distribution across top departments
        top_depts = socio_df_clean_reduced["department"].value_counts().head(10).index

        plt.figure(figsize=(12, 6))
        gender_dept = socio_df_clean_reduced[socio_df_clean_reduced["department"].isin(top_depts)]

        sns.countplot(data=gender_dept, y="department", hue="gender", 
                      order=top_depts)
        plt.title("Gender Distribution Across Top 10 Departments")
        plt.xlabel("Count")
        plt.tight_layout()
        plt.show()
        return

    _(socio_df_clean_reduced, plt, sns)
    return


@app.cell
def _():
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Cohort Distribution by Degree Type: Understand age structure across degree programs
    """)
    return


@app.cell
def _(pd, plt, sns, socio_df_clean_reduced):
    def _(socio_df_clean_reduced, plt, sns):
        plt.figure(figsize=(10, 6))

        # Create crosstab for heatmap
        cohort_degree = pd.crosstab(
            socio_df_clean_reduced["cohort_group"], 
            socio_df_clean_reduced["degree"],
            normalize="columns"
        ) * 100

        sns.heatmap(cohort_degree, annot=True, fmt=".1f", cmap="YlOrRd")
        plt.title("Cohort Group Distribution by Degree Type (%)")
        plt.ylabel("Cohort Group")
        plt.xlabel("Degree Type")
        plt.tight_layout()
        plt.show()
        return

    _(socio_df_clean_reduced, plt, sns)

    return


@app.cell
def _(mo):
    mo.md(r"""
    # Big Five Personality Correlations: Explore relationships between personality dimensions
    """)
    return


@app.cell
def _(plt, sns, socio_df_clean_reduced):

    def _(socio_df_clean_reduced, plt, sns):
        big5_cols = ["extraversion", "agreeableness", "conscientiousness", 
                     "neuroticism", "openness"]

        # Check which columns exist
        available_big5 = [c for c in big5_cols if c in socio_df_clean_reduced.columns]

        if len(available_big5) >= 2:
            corr_matrix = socio_df_clean_reduced[available_big5].corr()

            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, fmt=".2f", 
                        cmap="coolwarm", center=0, vmin=-1, vmax=1,
                        square=True)
            plt.title("Big Five Personality Dimensions - Correlation Matrix")
            plt.tight_layout()
            plt.show()
        return

    _(socio_df_clean_reduced, plt, sns)
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Personality Profiles by Gender: Compare personality dimensions across genders
    """)
    return


@app.cell
def _(pd, plt, sns, socio_df_clean_reduced):
    def _(socio_df_clean_reduced, plt, sns, pd):
        big5_cols = ["extraversion", "agreeableness", "conscientiousness", 
                     "neuroticism", "openness"]
        available_big5 = [c for c in big5_cols if c in socio_df_clean_reduced.columns]

        if len(available_big5) >= 2 and "gender" in socio_df_clean_reduced.columns:
            # Melt data for grouped plot
            plot_df = socio_df_clean_reduced[available_big5 + ["gender"]].melt(
                id_vars="gender",
                var_name="Trait",
                value_name="Score"
            )

            plt.figure(figsize=(12, 6))
            sns.violinplot(data=plot_df, x="Trait", y="Score", hue="gender", split=True)
            plt.title("Big Five Personality Profiles by Gender")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
        return

    _(socio_df_clean_reduced, plt, sns, pd)
    return


@app.cell
def _(plt, socio_df_clean_reduced):
    def plot_gender_distribution(socio_df_clean_reduced):
        plt.figure(figsize=(8, 8))
        gender_counts = socio_df_clean_reduced['gender'].value_counts()
    
        plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', 
                colors=['#ff9999','#66b3ff'], startangle=140, explode=(0.05, 0))
    
        plt.title("Overall Gender Distribution", fontsize=14, fontweight='bold')
        plt.show()

    plot_gender_distribution(socio_df_clean_reduced)
    return


@app.cell
def _(pd, plt, socio_df_clean_reduced):
    def plot_gender_by_degree(socio_df_clean_reduced):
        # Create a cross-tabulation of gender and degree
        gender_degree = pd.crosstab(socio_df_clean_reduced['degree'], socio_df_clean_reduced['gender'], normalize='index') * 100
    
        gender_degree.plot(kind='bar', stacked=True, figsize=(12, 6), color=['#ff9999','#66b3ff'])
    
        plt.title("Gender Composition by Degree Type (%)", fontsize=14, fontweight='bold')
        plt.ylabel("Percentage (%)")
        plt.xlabel("Degree Type")
        plt.legend(title="Gender", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    plot_gender_by_degree(socio_df_clean_reduced)
    return


@app.cell
def _(plt, sns, socio_df_clean_reduced):
    def plot_top_nationalities(socio_df_clean_reduced):
        if 'nationality' in socio_df_clean_reduced.columns:
            plt.figure(figsize=(10, 6))
            top_10_nat = socio_df_clean_reduced['nationality'].value_counts().head(10)
        
            sns.barplot(x=top_10_nat.values, y=top_10_nat.index, palette="viridis")
        
            plt.title("Top 10 Nationalities in Sample", fontsize=14, fontweight='bold')
            plt.xlabel("Number of Students")
            plt.ylabel("Nationality")
            plt.grid(axis='x', linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.show()

    plot_top_nationalities(socio_df_clean_reduced)
    return


@app.cell
def _(plt, sns, socio_df_clean_reduced):
    def plot_department_volume(socio_df_clean_reduced):
        if 'department' in socio_df_clean_reduced.columns:
            plt.figure(figsize=(14, 7))
            dept_counts = socio_df_clean_reduced['department'].value_counts().head(15) # Top 15 for readability
        
            sns.barplot(x=dept_counts.index, y=dept_counts.values, palette="magma")
        
            plt.title("Top 15 Departments by Participation", fontsize=14, fontweight='bold')
            plt.xlabel("Department Name")
            plt.ylabel("Number of Participants")
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.show()

    plot_department_volume(socio_df_clean_reduced)
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
