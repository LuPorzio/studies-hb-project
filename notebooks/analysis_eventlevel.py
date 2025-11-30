import marimo

__generated_with = "0.18.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    from pathlib import Path

    sns.set(style="whitegrid")
    return Path, mo, pd, plt, sns


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Analysis – Event-level mood & phone use

    Obiettivi:
    - STEP 1: descrizione del mood, degli intervalli tra notifiche e dell'uso delle app
    - STEP 2: relazioni grezze tra uso app e mood
    - STEP 3: differenze per ora del giorno e gruppi
    - STEP 4: modelli (OLS e multilevel)
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #STEP 1
    """)
    return


@app.cell
def _(Path, pd):
    data_path = Path("./data/processed/eventlevel_mood_phoneuse.parquet")
    event_df = pd.read_parquet(data_path)

    event_df.shape, event_df.columns
    return (event_df,)


@app.cell
def _(event_df):
    # Info generale
    event_df.info()
    return


@app.cell
def _(event_df):
    vars_key = [
        "A6a", "mood_prev", "delta_mood",
        "delta_t_min", "delta_t_hours",
        "use_social", "use_communication", "use_other",
        "time_of_day", "gender", "cohort_group"
    ]

    missing = {v: event_df[v].isna().sum() for v in vars_key}
    missing
    return


@app.cell
def _(event_df):
    descr = event_df[[
        "A6a", "mood_prev", "delta_mood",
        "delta_t_min", "delta_t_hours",
        "use_social", "use_communication", "use_other"
    ]].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95]).T

    descr
    return


@app.cell
def _(event_df, plt, sns):
    plt.figure(figsize=(6,4))
    sns.histplot(event_df["A6a"], bins=20)
    plt.xlabel("Mood attuale (A6a)")
    plt.ylabel("Frequenza")
    plt.title("Distribuzione del mood negli eventi")
    plt.show()
    return


@app.cell
def _(event_df, plt, sns):
    plt.figure(figsize=(6,4))
    sns.histplot(event_df["delta_mood"].dropna(), bins=30)
    plt.axvline(0, color="black", linestyle="--")
    plt.xlabel("Delta mood (A6a - mood_prev)")
    plt.ylabel("Frequenza")
    plt.title("Distribuzione del cambiamento di mood")
    plt.show()
    return


@app.cell
def _():
    return


@app.cell
def _(event_df):
    # teniamo solo intervalli tra 30 e 180 minuti (0.5h – 3h)
    event_df_analysis = event_df.query("delta_t_min >= 30 and delta_t_min <= 180").copy()

    event_df.shape, event_df_analysis.shape
    return (event_df_analysis,)


@app.cell
def _(event_df_analysis, plt, sns):
    plt.figure(figsize=(6,4))
    sns.histplot(event_df_analysis["delta_t_min"], bins=[30, 60, 90, 120, 150, 180])
    plt.xlabel("Durata intervallo tra notifiche (minuti)")
    plt.ylabel("Frequenza")
    plt.title("Durata intervalli (30–180 min)")
    plt.show()
    return


@app.cell
def _(event_df):
    event_df["delta_t_min"].describe(percentiles=[0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
    return


@app.cell
def _(event_df):
    event_df["delta_t_min"].value_counts().sort_values(ascending=False).head(20)
    return


@app.cell
def _(event_df):
    event_df["use_social_min"] = event_df["use_social"] / 60
    event_df["use_communication_min"] = event_df["use_communication"] / 60
    event_df["use_other_min"] = event_df["use_other"] / 60

    event_df[[
        "use_social_min", 
        "use_communication_min", 
        "use_other_min"
    ]].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95]).T
    return


@app.cell
def _(event_df, plt, sns):
    vars_min = [
        ("use_social_min", "Uso SOCIAL (minuti)"),
        ("use_communication_min", "Uso COMMUNICATION (minuti)"),
        ("use_other_min", "Uso OTHER (minuti)")
    ]

    for var, label in vars_min:
        plt.figure(figsize=(6,4))
        sns.histplot(event_df[var], bins=40)
        plt.xlim(0, event_df[var].quantile(0.95))
        plt.xlabel(label)
        plt.ylabel("Frequenza")
        plt.title(f"Distribuzione {label.lower()}")
        plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #STEP 2
    """)
    return


@app.cell
def _(corr_vars, event_df_analysis):
    #MATRICE DI CORRELAZIONc
    orr_vars = [
        "A6a", "delta_mood",
        "use_social_min", "use_communication_min", "use_other_min",
        "delta_t_min"
    ]

    corr = event_df_analysis[corr_vars].corr()
    corr
    return (corr,)


@app.cell
def _(corr, plt, sns):
    plt.figure(figsize=(6,5))
    sns.heatmap(corr, annot=True, fmt=".2f", vmin=-1, vmax=1, cmap="coolwarm")
    plt.title("Matrice di correlazione (mood, delta_mood, uso app, intervallo)")
    plt.show()
    return


@app.cell
def _(event_df_analysis, plt, sns):
    plt.figure(figsize=(6,4))
    sns.regplot(
        data=event_df_analysis,
        x="use_social_min",
        y="A6a",
        scatter_kws={"alpha": 0.2}
    )
    plt.xlabel("Uso SOCIAL tra due notifiche (minuti)")
    plt.ylabel("Mood attuale (A6a)")
    plt.title("Mood vs uso SOCIAL tra notifiche")
    plt.show()
    return


@app.cell
def _(event_df_analysis, plt, sns):
    plt.figure(figsize=(6,4))
    sns.regplot(
        data=event_df_analysis,
        x="use_communication_min",
        y="A6a",
        scatter_kws={"alpha": 0.2}
    )
    plt.xlabel("Uso COMMUNICATION tra due notifiche (minuti)")
    plt.ylabel("Mood attuale (A6a)")
    plt.title("Mood vs uso COMMUNICATION tra notifiche")
    plt.show()
    return


@app.cell
def _(event_df_analysis, plt, sns):
    plt.figure(figsize=(6,4))
    sns.regplot(
        data=event_df_analysis,
        x="use_other_min",
        y="A6a",
        scatter_kws={"alpha": 0.2}
    )
    plt.xlabel("Uso OTHER tra due notifiche (minuti)")
    plt.ylabel("Mood attuale (A6a)")
    plt.title("Mood vs uso OTHER tra notifiche")
    plt.show()
    return


@app.cell
def _(event_df_analysis, pd, plt, sns):
    # Crea 3 fasce di uso social: low / medium / high
    bins = [0, 0.5, 2, event_df_analysis["use_social_min"].max()]
    labels = ["low", "medium", "high"]

    event_df_analysis["use_social_level"] = pd.cut(
        event_df_analysis["use_social_min"],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    plt.figure(figsize=(6,4))
    sns.boxplot(
        data=event_df_analysis,
        x="use_social_level",
        y="A6a",
        order=["low", "medium", "high"]
    )
    plt.xlabel("Livello di uso SOCIAL tra notifiche")
    plt.ylabel("Mood attuale (A6a)")
    plt.title("Mood per livello di uso SOCIAL")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Le analisi descrittive e le correlazioni semplici mostrano che l’uso del telefono tra due notifiche EMA è solo debolmente associato al mood riportato. Le correlazioni tra mood e uso delle app (social, communication, other) sono tutte inferiori a |0.05|, indicando assenza di una relazione lineare semplice. Anche dividendo l'uso social in livelli (low/medium/high), la distribuzione del mood rimane invariata.

    Questi risultati suggeriscono che eventuali relazioni tra uso dello smartphone e mood sono molto sottili, e richiedono modelli più avanzati (con controlli temporali e random effects) per essere identificate.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #STEP 3
    """)
    return


@app.cell
def _(event_df_analysis):
    event_df_analysis.groupby("time_of_day")["A6a"].mean()
    return


@app.cell
def _(event_df_analysis, plt, sns):
    plt.figure(figsize=(6,4))
    sns.boxplot(data=event_df_analysis, x="time_of_day", y="A6a")
    plt.xlabel("Fascia oraria")
    plt.ylabel("Mood attuale (A6a)")
    plt.title("Mood per fascia oraria")
    plt.show()
    return


@app.cell
def _(event_df_analysis):
    event_df_analysis.groupby("time_of_day")[
        ["use_social_min", "use_communication_min", "use_other_min"]
    ].mean()
    return


@app.cell
def _(event_df_analysis):
    use_means = (
        event_df_analysis
        .groupby("time_of_day")[["use_social_min", "use_communication_min", "use_other_min"]]
        .mean()
        .reset_index()
    )

    use_means
    return (use_means,)


@app.cell
def _(plt, sns, use_means):
    use_long = use_means.melt(
        id_vars="time_of_day",
        value_vars=["use_social_min", "use_communication_min", "use_other_min"],
        var_name="app_type",
        value_name="minutes"
    )

    plt.figure(figsize=(7,4))
    sns.barplot(data=use_long, x="time_of_day", y="minutes", hue="app_type")
    plt.xlabel("Fascia oraria")
    plt.ylabel("Minuti medi tra due notifiche")
    plt.title("Uso delle app per fascia oraria")
    plt.legend(title="Tipo di app")
    plt.show()
    return


@app.cell
def _(event_df_analysis, plt, sns):
    plt.figure(figsize=(5,4))
    sns.boxplot(data=event_df_analysis, x="gender", y="A6a")
    plt.xlabel("Genere")
    plt.ylabel("Mood attuale (A6a)")
    plt.title("Mood per genere")
    plt.show()
    return


@app.cell
def _(event_df_analysis, plt, sns):
    plt.figure(figsize=(5,4))
    sns.boxplot(data=event_df_analysis, x="gender", y="use_social_min")
    plt.xlabel("Genere")
    plt.ylabel("Uso SOCIAL tra notifiche (minuti)")
    plt.title("Uso social per genere")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #STEP 4
    """)
    return


@app.cell
def _():
    import statsmodels.formula.api as smf

    return (smf,)


@app.cell
def _(event_df_analysis):
    from sklearn.preprocessing import StandardScaler
    # Copia per sicurezza
    model_df = event_df_analysis.copy()

    # Se le colonne *_min non esistono, creiamole dai secondi
    if "use_social_min" not in model_df.columns:
        model_df["use_social_min"] = model_df["use_social"] / 60
        model_df["use_communication_min"] = model_df["use_communication"] / 60
        model_df["use_other_min"] = model_df["use_other"] / 60

    # Standardizza l'uso delle app (z-score)
    scaler = StandardScaler()
    model_df[["use_social_z", "use_communication_z", "use_other_z"]] = scaler.fit_transform(
        model_df[["use_social_min", "use_communication_min", "use_other_min"]]
    )

    # Togliamo righe con missing nelle variabili usate
    model_df = model_df.dropna(subset=[
        "A6a",
        "use_social_z", "use_communication_z", "use_other_z",
        "time_of_day",
        "gender",
        "userid"   # serve per il multilevel
    ])

    model_df.head()
    return (model_df,)


@app.cell
def _(model_df, smf):
    formula_ols = """
    A6a ~ use_social_z + use_communication_z + use_other_z
          + C(time_of_day) + C(gender)
    """

    ols_model = smf.ols(formula=formula_ols, data=model_df).fit()
    print(ols_model.summary())
    return


@app.cell
def _(model_df, smf):
    mixed_formula = """
    A6a ~ use_social_z + use_communication_z + use_other_z
          + C(time_of_day) + C(gender)
    """

    mixed_model = smf.mixedlm(
        mixed_formula,
        data=model_df,
        groups=model_df["userid"]
    ).fit(method="lbfgs")

    print(mixed_model.summary())
    return


@app.cell
def _(mo):
    mo.md(r"""
    #Collegare i risultati alla parte “over the weeks”
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Abbiamo stimato un modello di regressione lineare (e un modello multilevel con intercept casuale per utente) per predire il mood momento-per-momento (A6a) a partire dall’uso del telefono tra due notifiche EMA. I predittori principali erano l’uso delle app social, di comunicazione e di altre app (standardizzati), controllando per fascia oraria e genere.

    I risultati mostrano che l’uso delle app è associato al mood in modo estremamente debole. Nel modello OLS, un aumento di una deviazione standard nell’uso dei social è associato a una riduzione di circa 0.01 punti di mood (β = −0.013), l’uso di app di comunicazione a −0.006, mentre l’uso di altre app è associato a un aumento di circa 0.03 punti (β = +0.032) su una scala da 1 a 5. Questi effetti, pur statisticamente significativi a causa della grande numerosità del campione (N ≈ 69.000 eventi), hanno dimensioni di effetto trascurabili (R² ≈ 0.008).

    Il modello multilevel conferma il pattern: includendo un intercept casuale per utente, i coefficienti per le tre categorie di app restano molto piccoli e di segno analogo. Nel complesso, i risultati suggeriscono che non esiste una forte associazione lineare tra l’uso del telefono tra due notifiche e il livello di mood riportato, né a livello tra-persona né a livello entro-persona.
    """)
    return


@app.cell
def _(Path, event_df_analysis, pd, plt, sns):
    # --- Participation pattern: carica il riassunto del time diary e crea pattern per utente ---

    td_summary_path = Path("./data/processed/td_participation_summary.parquet")
    participation_summary = pd.read_parquet(td_summary_path)

    # (ri)calcola valid_user_period con le stesse soglie di eda_timedairy_participation.py
    def is_valid_user(row):
        if row["first2w"] == "First two weeks":
            return (row["mean_valid_per_day"] >= 30) and (row["days_with_valid"] >= 14)
        else:  # Second two weeks
            return (row["mean_valid_per_day"] >= 12) and (row["days_with_valid"] >= 5)

    participation_summary["valid_user_period"] = participation_summary.apply(is_valid_user, axis=1)

    # riassunto per utente: in quanti periodi è "valido"?
    user_patterns = (
        participation_summary
        .groupby("id", as_index=False)
        .agg(
            n_periods=("first2w", "nunique"),
            n_valid=("valid_user_period", "sum")
        )
    )

    def label_pattern(row):
        if row["n_valid"] >= 2:
            return "consistently_valid"    # valido in entrambi i periodi
        elif row["n_valid"] == 1:
            return "partially_valid"       # valido solo in uno dei due
        else:
            return "never_valid"           # mai valido

    user_patterns["participation_pattern"] = user_patterns.apply(label_pattern, axis=1)

    # rinomina id -> userid per il merge
    user_patterns = user_patterns.rename(columns={"id": "userid"})

    # --- merge del pattern nel dataset evento-level filtrato ---

    event_df_patterns = event_df_analysis.merge(
        user_patterns[["userid", "participation_pattern"]],
        on="userid",
        how="left"
    )

    print("Conteggio eventi per participation_pattern:")
    print(event_df_patterns["participation_pattern"].value_counts(dropna=False))

    # se non hai ancora le colonne *_min, creale ora
    if "use_social_min" not in event_df_patterns.columns:
        event_df_patterns["use_social_min"] = event_df_patterns["use_social"] / 60
        event_df_patterns["use_communication_min"] = event_df_patterns["use_communication"] / 60
        event_df_patterns["use_other_min"] = event_df_patterns["use_other"] / 60

    # --- mood medio per pattern di partecipazione ---

    print("\nMood medio (A6a) per participation_pattern:")
    print(
        event_df_patterns.groupby("participation_pattern")["A6a"].mean()
    )

    plt.figure(figsize=(6,4))
    sns.boxplot(
        data=event_df_patterns,
        x="participation_pattern",
        y="A6a",
        order=["consistently_valid", "partially_valid", "never_valid"]
    )
    plt.xlabel("Pattern di partecipazione al time diary")
    plt.ylabel("Mood attuale (A6a)")
    plt.title("Mood per pattern di partecipazione")
    plt.xticks(rotation=15)
    plt.show()

    # --- (opzionale) uso social per pattern di partecipazione ---

    print("\nUso social medio (minuti) per participation_pattern:")
    print(
        event_df_patterns.groupby("participation_pattern")["use_social_min"].mean()
    )

    plt.figure(figsize=(6,4))
    sns.boxplot(
        data=event_df_patterns,
        x="participation_pattern",
        y="use_social_min",
        order=["consistently_valid", "partially_valid", "never_valid"]
    )
    plt.xlabel("Pattern di partecipazione al time diary")
    plt.ylabel("Uso SOCIAL tra notifiche (minuti)")
    plt.title("Uso social per pattern di partecipazione")
    plt.xticks(rotation=15)
    plt.show()

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    “Over the weeks” (participation pattern → mood)

    Nessuna relazione forte.

    Il mood è praticamente lo stesso in tutti i gruppi.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #HYPOTHESIS 1
    """)
    return


@app.cell
def _(event_df_analysis, pd, plt, sns):
    # --- User-level: social share e mood medio ---

    # 1) Assicurati di avere le colonne in minuti
    if "use_social_min" not in event_df_analysis.columns:
        event_df_analysis["use_social_min"] = event_df_analysis["use_social"] / 60
        event_df_analysis["use_communication_min"] = event_df_analysis["use_communication"] / 60
        event_df_analysis["use_other_min"] = event_df_analysis["use_other"] / 60

    # totale uso app tra due notifiche
    event_df_analysis["total_use_min"] = (
        event_df_analysis["use_social_min"]
        + event_df_analysis["use_communication_min"]
        + event_df_analysis["use_other_min"]
    )

    # 2) Dataset a livello utente: medie per ciascun partecipante
    user_usage = (
        event_df_analysis
        .groupby("userid", as_index=False)
        .agg(
            mean_mood=("A6a", "mean"),
            mean_social_min=("use_social_min", "mean"),
            mean_comm_min=("use_communication_min", "mean"),
            mean_other_min=("use_other_min", "mean"),
            mean_total_use_min=("total_use_min", "mean")
        )
    )

    # quota di social sull'uso totale
    user_usage["social_share"] = (
        user_usage["mean_social_min"] / user_usage["mean_total_use_min"]
    )

    # togliamo utenti con total_use_min = 0 (social_share NaN)
    user_usage = user_usage.dropna(subset=["social_share"])

    print("Prime righe di user_usage:")
    print(user_usage.head())

    # 3) Correlazione tra quota social e mood medio
    print("\nCorrelazione mean_mood ~ social_share:")
    print(user_usage[["mean_mood", "social_share"]].corr())

    plt.figure(figsize=(6,4))
    sns.regplot(
        data=user_usage,
        x="social_share",
        y="mean_mood",
        scatter_kws={"alpha": 0.5}
    )
    plt.xlabel("Quota di social sull'uso totale (media utente)")
    plt.ylabel("Mood medio (A6a)")
    plt.title("Mood medio vs quota di social nella routine (livello utente)")
    plt.show()

    # 4) Gruppi di utenti per quota social (terzili)
    user_usage["social_share_group"] = pd.qcut(
        user_usage["social_share"],
        q=3,
        labels=["low_social", "medium_social", "high_social"]
    )

    print("\nNumero di utenti per gruppo di social_share:")
    print(user_usage["social_share_group"].value_counts())

    print("\nMood medio per gruppo di social_share:")
    print(user_usage.groupby("social_share_group")["mean_mood"].mean())

    plt.figure(figsize=(6,4))
    sns.boxplot(
        data=user_usage,
        x="social_share_group",
        y="mean_mood",
        order=["low_social", "medium_social", "high_social"]
    )
    plt.xlabel("Quota di social nella routine (gruppi di utenti)")
    plt.ylabel("Mood medio (A6a)")
    plt.title("Mood medio per livello di 'social-dominance' nella routine")
    plt.show()

    return


@app.cell
def _(mo):
    mo.md(r"""
    Analizzando i dati aggregati a livello utente, la quota di uso social sulla routine quotidiana non risulta associata al mood medio. La correlazione tra social-share e mood è praticamente nulla (r = −0.01).
    Suddividendo gli studenti in terzili di social-dominance (low / medium / high), le distribuzioni del mood medio risultano sovrapponibili. Questi risultati indicano che, nel nostro campione, gli studenti con routine più “social-dominated” non riportano un livello di benessere inferiore rispetto ai loro pari.
    """)
    return


if __name__ == "__main__":
    app.run()
