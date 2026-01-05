import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import altair as alt
    from pathlib import Path
    return Path, mo, pd, plt, sns


@app.cell
def _(mo):
    mo.md(r"""
    # EDA - App Use Participation
    """)
    return


@app.cell
def _(Path, pd):
    data_path_app_use = Path("../data/appuseIT_class_17_11.parquet")
    app_use = pd.read_parquet(data_path_app_use)
    app_use.head()
    return (app_use,)


@app.cell
def _(app_use):
    app_use.info()
    return


@app.cell
def _(mo):
    mo.md(r"""
    # EDA – Time Diary Participation (STEP 2)

    Obiettivi:
    - convertire timestamp e filtrare ore notturne non informative
    - calcolare **notifiche per giorno** per utente
    - calcolare **giorni risposti** per utente
    - creare summary di partecipazione
    - salvare file puliti in `data/processed/`
    """)
    return


@app.cell
def _(Path, pd):
    data_path = Path("../data/td_ita.csv")
    td_df = pd.read_csv(data_path, low_memory=False)
    td_df.head()
    return data_path, td_df


@app.cell
def _(td_df):
    td_df.info()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Conversione timestamp
    """)
    return


@app.cell
def _(pd, td_df):
    td_df["date_not"] = pd.to_datetime(td_df["date_not"])
    td_df["datein_ques"] = pd.to_datetime(td_df["datein_ques"])
    td_df["datein_answ"] = pd.to_datetime(td_df["datein_answ"])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Filtro ore non informative (1–5 AM)
    """)
    return


@app.cell
def _(td_df):
    # teniamo solo tra 05:00 e 01:00 (come nel tuo eda.py)
    td_df_cleaned = (
        td_df.set_index("date_not")
        .between_time(start_time="5:00am", end_time="1:00am")
        .reset_index()
    )
    td_df_cleaned.head()
    return (td_df_cleaned,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Giorno e risposta valida
    """)
    return


@app.cell
def _(td_df_cleaned):
    # giorno (senza ora)
    td_df_cleaned["day"] = td_df_cleaned["date_not"].dt.date

    # risposta valida = non "No information"
    td_df_cleaned["is_valid"] = td_df_cleaned["what"] != "No information"

    td_df_cleaned[["id", "day", "what", "is_valid"]].head()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Notifiche per giorno per utente

    Calcoliamo:
    - notifiche totali al giorno
    - notifiche valide al giorno (quindi risposte informative)
    """)
    return


@app.cell
def _(td_df_cleaned):
    daylevel_df = (
        td_df_cleaned.groupby(["id", "day", "first2w"], as_index=False)
        .agg(
            notifications_total=("what", "size"),
            notifications_valid=("is_valid", "sum"),
        )
    )

    daylevel_df.head(200000)
    return (daylevel_df,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Giorni risposti per utente + summary partecipazione
    """)
    return


@app.cell
def _(daylevel_df, pd):
    # CHANGED ON 05/01
    # 1. Define validity thresholds for individual periods
    # (Using the thresholds from your original script logic)
    def check_period_validity(group):
        is_f2w = (group["first2w"] == "First two weeks").any()
        mean_val = group["notifications_valid"].median()
        days_val = (group["notifications_valid"] > 0).sum()

        if is_f2w:
            return (mean_val >= 25) and (days_val >= 14)
        else:
            return (mean_val >= 12) and (days_val >= 7)


    # 2. Identify users valid in BOTH periods
    validity_per_period = (
        daylevel_df.groupby(["id", "first2w"])
        .apply(check_period_validity)
        .unstack(fill_value=False)
    )

    # Users who are True for both "First two weeks" and "Second two weeks"
    persistent_users = validity_per_period[
        (validity_per_period["First two weeks"] == True) & 
        (validity_per_period["Second two weeks"] == True)
    ].index.tolist()

    # 3. Create the "First 2 Weeks" slice (All valid users for that period)
    f2w_valid_users = validity_per_period[validity_per_period["First two weeks"] == True].index.tolist()
    f2w_data = daylevel_df[
        (daylevel_df["id"].isin(f2w_valid_users)) & 
        (daylevel_df["first2w"] == "First two weeks")
    ].copy()
    f2w_data["first2w"] = "First 2 Weeks"

    # 4. Create the "Whole Period" slice (ONLY users active in both)
    whole_data = daylevel_df[daylevel_df["id"].isin(persistent_users)].copy()
    whole_data["first2w"] = "Whole Period"

    # 5. Combine and aggregate
    combined_data = pd.concat([f2w_data, whole_data])

    participation_summary = (
        combined_data.groupby(["id", "first2w"], as_index=False)
        .agg(
            median_valid_per_day=("notifications_valid", "median"),
            days_with_valid=("notifications_valid", lambda x: (x > 0).sum()),
            total_valid=("notifications_valid", "sum"),
        )
    )
    participation_summary.head(20000000)

    return (participation_summary,)


@app.cell
def _(participation_summary, participation_summary2):
    """Classifichiamo i partecipanti in:
    - low contribution: sotto il 50° percentile di total_valid nel periodo
    - average: tra 50° e 75° percentile
    - outstanding: sopra il 75° percentile

    La classificazione è separata per:
    - First two weeks
    - Whole Period"""

    # descrittive per total_valid per ogni periodo
    desc = (
        participation_summary
        .groupby("first2w")["total_valid"]
        .describe(percentiles=[0.5, 0.75])
    )

    fw50p = desc.loc["First 2 Weeks", "50%"]
    fw75p = desc.loc["First 2 Weeks", "75%"]

    sw50p = desc.loc["Whole Period", "50%"]
    sw75p = desc.loc["Whole Period", "75%"]

    # funzione di classificazione per riga
    def classify(row):
        if row["first2w"] == "First 2 Weeks":
            if row["total_valid"] < fw50p:
                return "low"
            elif row["total_valid"] < fw75p:
                return "average"
            else:
                return "outstanding"
        else:  # Second two weeks
            if row["total_valid"] < sw50p:
                return "low"
            elif row["total_valid"] < sw75p:
                return "average"
            else:
                return "outstanding"

    _participation_summary2 = participation_summary.copy()
    _participation_summary2["contribution_level"] = participation_summary2.apply(
        classify, axis=1
    )

    _participation_summary2[["id", "first2w", "total_valid", "contribution_level"]].head(20)
    return


@app.cell
def _(participation_summary):
    def classify_fixed(row):
        # MOTIVATION: 
        # Low (< 28): Fewer than 2 valid responses per day on average.
        # Average (28 - 70): Between 2 and 5 responses per day.
        # Outstanding (> 70): More than 5 responses per day (High Density).
        if row["total_valid"] < 28:
            return "low"
        elif 28 <= row["total_valid"] <= 70:
            return "average"
        else:
            return "outstanding"

    participation_summary2 = participation_summary.copy()
    participation_summary["contribution_level"] = participation_summary.apply(classify_fixed, axis=1)

    return (participation_summary2,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Distribuzioni utili (per decidere soglie)

    soglie basate su:
    - notifiche valide per giorno
    - giorni risposti

    Guardiamo le distribuzioni per prime e seconde 2 settimane.
    """)
    return


@app.cell
def _(participation_summary, sns):
    _g = sns.FacetGrid(participation_summary, col="first2w", height=3, aspect=1.3)
    _g.map(sns.histplot, "median_valid_per_day")
    return


@app.cell
def _(participation_summary, sns):
    _g2 = sns.FacetGrid(participation_summary, col="first2w", height=3, aspect=1.3)
    _g2.map(sns.histplot, "days_with_valid")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Statistiche per soglie (quantili)
    """)
    return


@app.cell
def _(participation_summary):
    participation_summary["valid_user_period"].value_counts()
    return


@app.cell
def _(valid_overall):
    valid_overall["valid_user_overall"].value_counts()
    return


@app.cell
def _(participation_summary):
    participation_summary.groupby("first2w")[["median_valid_per_day", "days_with_valid"]].describe().T
    return


@app.cell
def _(participation_summary):
    # --- soglie "utente valido" per periodo ---
    # First two weeks: >=30 risposte valide/giorno e >=14 giorni risposti
    # Second two weeks: >=12 risposte valide/giorno e >=5 giorni risposti

    def is_valid_user(row):
        if row["first2w"] == "First two weeks":
            return (row["median_valid_per_day"] >= 30) and (row["days_with_valid"] >= 14)
        else:  # Second two weeks
            return (row["median_valid_per_day"] >= 12) and (row["days_with_valid"] >= 5)

    participation_summary["valid_user_period"] = participation_summary.apply(is_valid_user, axis=1)

    p_summary = participation_summary["valid_user_period"].value_counts(dropna=False)
    p_summary
    return


@app.cell
def _(participation_summary):
    _plot_df = participation_summary.groupby(["first2w", "valid_user_period"]).agg(count = ("total_valid", "size")).reset_index()
    _plot_df.head()
    return


@app.cell
def _(participation_summary, plt, sns):
    _plot_df = participation_summary.groupby(["first2w", "valid_user_period"]).agg(count = ("total_valid", "size")).reset_index()
    _g = sns.barplot(x = "first2w", y = "count", hue = "valid_user_period", data = _plot_df)
    plt.legend(loc = "upper left")
    plt.savefig("../plots/participation.png")
    _g
    return


@app.cell
def _(participation_summary):
    # --- validità complessiva (utente valido almeno in un periodo) ---
    valid_overall = (
        participation_summary
        .groupby("id")["valid_user_period"]
        .any()
        .reset_index(name="valid_user_overall")
    )

    # NON ridefiniamo participation_summary: creiamo una nuova tabella
    participation_summary_overall = participation_summary.merge(
        valid_overall, on="id", how="left"
    )

    participation_summary_overall[["id", "first2w", "valid_user_period", "valid_user_overall"]].head(20)
    return participation_summary_overall, valid_overall


@app.cell
def _():
    ## STEP 3 – Pattern di partecipazione nel tempo

    #Obiettivo:
    #- distinguere tipi di partecipanti (costanti, dropout, late joiner, low participation)
    #- usare queste etichette come informazione descrittiva e/o per i modelli.
    return


@app.cell
def _(participation_summary_overall):
    valid_wide = (
        participation_summary_overall
        .pivot(index="id", columns="first2w", values="valid_user_period")
    )

    valid_wide.head()
    return (valid_wide,)


@app.cell
def _(participation_summary):
    # CHANGED ON 05/01
    def _(participation_summary, pd):
        # Identifying the patterns based on our new analysis_period column
        # We want to see who appears in 'Whole Period' (which means they were in both)
        # vs who only appears in 'First 2 Weeks'

        user_counts = participation_summary.groupby("id")["first2w"].nunique()

        def label_pattern(user_id):
            # If they appear in 2 periods (First 2 Weeks AND Whole Period), 
            # it means they passed the 'Persistent' check.
            if user_counts[user_id] == 2:
                return "Consistent (Both Periods)"
            else:
                return "Early Phase Only"

        participation_summary["participation_pattern"] = participation_summary["id"].apply(label_pattern)

    participation_summary.head(200000)
    return


@app.cell
def _(participation_summary_overall, valid_wide):
    valid_wide_reset = valid_wide["participation_pattern"].reset_index()

    participation_summary_patterns = participation_summary_overall.merge(
        valid_wide_reset, on="id", how="left"
    )

    participation_summary_patterns[
        ["id", "first2w", "valid_user_period", "valid_user_overall", "participation_pattern"]
    ].head()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Salvataggio output puliti
    """)
    return


@app.cell
def _(data_path, daylevel_df, participation_summary):
    processed_path = data_path.parent / "processed"
    processed_path.mkdir(exist_ok=True)

    out_daylevel = processed_path / "td_participation_daylevel.parquet"
    out_summary = processed_path / "td_participation_summary.parquet"

    daylevel_df.to_parquet(out_daylevel, index=False)
    participation_summary.to_parquet(out_summary, index=False)

    out_daylevel, out_summary
    return


if __name__ == "__main__":
    app.run()
