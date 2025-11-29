import marimo

__generated_with = "0.18.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import altair as alt
    from pathlib import Path
    return Path, mo, pd, sns


@app.cell(hide_code=True)
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
    data_path = Path("./data/td_ita.csv")
    td_df = pd.read_csv(data_path, low_memory=False)
    td_df.head()
    return data_path, td_df


@app.cell
def _(td_df):
    td_df.info()
    return


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
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

    daylevel_df.head(20)
    return (daylevel_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Giorni risposti per utente + summary partecipazione
    """)
    return


@app.cell
def _(daylevel_df):
    # giorni con almeno 1 risposta valida
    days_with_valid = (
        daylevel_df.assign(has_valid=daylevel_df["notifications_valid"] > 0)
        .groupby(["id", "first2w"], as_index=False)
        .agg(days_with_valid=("has_valid", "sum"))
    )

    # summary notifiche/giorno per utente
    participation_summary = (
        daylevel_df.groupby(["id", "first2w"], as_index=False)
        .agg(
            mean_valid_per_day=("notifications_valid", "mean"),
            median_valid_per_day=("notifications_valid", "median"),
            mean_total_per_day=("notifications_total", "mean"),
            total_valid=("notifications_valid", "sum"),
            total_notifications=("notifications_total", "sum"),
        )
        .merge(days_with_valid, on=["id", "first2w"], how="left")
    )

    participation_summary.head(20)
    return (participation_summary,)


@app.cell
def _():
    """
    Classifichiamo i partecipanti in:
    - low contribution: sotto il 50° percentile di total_valid nel periodo
    - average: tra 50° e 75° percentile
    - outstanding: sopra il 75° percentile

    La classificazione è separata per:
    - First two weeks
    - Second two weeks
    """

    # descrittive per total_valid per ogni periodo
    desc = (
        participation_summary
        .groupby("first2w")["total_valid"]
        .describe(percentiles=[0.5, 0.75])
    )

    fw50p = desc.loc["First two weeks", "50%"]
    fw75p = desc.loc["First two weeks", "75%"]

    sw50p = desc.loc["Second two weeks", "50%"]
    sw75p = desc.loc["Second two weeks", "75%"]

    # funzione di classificazione per riga
    def classify(row):
        if row["first2w"] == "First two weeks":
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

    participation_summary = participation_summary.copy()
    participation_summary["contribution_level"] = participation_summary.apply(
        classify, axis=1
    )

    participation_summary[["id", "first2w", "total_valid", "contribution_level"]].head(20)
    return (participation_summary,)


@app.cell(hide_code=True)
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
    _g.map(sns.histplot, "mean_valid_per_day")
    return


@app.cell
def _(participation_summary, sns):
    _g2 = sns.FacetGrid(participation_summary, col="first2w", height=3, aspect=1.3)
    _g2.map(sns.histplot, "days_with_valid")
    return


@app.cell(hide_code=True)
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
    participation_summary.groupby("first2w")[["mean_valid_per_day", "days_with_valid"]].describe().T
    return


@app.cell
def _(participation_summary):
    # --- soglie "utente valido" per periodo ---
    # First two weeks: >=30 risposte valide/giorno e >=14 giorni risposti
    # Second two weeks: >=12 risposte valide/giorno e >=5 giorni risposti

    def is_valid_user(row):
        if row["first2w"] == "First two weeks":
            return (row["mean_valid_per_day"] >= 30) and (row["days_with_valid"] >= 14)
        else:  # Second two weeks
            return (row["mean_valid_per_day"] >= 12) and (row["days_with_valid"] >= 5)

    participation_summary["valid_user_period"] = participation_summary.apply(is_valid_user, axis=1)

    participation_summary["valid_user_period"].value_counts(dropna=False)
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
def _(valid_wide):
    def classify_pattern(row):
        first_ok = row.get("First two weeks", False)
        second_ok = row.get("Second two weeks", False)

        if first_ok and second_ok:
            return "consistent_high"
        elif first_ok and not second_ok:
            return "early_dropout"
        elif (not first_ok) and second_ok:
            return "late_joiner"
        else:
            return "low_participation"

    valid_wide["participation_pattern"] = valid_wide.apply(classify_pattern, axis=1)
    valid_wide["participation_pattern"].value_counts()
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


@app.cell(hide_code=True)
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
