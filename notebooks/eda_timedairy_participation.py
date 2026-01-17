import marimo

__generated_with = "0.19.4"
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
    return Path, alt, mo, np, pd, plt, sns


@app.cell
def _(mo):
    mo.md(r"""
    # EDA - App Use Participation
    """)
    return


@app.cell
def _(Path, pd):
    # read data
    data_path_app_use = Path("../data/appuseIT_class_17_11.parquet")
    app_use = pd.read_parquet(data_path_app_use)
    app_use.head()
    return (app_use,)


@app.cell
def _(app_use):
    len(app_use["userid"].unique())
    return


@app.cell
def _(app_use):
    app_use.info()
    # encoding info about variables in dataset
    return


@app.cell
def _(mo):
    mo.md(r"""
    # EDA – Time Diary Participation

    Goals:
    - Convert timestamps and filter non-informative nighttime hours.
    - Calculate notifications per day per user.
    - Calculate days responded per user.
    - Create a participation summary.
    - Save cleaned files in data/processed/.
    """)
    return


@app.cell
def _(Path, pd):
    data_path = Path("../data/td_ita.csv")
    td_df = pd.read_csv(data_path, low_memory=False)
    td_df.head(20000000)
    # read dataù
    return data_path, td_df


@app.cell
def _(td_df):
    len(td_df["id"].unique())
    return


@app.cell
def _(td_df):
    td_df.info()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Timestamp Conversion
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
    # Density of valid responses by hour and day
    This shows that between 00.00 and 05.00 there is a lower density of answers
    """)
    return


@app.cell
def _(pd, plt, sns, td_df, td_df_fixed):
    def plot_valid_answers_analysis(td_df):
        # 1. Prepare Data
        df_plot = td_df_fixed.copy()
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
        # Ensure categorical sorting for days
        if df_plot['week'].dtype.name != 'category':
            df_plot['week'] = pd.Categorical(df_plot['week'], categories=day_order, ordered=True)
    
        # Ensure hours are integers
        df_plot['hh_not'] = df_plot['hh_not'].astype(int)
    
        # 2. Aggregation
        # Count valid answers (rows) per Week-Day for each Hour
        pivot = df_plot.groupby(['week', 'hh_not'], observed=False).size().unstack(fill_value=0)
        # Ensure all 24 hours exist
        pivot = pivot.reindex(columns=range(0, 24), fill_value=0)
    
        # 3. Plotting
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10))
    
        # Plot 1: Heatmap
        # This directly shows the density. If 1-5am is low, it will be light-colored.
        sns.heatmap(pivot, ax=ax1, cmap="YlGnBu", annot=True, fmt="d", cbar_kws={'label': 'Count'})
        ax1.set_title("Heatmap: Volume of Valid Answers by Day and Hour", fontsize=14)
        ax1.set_xlabel("Hour of Day (0-23)")
        ax1.set_ylabel("Day of Week")
    
        # Plot 2: Multi-Line Chart
        # This allows comparing the 'shape' of the day. The dip at 1-5am should be visible as a valley.
        pivot_t = pivot.T # Transpose so Index=Hour, Columns=Days
        pivot_t.plot(ax=ax2, marker='o', linewidth=2)
    
        ax2.set_title("Hourly Trend of Valid Answers (Line View)", fontsize=14)
        ax2.set_xlabel("Hour of Day (0-23)")
        ax2.set_ylabel("Count of Answers")
        ax2.set_xticks(range(0, 24))
        ax2.grid(True, linestyle='--', alpha=0.5)
        ax2.legend(title="Day of Week", bbox_to_anchor=(1.02, 1), loc='upper left')
    
        plt.tight_layout()
        return fig

    # Run the function
    _ = plot_valid_answers_analysis(td_df)
    plt.show()
    return


@app.cell
def _(pd, plt, sns, td_df):

    def diagnose_and_fix(td_df):
        print("--- DIAGNOSIS ---")
        # 1. Check the exact counts for Friday (from your screenshot)
        # If this prints the same number 24 times, your data is artificially duplicated.
        friday_check = td_df[td_df['week'] == 'Friday']['hh_not'].value_counts().sort_index()
        print("Counts per hour on Friday (Should vary, but likely flat):")
        print(friday_check)
    
        # 2. THE FIX
        # We must regenerate 'hh_not' from the original timestamp column.
        # REPLACE 'created_at' below with your actual time column name (e.g., 'start_time', 'timestamp')
        timestamp_col = 'datein_answ' # <--- CHANGE THIS to your column name
    
        if timestamp_col in td_df.columns:
            print(f"\nFixing data using column: {timestamp_col}...")
        
            # Convert to datetime if needed
            td_df[timestamp_col] = pd.to_datetime(td_df[timestamp_col])
        
            # Overwrite the corrupted 'hh_not' with the actual hour
            td_df['hh_not'] = td_df[timestamp_col].dt.hour
        
            # Verify the fix
            print("New distribution sample (First 5 hours of Friday):")
            print(td_df[td_df['week'] == 'Friday']['hh_not'].value_counts().sort_index().head(5))
        
            return td_df
        else:
            print(f"\nERROR: Column '{timestamp_col}' not found. Cannot fix hh_not without source time.")
            return td_df

    # Run the diagnosis (and fix if you have the timestamp column)
    td_df_fixed = diagnose_and_fix(td_df)

    # --- PLOT THE FIXED DATA ---
    # Only run this if the fix was successful
    if 'datein_answ' in td_df.columns: # Update this check too
    
        # Re-run the aggregation on the FIXED dataframe
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        td_df_fixed['week'] = pd.Categorical(td_df_fixed['week'], categories=day_order, ordered=True)
    
        pivot = td_df_fixed.groupby(['week', 'hh_not'], observed=False).size().unstack(fill_value=0)
        pivot = pivot.reindex(columns=range(0, 24), fill_value=0)
    
        _, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(pivot, ax=ax, cmap="YlGnBu", annot=True, fmt="d", cbar_kws={'label': 'Count'})
        ax.set_title("Fixed: Real Volume of Valid Answers by Hour")
        plt.tight_layout()
        plt.show()
    return (td_df_fixed,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Filter out non informative nighthours (1–5 AM)
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
    ## Day and Valid answers

    - we aggregated the data by day
    - then we filter out the record with "no information"
    """)
    return


@app.cell
def _(td_df_cleaned):
    # giorno (senza ora)
    td_df_cleaned["day"] = td_df_cleaned["date_not"].dt.date

    # risposta valida = non "No information"
    td_df_cleaned["is_valid"] = td_df_cleaned["what"] != "No information"

    td_df_cleaned[["id", "day", "what", "is_valid"]].head(1000)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Pings by user each day

    Here we are computing:
    - The total pings received by a user each day
    - The valid pings of a user each day (so the "answered" pings)
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
    ## Answered days for each user + summary participation

    Here we compute:
    - Threshold for individual periods validity (How many days is the user active for each period)
    - Threshold for period division:
        - valid users for only first-2-weeks period
        - Valid users for the whole period (users valid in both first-2-weeks and second-2-weeks)
    """)
    return


@app.cell
def _(daylevel_df, np, pd):
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
    participation_summary["median_valid_per_day"] = np.floor(participation_summary["median_valid_per_day"])
    participation_summary.head(20000000)
    return (participation_summary,)


@app.cell
def _(mo):
    mo.md(r"""
    # PLOT SHOWING THE NEW PARTICIPANT DIVISION
    We can see the number of partecipants in each period, and we see that there are more partecipants in the first-2-weeks period then in the whole period section, as many users who started the experiment decided to drop-out after the first-2-weeks period, and so are not included in the whole period section that only counts the people who started the experiment and contiuned it consistently in the second-2-weeks period.
    """)
    return


@app.cell
def _(alt, participation_summary):
    def participation_flow_plot(participation_summary, alt):
        # Prepare data to see the count of users per period
        flow_df = participation_summary.groupby("first2w")["id"].nunique().reset_index()
        flow_df.columns = ["Period", "User Count"]
    
        chart = alt.Chart(flow_df).mark_bar(size=60).encode(
            x=alt.X("Period:N", sort=["First 2 Weeks", "Whole Period"], title="Analysis Group", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("User Count:Q", title="Number of Unique Users"),
            color=alt.Color("Period:N"),
            tooltip=["Period", "User Count"]
        ).properties(
            title="Sample Attrition: First 2 Weeks vs. Whole Period",
            width=400
        )
    
        return chart
    
    participation_flow_plot(participation_summary, alt)
    return


@app.cell
def _(mo):
    mo.md(r"""
    # PLOTS SHOWING THE TRESHOLDS
    These two plots show why we decided to select the specific threshold stated above in order to select the users that have been active enough in the first-2-weeks period and in the second-2-weeks period respectively.
    """)
    return


@app.cell
def _(daylevel_df, plt, sns):
    def threshold_scatter_plot(daylevel_df, plt, sns, f2w=True):
        if f2w:
            # Calculate metrics per user for the first 2 weeks to test thresholds
            check_df = daylevel_df[daylevel_df["first2w"] == "First two weeks"].groupby("id").agg(
                median_notif=("notifications_valid", "median"),
                days_active=("notifications_valid", lambda x: (x > 0).sum())
            ).reset_index()
        else:
            # Calculate metrics per user for the whole period to test thresholds
            check_df = daylevel_df[daylevel_df["first2w"] == "Second two weeks"].groupby("id").agg(
                median_notif=("notifications_valid", "median"),
                days_active=("notifications_valid", lambda x: (x > 0).sum())
            ).reset_index()

        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=check_df, x="days_active", y="median_notif", alpha=0.5)
    
        # Draw your threshold lines
        if f2w:
            plt.axvline(x=14, color='red', linestyle='--', label='Min Days (14)')
            plt.axhline(y=25, color='green', linestyle='--', label='Min Median Notif (25)')
        else:
            plt.axvline(x=7, color='red', linestyle='--', label='Min Days (7)')
            plt.axhline(y=12, color='green', linestyle='--', label='Min Median Notif (12)')
    
        if f2w:
            plt.title("User Distribution vs. Validity Thresholds (first-2-weeks)")
        else:
            plt.title("User Distribution vs. Validity Thresholds (second-2-weeks)")

        plt.xlabel("Days with at least one valid response")
        plt.ylabel("Median valid notifications per day")
        plt.legend()
        return plt.gca()

    threshold_scatter_plot(daylevel_df, plt, sns)
    return (threshold_scatter_plot,)


@app.cell
def _(daylevel_df, plt, sns, threshold_scatter_plot):
    threshold_scatter_plot(daylevel_df, plt, sns, False)
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Contribution Logic Implementation
    Here we attribute to the users different categories to show their parteicpation:
    - Low contribution = users below the 50° percentile in total-valid in each period
    - average = users between 50° and 75° percentile
    - outstanding = above the 75° percentile
    """)
    return


@app.cell
def _(participation_summary):
    def _(participation_summary):
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
        _participation_summary2["contribution_level"] = _participation_summary2.apply(
            classify, axis=1
        )

        _participation_summary2[["id", "first2w", "total_valid", "contribution_level"]].head(20)
        return _participation_summary2

    df_prova = _(participation_summary)
    df_prova.head(20)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## RANDOM PLOTS
    """)
    return


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
def _(mo):
    mo.md(r"""
    ## Useful statistics valid participants distributions (quantiles)
    These descriptives show that the valid partecipants distributions are very skewed, as we can see that when we look at the quantiles for both median_valid_per_day and days_with_valid we see that the quantiles from the 50% on are the same.
    """)
    return


@app.cell
def _(participation_summary):
    participation_summary.groupby("first2w")[["median_valid_per_day", "days_with_valid"]].describe()
    return


@app.cell
def _(mo):
    mo.md(r"""
    # STEP 3 - Participation Patterns

    Goals:
    - define Participation pattern: First 2 weeks and whole period
    - divided two types of partecipants:
        - **Early Phase out** --> only partecipating to the first-2-weeks
        - **Consistent** --> partecipating to the whole period
    """)
    return


@app.cell
def _(participation_summary):
    # Here we are creating the variable "valid_user_period" that we will need later

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
        return p_summary
    _participation_summary = _(participation_summary)
    _participation_summary.head(10)
    return


@app.cell
def _(participation_summary):

    #Here we are aggregating that variable to the whole dataset

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

    participation_summary_overall, _ = _(participation_summary)
    participation_summary_overall.head(200000)
    return (participation_summary_overall,)


@app.cell
def _(mo):
    mo.md(r"""
    # Partecipation Pattern
    Here we create the labels for the partecipation pattern, and we can see that there are 156 users who are consistent in both section of the experiment. Instead, we can see that 16 people are classified as early phase only, which means that they dropped the experiment after the end of the first phase and were not included in the whole period section.
    """)
    return


@app.cell
def _(participation_summary):
    # Here we are labeling the new variable that shows the partecipation pattern of the users
    # - consistent
    # - early phase only


    def _(participation_summary):
        # Identifying the patterns based on our new analysis_period column
        # We want to see who appears in 'Whole Period' (which means they were in both)
        # vs who only appears in 'First 2 Weeks'

        user_counts = participation_summary
        # return user_counts

        def label_pattern(user_id):
            # If they appear in 2 periods (First 2 Weeks AND Whole Period), 
            # it means they passed the 'Persistent' check.
            reduced_df = user_counts[user_counts["id"] == user_id]
            if reduced_df["first2w"].isin(["Whole Period"]).any():
                return "Consistent (Both Periods)"
            else:
                return "Early Phase Only"

        participation_summary["participation_pattern"] = participation_summary["id"].apply(label_pattern)
        return participation_summary.drop_duplicates("id")

    _participation_summary = _(participation_summary)
    _participation_summary["participation_pattern"].value_counts()
    return


@app.cell
def _(participation_summary_overall):
    valid_wide = (
        participation_summary_overall
        .pivot(index="id", columns="first2w", values="valid_user_period")
    )
    # def label_pattern(user_id):
    #     reduced_df = participation_summary[participation_summary["id"] == user_id]
    #     if reduced_df["first2w"].isin(["Whole Period"]).any():
    #         return "Consistent (Both Periods)"
    #     else:
    #         return "Early Phase Only"

    # valid_wide["participation_pattern"] = valid_wide.reset_index()["id"].apply(label_pattern)

    valid_wide.head(2000)
    return (valid_wide,)


@app.cell
def _(mo):
    mo.md(r"""
    # Drop-out users
    Here we can see specifically the ids of the users who dropped out. Indeed, later in the analysis we can look at spcific carachteristics of these users, understading if there is a pattern among them that might motivate why the left the experimet early.
    """)
    return


@app.cell
def _(valid_wide):
    _mask = (valid_wide["First 2 Weeks"]) & (valid_wide["Whole Period"].isna())
    _dropped_participants = valid_wide[_mask]
    _dropped_participants.head(2000)
    return


@app.cell
def _():
    """# valid_wide_reset = valid_wide["participation_pattern"].reset_index()

    # participation_summary_patterns = participation_summary_overall.merge(
    #     valid_wide_reset, on="id", how="left"
    # )

    # participation_summary_patterns[
    #     ["id", "first2w", "valid_user_period", "valid_user_overall", "participation_pattern"]
    # ].head()"""
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Salvataggio output puliti
    """)
    return


@app.cell
def _(data_path, daylevel_df, participation_summary, pd):
    def _(data_path, daylevel_df, participation_summary, pd):
        processed_path = data_path.parent / "processed"
        processed_path.mkdir(exist_ok=True)

        out_daylevel = processed_path / "td_participation_daylevel.parquet"
        out_summary = processed_path / "td_participation_summary.parquet"

        # FIX: Convert 'day' column from python 'date' objects to pandas datetime objects
        # This prevents the "Can't infer object conversion type" error in fastparquet
        daylevel_df["day"] = pd.to_datetime(daylevel_df["day"])

        daylevel_df.to_parquet(out_daylevel, index=False)
        participation_summary.to_parquet(out_summary, index=False)

        return out_daylevel, out_summary

    _(data_path, daylevel_df, participation_summary, pd)
    return


if __name__ == "__main__":
    app.run()
