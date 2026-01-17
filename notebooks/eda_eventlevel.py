import marimo

__generated_with = "0.19.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    from pathlib import Path
    import matplotlib.pyplot as plt
    import seaborn as sns
    return Path, mo, np, pd, plt, sns


@app.cell
def _(mo):
    mo.md(r"""
    # Event-level dataset: mood & smartphone use

    Goal:
    - Building a dataset using **mood events**:
      - actual mood (`A6a`)
      - previous mood
      - change of mood (`delta_mood`) and category (migliora/uguale/peggiora)

    - Implement a logic for the use of the smartphone between two pings:
      - amount time of usage (o number of ping) for:
        - **social** apps
        - **communication** apps
        - **travel** apps
        - **productivity** apps
        - **gaming** apps
        - **finance** apps
        - **news** apps
        - **entertainment** apps

    - We keep only the valid users (how it was defined in the time diary file)

    The final output:
    `./data/processed/eventlevel_mood_phoneuse.parquet`
    """)
    return


@app.cell
def _(Path, pd):
    # percorsi ai file di input
    appregress_path = Path("../data/app4regress_IT_new_v2.parquet")
    appuse_path = Path("../data/appuseIT_class_17_11.parquet")
    td_summary_path = Path("../data/processed/td_participation_summary.parquet")
    socio_path = Path("../data/processed/socio_demo_cleaned.parquet")

    appregress_df = pd.read_parquet(appregress_path)
    appuse_df = pd.read_parquet(appuse_path)
    participation_summary = pd.read_parquet(td_summary_path)
    socio_df_clean = pd.read_parquet(socio_path)

    appregress_df.shape, appuse_df.shape, participation_summary.shape, socio_df_clean.shape
    return appregress_df, appuse_df, participation_summary, socio_df_clean


@app.cell
def _(mo):
    mo.md(r"""
    ## 1.a – Quick peek dei dataset
    """)
    return


@app.cell
def _(appregress_df, appuse_df, participation_summary, socio_df_clean):
    appregress_df.head(), appuse_df.head(), participation_summary.head(), socio_df_clean.head()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2 – Costruzione dataset evento-per-evento di mood

    Usiamo `app4regress_IT_new_v2.parquet`, che contiene:
    - `userid`
    - `timestamp`
    - `A6a` (mood attuale)
    - `delta_mood1` (differenza con la misurazione precedente, già calcolata)

    Costruiamo:
    - `mood_prev`
    - `delta_mood` (A6a - mood_prev)
    - `mood_change_cat` ∈ {migliora, uguale, peggiora}
    - `event_id` (indice evento per utente)
    """)
    return


@app.cell
def _(appregress_df, pd):
    mood_df = (
        appregress_df[["userid", "timestamp", "A6a", "delta_mood1"]]
        .copy()
        .sort_values(["userid", "timestamp"])
    )

    # timestamp sicuro come datetime
    mood_df["timestamp"] = pd.to_datetime(mood_df["timestamp"])

    # mood precedente per utente
    mood_df["mood_prev"] = mood_df.groupby("userid")["A6a"].shift(1)

    # delta_mood: calcoliamo sempre noi la differenza con la misura precedente
    mood_df["delta_mood"] = mood_df["A6a"] - mood_df["mood_prev"]

        # categoria di cambiamento
    def categorize_delta(d):
            if pd.isna(d):
                return pd.NA
            if d > 0:
                return "migliora"
            elif d < 0:
                return "peggiora"
            else:
                return "uguale"

    mood_df["mood_change_cat"] = mood_df["delta_mood"].apply(categorize_delta)

        # id evento per utente
    mood_df["event_id"] = mood_df.groupby("userid").cumcount() + 1

    mood_df.head(20)
    return (mood_df,)


@app.cell
def _(mood_df):
    # confronto tra delta_mood1 (fornito) e delta calcolato da noi
    diff = mood_df["delta_mood"] - mood_df["delta_mood1"]
    diff.value_counts(dropna=False).head()


    #Abbiamo verificato che la variabile pre-calcolata delta_mood1 coincide con la differenza calcolata da noi
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Mood Latent Variable
    Here we are creating the new column where we have the latent variable
    """)
    return


@app.cell
def _(mood_df):
    mood_df["latent_var"] = mood_df["A6a"].map({1:-1.75,
    2:-0.86,
    3:0,
    4:0.86,
    5:1.75})

    mood_df
    return


@app.cell
def _(mood_df):
    mood_df_sorted = mood_df.sort_values(["userid","timestamp"],ascending=True)
    mood_df_sorted.head()
    return (mood_df_sorted,)


@app.cell
def _(mood_df_sorted, pd):
    dfs = []
    for id in mood_df_sorted["userid"].unique():
        id_df = mood_df_sorted[mood_df_sorted["userid"] == id].copy()
        id_df["latent_prev"] = id_df["latent_var"].shift(1)
        id_df["latent_change"] = id_df["latent_var"] - id_df["latent_prev"]
        dfs.append(id_df)

    df_mood_latent = pd.concat(dfs, axis=0).reset_index(drop=True)
    return (df_mood_latent,)


@app.cell
def _(df_mood_latent):
    df_mood_latent.head(100000) # use this dataset for all later use of the
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3 – Utenti validi (da time diary)

    Partiamo da `td_participation_summary.parquet` e ricostruiamo:

    - `valid_user_period` (per ciascun periodo, usando le soglie)
    - `valid_user_overall` (utente valido **almeno in un periodo**)

    Poi teniamo solo gli eventi di mood di utenti validi.
    """)
    return


@app.cell
def _(participation_summary):
    participation_summary_clean = participation_summary.copy()

    # soglie per periodo: riprese dallo script del time diary
    def is_valid_user(row):
        if row["first2w"] == "First two weeks":
            return (row["median_valid_per_day"] >= 25) and (row["days_with_valid"] >= 14)
        else:  # Second two weeks
            return (row["median_valid_per_day"] >= 12) and (row["days_with_valid"] >= 7)

    participation_summary_clean["valid_user_period"] = participation_summary_clean.apply(is_valid_user, axis=1)

    # validità complessiva: utente valido almeno in un periodo
    valid_overall = (
        participation_summary_clean
        .groupby("id", as_index=False)["valid_user_period"]
        .any()
        .rename(columns={"valid_user_period": "valid_user_overall"})
    )

    participation_summary_clean.head(), valid_overall.head()



    #valid_user_overall = utente valido in almeno uno dei due periodi (any()):se è valido in prime 2 settimane ma non nelle seconde → resta;se è valido solo dopo → resta;se non è mai valido → escluso.
    return (valid_overall,)


@app.cell
def _(appuse_df, valid_overall):
    # CHANGED ON 05/01
    # Get unique users who actually have phone logs
    users_with_phone_data = set(appuse_df["userid"].unique())

    # Filter valid_overall to only include those who also have phone data
    valid_users_with_phone = valid_overall[
        (valid_overall["id"].isin(users_with_phone_data)) & 
        (valid_overall["valid_user_overall"] == True)
    ].rename(columns={"id": "userid"})

    valid_users_with_phone.head(100000000)
    return (valid_users_with_phone,)


@app.cell
def _(df_mood_latent, valid_users_with_phone):
    # CHANGED ON 05/01
    # rinominiamo id -> userid per fare merge con appregress
    valid_users = valid_users_with_phone.rename(columns={"id": "userid"})

    # teniamo solo utenti con valid_user_overall == True
    valid_users = valid_users_with_phone.query("valid_user_overall == True")

    mood_valid_df = df_mood_latent.merge(
        valid_users_with_phone[["userid", "valid_user_overall"]],
        on="userid",
        how="inner",
    )

    mood_valid_df.head(100000), valid_users["userid"].nunique()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4 – Intervalli tra notifiche

    Per ogni evento di mood (tranne il primo per utente) definiamo:

    - `t_prev` = timestamp della notifica precedente
    - `t_curr` = timestamp della notifica corrente

    Useremo questi intervalli per aggregare l’uso del telefono tra le due notifiche.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Nota sulla scelta della finestra temporale**

    In questa analisi aggreghiamo l’uso dello smartphone **nell’intero intervallo tra una notifica di mood e la successiva**.
    L’idea è che il mood rilevato a T1 rifletta (almeno in parte) le attività svolte dopo la misurazione precedente T0.

    Siamo consapevoli che esistono altre strategie possibili (es. finestre più brevi
    o centrate sulla notifica) e che ogni scelta ha implicazioni. Se necessario
    potremmo testare anche specifiche alternative, ma qui usiamo l’intervallo
    T0–T1 perché è la scelta più coerente con la nostra domanda di ricerca.
    """)
    return


@app.cell
def _():
    """# partiamo da mood_valid_df (creato nella cella precedente)
    mood_valid_sorted = mood_valid_df.sort_values(["userid", "timestamp"]).copy()

    mood_valid_sorted["t_curr"] = mood_valid_sorted["timestamp"]
    mood_valid_sorted["t_prev"] = mood_valid_sorted.groupby("userid")["t_curr"].shift(1)

    # teniamo solo gli eventi che hanno una notifica precedente definita
    mood_interval_df = mood_valid_sorted.dropna(subset=["t_prev"]).copy()

    mood_interval_df[[
        "userid",
        "event_id",
        "t_prev",
        "t_curr",
        "A6a",
        "mood_prev",
        "delta_mood",
        "mood_change_cat",
    ]].head(30)"""
    return


@app.cell
def _():
    """# aggiungiamo la durata dell'intervallo tra due EMA

    # 1) durata in minuti
    mood_interval_df["delta_t_min"] = (
        (mood_interval_df["t_curr"] - mood_interval_df["t_prev"])
        .dt.total_seconds() / 60
    )

    # 2) durata in ore
    mood_interval_df["delta_t_hours"] = mood_interval_df["delta_t_min"] / 60

    # 3) descrittive della durata (in minuti) con alcuni quantili
    mood_interval_df["delta_t_min"].describe(
        percentiles=[0.25, 0.5, 0.75, 0.9, 0.95]
    )

    # 4) preview per controllare che tutto abbia senso
    mood_interval_df[[
        "userid", "event_id", "t_prev", "t_curr",
        "delta_t_min", "delta_t_hours",
        "A6a", "mood_prev", "delta_mood", "mood_change_cat",
    ]].head(20)"""
    return


@app.cell
def _(mood_interval_df, pd):
    # distribuzione della durata degli intervalli tra due EMA

    # fasce di durata in MINUTI
    bins = [0, 30, 60, 90, 120, 240, 480, 1440]  # 0–30, 30–60, 60–90, ecc.
    labels = [
        "0–30 min",
        "30–60 min",
        "60–90 min",
        "90–120 min",
        "2–4 h",
        "4–8 h",
        "8–24 h",
    ]

    # categorizziamo ogni intervallo nella sua fascia
    delta_cat = pd.cut(
        mood_interval_df["delta_t_min"],
        bins=bins,
        labels=labels,
        right=True
    )

    # calcoliamo la proporzione di intervalli in ogni fascia
    dist = (
        delta_cat.value_counts(dropna=True, normalize=True)
        .sort_index()
        .rename("proportion")
        .reset_index()
    )

    dist
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Preparation App-Use
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    - convertiamo timestamp in datetime
    - creiamo la macro-categoria:
      * communication: chat, chiamate, email
      * social: social network, tiktok, ecc.
      * entertainment
      * gaming
      * finance
      * news
      * travel
      * productivity
    """)
    return


@app.cell
def _(appuse_df, plt):
    #Here we try to see the total apps recorded in the study

    # Extract the complete list of unique app names
    app_column = 'App' if 'App' in appuse_df.columns else 'applicationname'
    unique_apps = appuse_df[app_column].unique()

    # Sort and print the full list
    unique_apps.sort()
    print(f"Total Unique Apps: {len(unique_apps)}")
    for app in unique_apps:
        print(app)

    # Optional: Visualize the top 20 most frequent apps in the dataset

    appuse_df[app_column].value_counts().head(20).plot(kind='barh', color='skyblue')
    plt.title('Top 20 Apps by Usage Frequency')
    plt.xlabel('Frequency')
    plt.ylabel('App Name')
    plt.gca().invert_yaxis()
    plt.show()
    return


@app.cell
def _():
    """
    Preparazione app use:
    - togliamo 'android' (processo di sistema)
    - convertiamo timestamp in datetime
    - creiamo la macro-categorie
    """

    def process_app_usage_verified(appuse_df):
        # Verified category map based on unique strings in appuseIT_class_17_11.parquet
        category_map = {
            'SOCIAL': ['instagram', 'facebook', 'tiktok', 'reddit', 'linkedin', 'snapchat', 'weverse', 'tinder', 'badoo', 'twitter'],
            'COMMUNICATION': ['whatsapp', 'telegram', 'gmail', 'slack', 'messenger', 'discord', 'outlook', 'protonmail', 'yahoo', "email"],
            'ENTERTAINMENT': ['youtube', 'spotify', 'netflix', 'twitch', 'kindle', 'disney', 'infinitytv', 'audiobook', 'shazam'],
            'PRODUCTIVITY': ['microsoft', 'adobe', 'zoom', 'teams', 'classroom'],
            'TRAVEL': ['maps', 'ryanair', 'flixbus', 'booking', 'uber', 'airbnb', 'meteo', 'komoot', 'trentino', 'italotreno', 'trenitalia'],
            'GAMING': ['puzzle', 'game', 'candycrush', 'chess', 'escape', 'clash', 'Briscola', 'BurracoOnline', 'Scopa'],
            'NEWS': ['bbc news', 'tg.la7', 'tg5', 'skytg24', 'radio24', 'magazines', 'newspaper', 'corriere'],
            'FINANCE': ['paypal', 'revolut', 'satispay', 'bitcoin', 'mypayroll', 'scrigno', 'fineco', 'wallet', 'crypto', 'bank', 'posteitaliane', 'banca']
        }

        def categorize_strict(app_name):
            app_lower = str(app_name).lower()
            for category, keywords in category_map.items():
                if any(kw in app_lower for kw in keywords):
                    return category
            # If it doesn't match any of the 8, return None to DISCARD
            return None

        # Apply categorization and explicitly DROP all unmapped apps
        appuse_df['research_category'] = appuse_df['applicationname'].apply(categorize_strict)
        filtered_df = appuse_df.dropna(subset=['research_category']).copy()

        # Create the wide table for the final dataset merge
        usage_wide = (
            filtered_df.groupby(["userid", "research_category"])
            .size()
            .unstack(fill_value=0)
        )

        # Standardize column names to: use_social, use_communication, etc.
        usage_wide.columns = [f"use_{col.lower()}" for col in usage_wide.columns]

        return usage_wide, filtered_df


    return (process_app_usage_verified,)


@app.cell
def _(appuse_df, process_app_usage_verified):
    _usage_wide, filtered_df = process_app_usage_verified(appuse_df)
    return (filtered_df,)


@app.cell
def _(filtered_df):
    # 2. Extract unique app names that were NOT discarded
    # We sort them to make the list easier to read
    kept_app_names = sorted(filtered_df['applicationname'].unique())

    # 3. Display the results
    print(f"Total Unique Apps Kept: {len(kept_app_names)}")
    print("-" * 30)
    for name in kept_app_names:
        print(name)

    # Grouping the unique names by their assigned research category
    grouped_apps = filtered_df.groupby('research_category')['applicationname'].unique()

    for category, apps in grouped_apps.items():
        print(f"\n[ CATEGORY: {category} ]")
        # Sort the apps within each category for clarity
        for a in sorted(apps):
            print(f"  - {a}")
    return


@app.cell
def _(filtered_df, plt):
    def visualize_results(filtered_df, plt):
        # 1. Visualize Overall Category Distribution
        def plot_distribution(df):
            # We use a unique name for the figure to avoid global collisions
            fig_dist, ax_dist = plt.subplots(figsize=(10, 6))
            category_counts = df['research_category'].value_counts()
            category_counts.plot(kind='bar', color='teal', ax=ax_dist)

            ax_dist.set_title('Distribution of Smartphone Use by Research Category')
            ax_dist.set_xlabel('Category')
            ax_dist.set_ylabel('Number of Logs')
            plt.xticks(rotation=45)
            ax_dist.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            return fig_dist

        # 2. Visualize Top Apps per Category (Verification)
        def plot_top_apps(df):
            categories_list = sorted(df['research_category'].unique())
            # Using a unique name for this specific figure and axes grid
            fig_grid, axes_grid = plt.subplots(nrows=4, ncols=2, figsize=(15, 18))
            axes_flat = axes_grid.flatten()

            for idx, cat_name in enumerate(categories_list):
                # Verify the classification of apps like 'WPS Office' or 'Scrigno'
                top_ten = df[df['research_category'] == cat_name]['applicationname'].value_counts().head(10)
                top_ten.plot(kind='barh', ax=axes_flat[idx], color='skyblue')
                axes_flat[idx].set_title(f'Top Apps in {cat_name}')
                axes_flat[idx].invert_yaxis()
                axes_flat[idx].set_xlabel('Frequency')

            # Remove empty subplots if you have fewer than 8 categories
            for j in range(idx + 1, len(axes_flat)):
                fig_grid.delaxes(axes_flat[j])

            plt.tight_layout()
            return fig_grid

        # Display both plots in the marimo UI by returning them as a vertical stack
        import marimo as mo
        return mo.vstack([
            mo.md("### Category Distribution"),
            plot_distribution(filtered_df),
            mo.md("### Top Apps per Category Verification"),
            plot_top_apps(filtered_df)
        ])

    visualize_results(filtered_df, plt)
    return


@app.cell
def _(appuse_df):
    sorted(appuse_df["userid"].unique())
    return


@app.cell
def _(filtered_df, pd, plt, sns):
    # 1. Ensure timestamp is datetime and extract the hour
    # (Using 'filtered_df' which is returned by your function)
    filtered_df['timestamp'] = pd.to_datetime(filtered_df['timestamp'])
    filtered_df['hour'] = filtered_df['timestamp'].dt.hour

    # 2. Define the time_of_day function (consistent with your script)
    def get_time_of_day(h):
        if 5 <= h < 12:
            return "1. Morning (5-12)"
        elif 12 <= h < 18:
            return "2. Afternoon (12-18)"
        elif 18 <= h < 24:
            return "3. Evening (18-24)"
        else:
            return "4. Night (0-1)"

    # 3. Apply the time of day bins
    filtered_df['time_period'] = filtered_df['hour'].apply(get_time_of_day)

    # 4. Create the Visualization
    def plot_category_usage_by_time(df):
        # Unique variable names to avoid marimo errors
        fig_time, ax_time = plt.subplots(figsize=(14, 8))
    
        # Aggregate data: Count occurrences of each category per time period
        plot_data = df.groupby(['time_period', 'research_category']).size().reset_index(name='count')
    
        # Create a grouped bar chart
        sns.barplot(
            data=plot_data, 
            x='time_period', 
            y='count', 
            hue='research_category', 
            ax=ax_time,
            palette='viridis'
        )
    
        ax_time.set_title('App Category Usage Frequency by Time of Day', fontsize=16)
        ax_time.set_xlabel('Time Period', fontsize=12)
        ax_time.set_ylabel('Number of App Uses', fontsize=12)
        ax_time.legend(title='App Category', bbox_to_anchor=(1.05, 1), loc='upper left')
    
        plt.tight_layout()
        return fig_time

    # 5. Display the plot
    plot_category_usage_by_time(filtered_df)
    return (get_time_of_day,)


@app.cell
def _(filtered_df, pd, plt, sns):
    def visualize_category_facets(filtered_df, plt):

        # 1. Prepare the data: Extract hour from timestamp
        df_plot = filtered_df.copy()
        df_plot['timestamp'] = pd.to_datetime(df_plot['timestamp'])
        df_plot['hour'] = df_plot['timestamp'].dt.hour
    
        # 2. Get the list of categories
        categories = sorted(df_plot['research_category'].unique())
    
        # 3. Create a grid of subplots (4 rows, 2 columns for 8 categories)
        # Use unique variable names to avoid marimo "redefined variables" errors
        fig_facets, axes_facets = plt.subplots(nrows=4, ncols=2, figsize=(15, 20))
        axes_flat = axes_facets.flatten()
    
        for i, cat in enumerate(categories):
            # Filter data for the specific category
            cat_data = df_plot[df_plot['research_category'] == cat]
        
            # Plot the hourly distribution (0-23)
            sns.histplot(
                data=cat_data, 
                x='hour', 
                bins=24, 
                binrange=(0, 24), 
                ax=axes_flat[i], 
                color='teal',
                kde=False  # Adds a trend line to see the "routine" shape
            )
        
            axes_flat[i].set_title(f'Daily Distribution: {cat}', fontsize=14, fontweight='bold')
            axes_flat[i].set_xlabel('Hour of Day (0-23)')
            axes_flat[i].set_ylabel('Frequency of Use')
            axes_flat[i].set_xlim(0, 23)
            axes_flat[i].grid(axis='y', linestyle='--', alpha=0.5)

        # Clean up empty subplots if there are fewer than 8 categories
        for j in range(i + 1, len(axes_flat)):
            fig_facets.delaxes(axes_flat[j])

        plt.tight_layout()
    
        # Return the figure object so marimo renders it immediately
        return fig_facets

    visualize_category_facets(filtered_df, plt)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6 – Uso del telefono tra due notifiche

    Idea:
    - per ogni intervallo [t_prev, t_curr) di un evento di mood
    - prendiamo i ping di `appuse_df` con stesso `userid` e `timestamp` compreso nell'intervallo
    - aggreghiamo il numero di ping (o unità di tempo) per macro-categoria.

    Usiamo `merge_asof` per associare ogni ping al precedente `t_prev` dello stesso utente.

    cioè capire quante volte l’utente ha usato il telefono in quel pezzo di tempo,
    e per ogni ping di uso app (righe di appuse_prepped):
    """)
    return


@app.cell
def _(filtered_df, mood_interval_df, pd):
    # 6 – Uso del telefono tra due notifiche
    # Obiettivo: per ogni ping di app capire in quale intervallo [t_prev, t_curr) cade.

    # Copie di lavoro
    appuse_for_merge = filtered_df.copy()
    mood_intervals_for_merge = mood_interval_df.copy()

    # 1. Chiavi temporali coerenti (stesso dtype esatto)
    #    - ts_app: timestamp dei ping di app
    #    - t_prev_key: inizio intervallo EMA
    appuse_for_merge["ts_app"] = pd.to_datetime(appuse_for_merge["timestamp"]).astype("datetime64[ns]")
    mood_intervals_for_merge["t_prev_key"] = pd.to_datetime(mood_intervals_for_merge["t_prev"]).astype("datetime64[ns]")
    mood_intervals_for_merge["t_curr"] = pd.to_datetime(mood_intervals_for_merge["t_curr"]).astype("datetime64[ns]")

    # allineiamo il tipo di userid (int) in entrambi i dataset
    appuse_for_merge["userid"] = appuse_for_merge["userid"].astype(int)
    mood_intervals_for_merge["userid"] = mood_intervals_for_merge["userid"].astype(int)

    merged_list = []

    # 2. Merge utente per utente
    for uid, user_logs in appuse_for_merge.groupby("userid"):

        # intervalli EMA di quell'utente
        user_intervals = (
            mood_intervals_for_merge
            .loc[mood_intervals_for_merge["userid"] == uid]
            .sort_values("t_prev_key")
            .copy()
        )
        if user_intervals.empty:
            # utente che non ha EMA validi -> saltiamo
            continue

        # ordiniamo i ping di uso app nel tempo
        user_logs = user_logs.sort_values("ts_app").copy()

        # per ogni ping di app, agganciamo l'ultimo t_prev precedente (stesso utente)
        tmp = pd.merge_asof(
            user_logs,
            user_intervals,
            left_on="ts_app",
            right_on="t_prev_key",
            direction="backward",
            allow_exact_matches=True,
        )

        # teniamo solo i ping compresi nell'intervallo [t_prev, t_curr)
        tmp = tmp[tmp["ts_app"] < tmp["t_curr"]]

        merged_list.append(tmp)

    # 3. Uniamo tutti gli utenti
    merged = pd.concat(merged_list, ignore_index=True)

    # 4. Sistemiamo eventuali colonne userid_x / userid_y
    if "userid_x" in merged.columns:
        merged = merged.rename(columns={"userid_x": "userid"})
    if "userid_y" in merged.columns:
        merged = merged.drop(columns=["userid_y"])

    # 5. Controllo rapido
    merged[[
        "userid",
        "ts_app",           # momento del ping app
        "applicationname",
        "research_category",
        "event_id",
        "t_prev",
        "t_curr",
    ]].head(20)
    return (merged,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 7 – Aggregazione: app-categories per evento

    Per ogni coppia (`userid`, `event_id`) calcoliamo:
    - numero di ping (o unità di tempo) per `research_categories`
    - e trasformiamo in formato wide:
    """)
    return


@app.cell
def _(merged):
    # 7 – Aggregazione

    # per sicurezza
    merged["event_id"] = merged["event_id"].astype(int)

    # tempo totale di uso per intervallo *categoria*
    usage_agg = (
        merged
        .groupby(["userid", "event_id", "research_category"], as_index=False)
        ["time_use_second_app"]
        .sum()
        .rename(columns={"time_use_second_app": "time_use_sec"})
    )

    # passiamo a formato wide: una colonna per categoria
    usage_wide = (
        usage_agg
        .pivot(index=["userid", "event_id"], columns="research_category", values="time_use_sec")
        .fillna(0)
        .reset_index()
    )


    # ADD THIS: Rename columns to match what Cell 8 expects
    usage_wide.columns = [
        f"use_{col.lower()}" if col not in ["userid", "event_id"] else col 
        for col in usage_wide.columns
    ]

    # ADD THIS: Rename columns to match what Cell 8 expects
    required_cats = ["social", "communication", "gaming", "entertainment", "travel", "finance", "news", "productivity"]
    for cat in required_cats:
        col_name = f"use_{cat}"
        if col_name not in usage_wide.columns:
            usage_wide[col_name] = 0.0


    usage_wide.head(30000000)
    return (usage_wide,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 8 – Dataset finale evento-per-evento

    Merge di:
    - `mood_interval_df` (mood, delta, categorie, intervalli)
    - `usage_wide` (uso del telefono tra notifiche)
    - `socio_df_clean` (variabili socio-demo)

    Output: una riga per **evento di mood**.
    """)
    return


@app.cell
def _(get_time_of_day, mood_interval_df, socio_df_clean, usage_wide):
    # 8 – Dataset finale evento-per-evento

    def time_of_day(h):
        if 5 <= h < 12:
            return "morning"
        elif 12 <= h < 18:
            return "afternoon"
        elif 18 <= h < 24:
            return "evening"
        else:
            return "night"

    event_df = mood_interval_df.merge(
        usage_wide,
        on=["userid", "event_id"],
        how="left",
    )

    # se per qualche evento non abbiamo uso registrato, mettiamo 0
    for var in ["use_social", "use_communication", "use_travel", "use_entertainment", "use_gaming", "use_productivity", "use_finance", "use_news"]:
        event_df[var] = event_df[var].fillna(0)


    # aggiungiamo ora e fascia oraria dell’evento (t_curr)
    event_df["hour_curr"] = event_df["t_curr"].dt.hour
    event_df["time_of_day"] = event_df["hour_curr"].apply(get_time_of_day)

    # selezione di alcune variabili socio-demo (usa quelle che esistono davvero in socio_df_clean)
    socio_cols = [
        "userid",
        "gender",
        "degree",
        "department",
        "cohort_group",
    ]

    socio_small = socio_df_clean[[c for c in socio_cols if c in socio_df_clean.columns]].copy()

    # Before merging socio_small in Cell 8
    socio_small["userid"] = socio_small["userid"].astype(int) 
    event_df = event_df.merge(socio_small, on="userid", how="left")

    # IMPORTANT: Ensure latent variables are included
    # They should already be in mood_interval_df if you used df_mood_latent earlier
    print(f"Final event_df shape: {event_df.shape}")
    print(f"Columns in event_df: {event_df.columns.tolist()}")

    event_df.head(50)
    return (event_df,)


@app.cell
def _(event_df):
    event_df["userid"].unique()
    #per vedere quali sono gli id validi
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Mood Transitions Over Events: Visualize how mood changes from event to event
    """)
    return


@app.cell
def _(event_df, pd, plt, sns):
    def _(event_df, plt, sns):
        # Calculate transition probabilities
        mood_transitions = pd.crosstab(
            event_df["mood_prev"], 
            event_df["A6a"],
            normalize="index"
        ) * 100
    
        plt.figure(figsize=(10, 8))
        sns.heatmap(mood_transitions, annot=True, fmt=".1f", cmap="RdYlGn",
                    center=20, vmin=0, vmax=40)
        plt.title("Mood Transition Matrix (% from previous to current)")
        plt.xlabel("Current Mood (A6a)")
        plt.ylabel("Previous Mood")
        plt.tight_layout()
        plt.show()

    _(event_df, plt, sns)
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Latent Mood Change Distribution: Examine the distribution of mood changes on the latent scale
    """)
    return


@app.cell
def _(event_df, plt, sns):
    def _(event_df, plt, sns):
        if "latent_change" in event_df.columns:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
            # Histogram
            axes[0].hist(event_df["latent_change"].dropna(), bins=50, 
                         edgecolor='black', alpha=0.7)
            axes[0].axvline(0, color='red', linestyle='--', linewidth=2)
            axes[0].set_xlabel("Latent Mood Change")
            axes[0].set_ylabel("Frequency")
            axes[0].set_title("Distribution of Latent Mood Changes")
            axes[0].grid(True, alpha=0.3)
        
            # Compare ordinal vs latent changes
            comparison_df = event_df[["delta_mood", "latent_change"]].dropna()
        
            sns.violinplot(data=comparison_df, x="delta_mood", y="latent_change", 
                          ax=axes[1])
            axes[1].axhline(0, color='red', linestyle='--', alpha=0.5)
            axes[1].set_xlabel("Ordinal Mood Change")
            axes[1].set_ylabel("Latent Mood Change")
            axes[1].set_title("Ordinal vs Latent Mood Change")
        
            plt.tight_layout()
            plt.show()
        return

    _(event_df, plt, sns)
    return


@app.cell
def _(event_df, plt, sns):

    # =============================================================================
    # PLOT 1: Overview - Total usage by app category (descriptive)
    # =============================================================================

    def _(event_df, plt, sns):
        """
        Descriptive plot showing the distribution of usage across all 8 categories
        """
        usage_cols = [col for col in event_df.columns if col.startswith('use_')]
    
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
    
        for idx, col in enumerate(sorted(usage_cols)):
            cat_name = col.replace('use_', '').capitalize()
        
            # Convert to minutes for readability
            data_minutes = event_df[col] / 60
        
            # Filter out zeros for better visualization
            data_nonzero = data_minutes[data_minutes > 0]
        
            axes[idx].hist(data_nonzero, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
            axes[idx].set_title(f'{cat_name}\n(Non-zero values only)', fontweight='bold')
            axes[idx].set_xlabel('Minutes')
            axes[idx].set_ylabel('Frequency')
            axes[idx].axvline(data_nonzero.median(), color='red', linestyle='--', 
                             linewidth=2, label=f'Median: {data_nonzero.median():.1f}min')
            axes[idx].legend()
            axes[idx].grid(alpha=0.3, axis='y')
        
            # Add text with % of zeros
            pct_zeros = (event_df[col] == 0).sum() / len(event_df) * 100
            axes[idx].text(0.95, 0.95, f'{pct_zeros:.1f}% zeros', 
                          transform=axes[idx].transAxes, 
                          verticalalignment='top', horizontalalignment='right',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
        plt.tight_layout()
        plt.savefig('../plots/1_usage_distribution_by_category.png', dpi=300, bbox_inches='tight')
        plt.show()
        return

    _(event_df, plt, sns)
    return


@app.cell
def _(event_df, pd, plt, sns):
    # =============================================================================
    # PLOT 2: Mood trajectory over weeks by usage level (top 3 categories)
    # =============================================================================

    def _(event_df, pd, plt, sns):
        """
        Shows how mood evolves over study weeks for high vs low users
        Uses the top 3 most-used categories
        """
        usage_cols = [col for col in event_df.columns if col.startswith('use_')]
    
        plot_df = event_df.copy()
    
        # Find top 3 most-used categories
        category_totals = {col: plot_df[col].sum() for col in usage_cols}
        top_3_categories = sorted(category_totals.items(), key=lambda x: x[1], reverse=True)[:3]
        top_3_cols = [col for col, _ in top_3_categories]
    
        print(f"Top 3 used categories: {top_3_cols}")
    
        # Add study week
        plot_df['study_week'] = (
            (plot_df['t_curr'] - plot_df.groupby('userid')['t_curr'].transform('min')).dt.days // 7
        )
    
        # Filter to weeks 0-3 (first 4 weeks)
        plot_df = plot_df[plot_df['study_week'] <= 3]
    
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
        for idx, cat in enumerate(top_3_cols):
            cat_name = cat.replace('use_', '').capitalize()
        
            # Create usage tertiles (low/medium/high)
            user_usage = plot_df.groupby('userid')[cat].mean()
            usage_groups = pd.qcut(user_usage, q=3, labels=['Low', 'Medium', 'High'], duplicates='drop')
        
            temp_df = plot_df.copy()
            temp_df['usage_group'] = temp_df['userid'].map(usage_groups)
        
            # Aggregate mood by week and usage group
            weekly_mood = temp_df.groupby(['study_week', 'usage_group'])['A6a'].mean().reset_index()
        
            sns.lineplot(
                data=weekly_mood,
                x='study_week',
                y='A6a',
                hue='usage_group',
                ax=axes[idx],
                palette='tab10',
                linewidth=2.5,
                marker='o',
                markersize=8
            )
            axes[idx].set_title(f'Mood Over Weeks by {cat_name} Usage', fontsize=13, fontweight='bold')
            axes[idx].set_xlabel('Study Week')
            axes[idx].set_ylabel('Average Mood')
            axes[idx].legend(title=f'{cat_name} Usage', loc='best')
            axes[idx].grid(alpha=0.3)
            axes[idx].set_ylim(0, 5.0)
    
        plt.tight_layout()
        plt.savefig('../plots/2_mood_trajectory_by_usage.png', dpi=300, bbox_inches='tight')
        plt.show()
        return

    _(event_df, pd, plt, sns)
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Lagged Mood Impact by App Category: This plot helps answer: "Does using a specific category of app (e.g., Social) lead to a positive or negative mood change in the next notification?"
    """)
    return


@app.cell
def _(event_df, plt, sns):

    def _(event_df, plt, sns):
        # Melt the dataframe to compare different app categories against mood change
        categories = ['use_social', 'use_communication', 'use_entertainment', 'use_productivity']
        plot_df = event_df.melt(
            id_vars=['mood_change_cat'], 
            value_vars=categories,
            var_name='App_Category', 
            value_name='Usage_Seconds'
        )
    
        # Convert to minutes and filter out outliers for better visualization
        plot_df['Usage_Minutes'] = plot_df['Usage_Seconds'] / 60
        plot_df = plot_df[plot_df['Usage_Minutes'] < plot_df['Usage_Minutes'].quantile(0.95)]

        plt.figure(figsize=(12, 6))
        sns.boxplot(data=plot_df, x='App_Category', y='Usage_Minutes', hue='mood_change_cat', palette='Set2')
        plt.title("Smartphone Usage Duration vs. Resulting Mood Change", fontsize=14)
        plt.ylabel("Minutes used between notifications")
        plt.xlabel("App Category")
        plt.legend(title="Mood Direction")
        plt.xticks(rotation=15)
        return plt.gca()

    _(event_df, plt, sns)
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Time-of-Day Intensity Heatmap
    """)
    return


@app.cell
def _(event_df, np, plt, sns):
    def _(event_df, plt, sns, np):
        # Pivot data to see average usage per hour per category
        usage_cols = [c for c in event_df.columns if c.startswith('use_')]
        hourly_usage = event_df.groupby('hour_curr')[usage_cols].mean()
    
        # Normalize by column (category) to see the relative peak hours
        hourly_norm = (hourly_usage - hourly_usage.min()) / (hourly_usage.max() - hourly_usage.min())

        plt.figure(figsize=(12, 8))
        sns.heatmap(hourly_norm.T, cmap="YlGnBu", annot=False)
        plt.title("Relative Peak Usage Hours by Category (Normalized)", fontsize=14)
        plt.xlabel("Hour of Day (24h)")
        plt.ylabel("App Category")
        return plt.gca()

    _(event_df, plt, sns, np)
    return


@app.cell
def _(event_df, pd, plt, sns):
    # =============================================================================
    # PLOT 3: Within-person correlations (all 8 categories)
    # =============================================================================

    def _(event_df, plt, sns, pd):
        """
        Calculate within-person correlations for ALL 8 app categories
        Shows heterogeneity across users
        """
        usage_cols = [col for col in event_df.columns if col.startswith('use_')]
    
        correlations = []
    
        for user in event_df['userid'].unique():
            user_data = event_df[event_df['userid'] == user]
            if len(user_data) > 15:  # Need sufficient data points
                try:
                    user_corrs = {'userid': user}
                    for col in usage_cols:
                        corr = user_data[['A6a', col]].corr().iloc[0, 1]
                        user_corrs[col] = corr
                    correlations.append(user_corrs)
                except:
                    pass
    
        corr_df = pd.DataFrame(correlations)
    
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
    
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#e67e22', '#34495e']
    
        for idx, (col, color) in enumerate(zip(sorted(usage_cols), colors)):
            cat_name = col.replace('use_', '').capitalize()
        
            axes[idx].hist(corr_df[col].dropna(), bins=30, edgecolor='black', alpha=0.7, color=color)
            axes[idx].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero')
            mean_val = corr_df[col].mean()
            axes[idx].axvline(mean_val, color='darkblue', linestyle='--', 
                             linewidth=2, label=f'Mean: {mean_val:.3f}')
            axes[idx].set_title(f'{cat_name}', fontweight='bold', fontsize=12)
            axes[idx].set_xlabel('Within-Person Correlation with Mood')
            axes[idx].set_ylabel('Number of Users')
            axes[idx].legend(fontsize=9)
            axes[idx].grid(alpha=0.3, axis='y')
        
            # Add percentage of positive/negative correlations
            pct_positive = (corr_df[col] > 0).sum() / len(corr_df[col].dropna()) * 100
            axes[idx].text(0.05, 0.95, f'{pct_positive:.0f}% positive', 
                          transform=axes[idx].transAxes, 
                          verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                          fontsize=9)
    
        plt.suptitle('Within-Person Correlations: App Usage and Mood\n(Heterogeneity Across Users)', 
                     fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.savefig('../plots/3_within_person_correlations_all.png', dpi=300, bbox_inches='tight')
        plt.show()
        return (corr_df,)

    _(event_df, plt, sns, pd)
    return


@app.cell
def _(event_df, pd, plt, sns):
    def _(event_df, plt, sns, pd):
        """
        Calculate within-person correlations for ALL 8 app categories
        Shows heterogeneity across users, including variance.
        """
        usage_cols = [col for col in event_df.columns if col.startswith('use_')]
    
        correlations = []
    
        for user in event_df['userid'].unique():
            user_data = event_df[event_df['userid'] == user]
            if len(user_data) > 15:  # Need sufficient data points
                try:
                    user_corrs = {'userid': user}
                    for col in usage_cols:
                        # Calculate correlation between Mood (A6a) and specific app category
                        corr = user_data[['A6a', col]].corr().iloc[0, 1]
                        user_corrs[col] = corr
                    correlations.append(user_corrs)
                except:
                    pass
    
        corr_df = pd.DataFrame(correlations)
    
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
    
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#e67e22', '#34495e']
    
        for idx, (col, color) in enumerate(zip(sorted(usage_cols), colors)):
            cat_name = col.replace('use_', '').capitalize()
        
            # Clean data for current category
            clean_corrs = corr_df[col].dropna()
        
            # Plot histogram
            axes[idx].hist(clean_corrs, bins=30, edgecolor='black', alpha=0.7, color=color)
        
            # Reference line at zero
            axes[idx].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero')
        
            # Calculate and plot Mean
            mean_val = clean_corrs.mean()
            # Calculate Variance
            var_val = clean_corrs.std()
        
            axes[idx].axvline(mean_val, color='darkblue', linestyle='--', 
                             linewidth=2, label=f'Mean: {mean_val:.3f}\nSd: {var_val:.3f}')
        
            axes[idx].set_title(f'{cat_name}', fontweight='bold', fontsize=12)
            axes[idx].set_xlabel('Within-Person Correlation with Mood')
            axes[idx].set_ylabel('Number of Users')
            axes[idx].legend(fontsize=9, loc='upper right')
            axes[idx].grid(alpha=0.3, axis='y')
        
            # Add percentage of positive correlations
            pct_positive = (clean_corrs > 0).sum() / len(clean_corrs) * 100
            axes[idx].text(0.05, 0.95, f'{pct_positive:.0f}% positive', 
                          transform=axes[idx].transAxes, 
                          verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                          fontsize=9)
    
        plt.suptitle('Within-Person Correlations: App Usage and Mood\n(Heterogeneity and Variance Across Users)', 
                     fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig('../plots/3_within_person_correlations_with_variance.png', dpi=300, bbox_inches='tight')
        plt.show()
        return (corr_df,)

    _(event_df, plt, sns, pd)
    return


@app.cell
def _(event_df, pd, plt, sns):
    # =============================================================================
    # PLOT 4: Mood change by app usage categories (top 4)
    # =============================================================================

    def _(event_df, pd, plt, sns):
        """
        Shows whether app usage predicts mood improvement/worsening
        """
        usage_cols = [col for col in event_df.columns if col.startswith('use_')]
    
        plot_df = event_df.copy()
    
        # Get top 4 categories
        category_totals = {col: plot_df[col].sum() for col in usage_cols}
        top_4_categories = sorted(category_totals.items(), key=lambda x: x[1], reverse=True)[:4]
        top_4_cols = [col for col, _ in top_4_categories]
    
        # Create usage bins (in seconds): None, Light (0-60s), Moderate (60-300s), Heavy (300+s)
        for cat in top_4_cols:
            plot_df[f'{cat}_category'] = pd.cut(
                plot_df[cat],
                bins=[-0.1, 0, 60, 300, 10000],
                labels=['None', 'Light\n(0-1min)', 'Moderate\n(1-5min)', 'Heavy\n(5+min)']
            )
    
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
        for idx, cat in enumerate(top_4_cols):
            cat_name = cat.replace('use_', '').capitalize()
        
            # Top row: Delta mood violin plots
            sns.violinplot(
                data=plot_df,
                x=f'{cat}_category',
                y='delta_mood',
                ax=axes[0, idx],
                palette='Set2'
            )
            axes[0, idx].set_title(f'{cat_name}: Mood Change', fontweight='bold')
            axes[0, idx].set_xlabel('')
            axes[0, idx].set_ylabel('Mood Change (Δ)')
            axes[0, idx].axhline(0, color='red', linestyle='--', alpha=0.5, linewidth=2)
            axes[0, idx].tick_params(axis='x', rotation=0)
        
            # Bottom row: Mood change categories (stacked bar)
            mood_change_summary = (
                plot_df.groupby(f'{cat}_category')['mood_change_cat']
                .value_counts(normalize=True)
                .unstack(fill_value=0)
            )
        
            # Reorder columns to: peggiora, uguale, migliora
            col_order = ['peggiora', 'uguale', 'migliora']
            mood_change_summary = mood_change_summary[[c for c in col_order if c in mood_change_summary.columns]]
        
            mood_change_summary.plot(
                kind='bar',
                stacked=True,
                ax=axes[1, idx],
                color=['#e74c3c', '#95a5a6', '#2ecc71']
            )
            axes[1, idx].set_title(f'{cat_name}: Mood Improvement Rate', fontweight='bold')
            axes[1, idx].set_xlabel('Usage Level')
            axes[1, idx].set_ylabel('Proportion')
            axes[1, idx].legend(title='Mood Change', labels=['Worsens', 'Same', 'Improves'], loc='best')
            axes[1, idx].tick_params(axis='x', rotation=0)
            axes[1, idx].set_ylim(0, 1)
    
        plt.suptitle('App Usage and Mood Changes', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig('../plots/4_mood_change_by_usage.png', dpi=300, bbox_inches='tight')
        plt.show()
        return

    _(event_df, pd, plt, sns)
    return


@app.cell
def _(event_df):
    event_df["userid"].unique()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 9 – Salvataggio

    Salviamo il dataset evento-per-evento in:

    `./data/processed/eventlevel_mood_phoneuse.parquet`
    """)
    return


@app.cell
def _(Path, event_df):
    processed_path = Path("../data/processed")
    processed_path.mkdir(exist_ok=True)

    out_file = processed_path / "eventlevel_mood_phoneuse.parquet"
    event_df.to_parquet(out_file, index=False)

    out_file
    return


if __name__ == "__main__":
    app.run()
