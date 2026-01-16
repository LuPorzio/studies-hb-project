import marimo

__generated_with = "0.19.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    from pathlib import Path
    import matplotlib.pyplot as plt
    import seaborn as sns
    return Path, mo, pd, plt


@app.cell
def _(mo):
    mo.md(r"""
    # Event-level dataset: mood & smartphone use

    Obiettivo:
    - Costruire un dataset a livello **evento di mood**:
      - mood attuale (`A6a`)
      - mood precedente
      - cambiamento di mood (`delta_mood`) e categoria (migliora/uguale/peggiora)
    - Ricostruire l’uso del telefono tra una notifica e la successiva:
      - tempo di uso (o numero di ping) per:
        - app **social**
        - app di **comunicazione**
        - **altre** app
    - Tenere solo utenti **validi** (secondo le soglie definite sul time diary)
    - Aggiungere variabili socio-demografiche.

    L’output finale viene salvato in:
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

    df_mood_latent = pd.concat(dfs, axis=0)
    return (df_mood_latent,)


@app.cell
def _(df_mood_latent):
    df_mood_latent # use this dataset for all later use of the 
    return


@app.cell
def _(mood_df_sorted):
    mood_df_sorted["latent_prev"] = mood_df_sorted["latent_var"].shift(1)
    mood_df_sorted["latent_change"] = mood_df_sorted["latent_var"] - mood_df_sorted["latent_prev"]
    mood_df_sorted.head(100)
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

    # soglie per periodo: riprese dallo script del time diary
    def is_valid_user(row):
        if row["first2w"] == "First two weeks":
            return (row["median_valid_per_day"] >= 25) and (row["days_with_valid"] >= 14)
        else:  # Second two weeks
            return (row["median_valid_per_day"] >= 12) and (row["days_with_valid"] >= 7)

    participation_summary["valid_user_period"] = participation_summary.apply(is_valid_user, axis=1)

    # validità complessiva: utente valido almeno in un periodo
    valid_overall = (
        participation_summary
        .groupby("id", as_index=False)["valid_user_period"]
        .any()   #true se l’utente è valido in almeno uno dei due periodifalse solo se è NON valido in entrambi.
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
def _(mood_df, valid_users_with_phone):
    # CHANGED ON 05/01
    # rinominiamo id -> userid per fare merge con appregress
    valid_users = valid_users_with_phone.rename(columns={"id": "userid"})

    # teniamo solo utenti con valid_user_overall == True
    valid_users = valid_users_with_phone.query("valid_user_overall == True")

    mood_valid_df = mood_df.merge(
        valid_users_with_phone[["userid", "valid_user_overall"]],
        on="userid",
        how="inner",
    )

    mood_valid_df.head(100000), valid_users["userid"].nunique()
    return (mood_valid_df,)


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
def _(mood_valid_df):
    # partiamo da mood_valid_df (creato nella cella precedente)
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
    ]].head(30)
    return (mood_interval_df,)


@app.cell
def _(mood_interval_df):
    # aggiungiamo la durata dell'intervallo tra due EMA

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
    ]].head(20)
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
    #Preparazione app use:
    - togliamo 'android' (processo di sistema)
    - convertiamo timestamp in datetime
    - creiamo la macro-categoria:
      * communication: chat, chiamate, email
      * social: social network, tiktok, ecc.
      * other: tutto il resto
    "\"\"
    """)
    return


@app.cell
def _(appuse_df, plt):
    # Extract the complete list of unique app names
    # Based on the file structure, the column is typically named 'App' or 'applicationname'
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
def _(appuse_df, pd):
    """
    Cella 5 – Preparazione app use:
    - togliamo 'android' (processo di sistema)
    - convertiamo timestamp in datetime
    - creiamo la macro-categoria:
      * communication: chat, chiamate, email
      * social: social network, tiktok, ecc.
      * other: tutto il resto
    """

    # 1) filtro di base: rimuoviamo il package 'android'
    appuse_prepped = appuse_df.query("applicationname != 'android'").copy()
    appuse_prepped["to_keep"] = ~(appuse_df["applicationname"].str.contains("launcher"))
    appuse_prepped = appuse_prepped.loc[appuse_prepped["to_keep"]]

    # 2) timestamp a datetime
    appuse_prepped["timestamp"] = pd.to_datetime(appuse_prepped["timestamp"])

    # 3) funzione per macro-categoria
    def map_macro(pkg: str) -> str:
        p = str(pkg).lower()

        # --- communication (chat, telefonate, email) ---
        comm_keywords = [
            "whatsapp",          # com.whatsapp
            "telegram",          # org.telegram.messenger
            "messenger",         # eventuali messenger
            "android.gm",        # gmail (package)
            "gmail",
            "mail",
            "incallui",          # interfaccia chiamate
            "dialer",
            "sms",
            "phone",             # com.samsung.android.phone ecc.
        ]
        if any(k in p for k in comm_keywords):
            return "communication"

        # --- social (social network, tiktok, ecc.) ---
        social_keywords = [
            "instagram",         # com.instagram.android
            "facebook.katana",   # com.facebook.katana
            "facebook",
            "twitter",           # com.twitter.android
            "musically",         # tiktok
            "tiktok",
            "snapchat",
            "reddit",
        ]
        if any(k in p for k in social_keywords):
            return "social"

        # --- tutto il resto: launcher, sistema, browser, media, ecc. ---
        return "other"

    # 4) applica macro-categoria
    appuse_prepped["macro_cat"] = appuse_prepped["applicationname"].apply(map_macro)

    # controllo veloce (per vedere se ha senso)
    appuse_prepped[["applicationname", "macro_cat"]].drop_duplicates().head(30)
    return (appuse_prepped,)


@app.cell
def _():
    """
    Cella 5 – Preparazione app use:
    - togliamo 'android' (processo di sistema)
    - convertiamo timestamp in datetime
    - creiamo la macro-categoria:
      * communication: chat, chiamate, email
      * social: social network, tiktok, ecc.
      * other: tutto il resto
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
def _(appuse_prepped, mood_interval_df, pd):
    # 6 – Uso del telefono tra due notifiche
    # Obiettivo: per ogni ping di app capire in quale intervallo [t_prev, t_curr) cade.

    # Copie di lavoro
    appuse_for_merge = appuse_prepped.copy()
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
        "macro_cat",
        "event_id",
        "t_prev",
        "t_curr",
    ]].head(20)
    return (merged,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 7 – Aggregazione: social / communication / other per evento

    Per ogni coppia (`userid`, `event_id`) calcoliamo:
    - numero di ping (o unità di tempo) per `macro_cat`
    - e trasformiamo in formato wide:
      - `use_social`
      - `use_communication`
      - `use_other`
    """)
    return


@app.cell
def _(merged):
    # 7 – Aggregazione: social / communication / other per evento

    # per sicurezza
    merged["event_id"] = merged["event_id"].astype(int)

    # tempo totale di uso per intervallo *categoria*
    usage_agg = (
        merged
        .groupby(["userid", "event_id", "macro_cat"], as_index=False)
        ["time_use_second_app"]
        .sum()
        .rename(columns={"time_use_second_app": "time_use_sec"})
    )

    # passiamo a formato wide: una colonna per categoria
    usage_wide = (
        usage_agg
        .pivot(index=["userid", "event_id"], columns="macro_cat", values="time_use_sec")
        .fillna(0)
        .reset_index()
    )

    # garantiamo che le tre colonne esistano sempre
    for cat in ["social", "communication", "other"]:
        if cat not in usage_wide.columns:
            usage_wide[cat] = 0

    # rinominiamo le colonne come "use_*"
    usage_wide = usage_wide.rename(
        columns={
            "social": "use_social",
            "communication": "use_communication",
            "other": "use_other",
        }
    )

    usage_wide.head(20)
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
def _(mood_interval_df, socio_df_clean, usage_wide):
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
    for var in ["use_social", "use_communication", "use_other"]:
        event_df[var] = event_df[var].fillna(0)


    # aggiungiamo ora e fascia oraria dell’evento (t_curr)
    event_df["hour_curr"] = event_df["t_curr"].dt.hour
    event_df["time_of_day"] = event_df["hour_curr"].apply(time_of_day)

    # selezione di alcune variabili socio-demo (usa quelle che esistono davvero in socio_df_clean)
    socio_cols = [
        "userid",
        "gender",
        "degree",
        "department",
        "cohort_group",
    ]
    socio_small = socio_df_clean[[c for c in socio_cols if c in socio_df_clean.columns]].copy()

    event_df = event_df.merge(socio_small, on="userid", how="left")

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
