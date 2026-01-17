import marimo

__generated_with = "0.19.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    import statsmodels.formula.api as smf
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from scipy.stats import entropy
    from pathlib import Path
    return PCA, Path, StandardScaler, entropy, mo, np, pd, plt, smf, sns


@app.cell
def _(Path, entropy, pd):
    def _(Path, entropy, pd):
        # --- 1. DATA PREPARATION & FEATURE ENGINEERING ---
        # Loading the provided parquet dataset [cite: 90, 91]
        data_path = Path("../data/processed/eventlevel_mood_phoneuse.parquet")
        df = pd.read_parquet(data_path)

        # Explicitly defining the app categories present in the dataset schema 
        categories = [
            'communication', 'entertainment', 'finance', 'gaming', 
            'news', 'productivity', 'social', 'travel'
        ]
        usage_cols = [f'use_{c}' for c in categories]

        # Convert seconds to minutes for interpretability
        for col in usage_cols:
            # Ensure numeric conversion and fill NaNs with 0 [cite: 91, 92]
            df[f"{col}_min"] = pd.to_numeric(df[col], errors='coerce').fillna(0) / 60

        # A. Calculate Usage Diversity (Shannon Entropy)
        def calc_entropy(row):
            # Select the columns
            subset = row[[f"{c}_min" for c in usage_cols]]
        
            # Convert to numeric (errors='coerce' turns strings/garbage into NaN)
            # then drop NaNs
            counts = pd.to_numeric(subset, errors='coerce').dropna().values
        
            # Check if we have data left and if the sum is greater than 0
            if len(counts) == 0 or counts.sum() <= 0: 
                return 0
            
            return entropy(counts)

        # Force all usage columns to be floats, filling missing values with 0
        for col in [f"{c}_min" for c in usage_cols]:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        df['usage_diversity'] = df.apply(calc_entropy, axis=1)

        # B. Within-Person Centering
        # Creating 'State' (centered) and 'Trait' (person mean) variables for the model
        for col in [f"{c}_min" for c in usage_cols] + ['usage_diversity']:
            df[f'{col}_person_mean'] = df.groupby('userid')[col].transform('mean')
            df[f'{col}_centered'] = df[col] - df[f'{col}_person_mean']
    
        return df, usage_cols
    df, usage_cols = _(Path, entropy, pd)
    return df, usage_cols


@app.cell
def _(PCA, StandardScaler, df, pd, usage_cols):
    def _(PCA, StandardScaler, df, pd, usage_cols, n_components = 3):
        # --- 2. PCA: DISCOVERING USAGE PROFILES ---
        # Aggregate usage per user for the PCA
        pca_data = df.groupby('userid')[[f"{c}_min" for c in usage_cols]].mean()
    
        scaler = StandardScaler()
        pca_scaled = scaler.fit_transform(pca_data)

        pca = PCA(n_components=n_components)
        pca_results = pca.fit_transform(pca_scaled)
        pca_cols = [f'PCA_Profile_{i+1}' for i in range(n_components)]
        pca_df = pd.DataFrame(pca_results, columns=pca_cols, index=pca_data.index)

        # Merge PCA profiles back to the main dataframe
        df2 = df.merge(pca_df, left_on='userid', right_index=True)

        return pca, df2

    pca, df2 = _(PCA, StandardScaler, df, pd, usage_cols, 3)
    return df2, pca


@app.cell
def _(df2):
    df2.head(100)
    return


@app.cell
def _():
    # cols_to_shift = ["use_communication", "use_entertainment", "use_finance", "use_gaming", "use_news", 
    # "use_productivity", "use_social", "use_travel"]
    # df_lagged = df2.groupby('userid')[cols_to_shift].shift(1).add_suffix('_prev')
    # df_lagged = pd.concat([df2, df_lagged], axis = 1)
    # len(df_lagged.columns)
    return


@app.cell
def _():
    # # 1. Use the Minute-scale columns (consistent with your other analysis)
    # categories = [
    #     "communication", "entertainment", "finance", "gaming", 
    #     "news", "productivity", "social", "travel"
    # ]
    # # Use the _min version you created earlier
    # cols_shifted = [f"use_{c}_min" for c in categories]

    # # 2. Perform the Shift
    # # We rename them to '_prev' to distinguish them from current usage
    # lagged_features = df2.groupby('userid')[cols_shifted].shift(1)
    # lagged_features.columns = [c.replace('_min', '_prev') for c in cols_shifted]

    # # 3. Concatenate
    # df_lagged2 = pd.concat([df2, lagged_features], axis=1)

    # # Check results
    # print(f"Added {len(lagged_features.columns)} lagged columns.")
    # print(df_lagged2[['userid', 'A6a', 'use_social_min', 'use_social_prev']].head())
    return


@app.cell
def _(df_lagged):
    print(df_lagged.columns)
    df_lagged.head()
    # df_lagged[["use_communication_prev", "use_entertainment_prev"]].head()
    return


@app.cell
def _(df2, smf):
    def _(df2, smf):
        # --- 3. ADVANCED MULTILEVEL MODEL ---
        # Target variable 'A6a' represents mood 
        # Fixed Effects include within-person use, between-person habits, and controls 
        formula = """
        A6a ~ use_social_min_centered + use_communication_min_centered + 
              use_productivity_min_centered + usage_diversity_centered +
              use_social_min_person_mean + use_productivity_min_person_mean +
              PCA_Profile_1 + PCA_Profile_2 + 
              C(time_of_day) + C(gender)
        """

        # Removing rows with missing values in the specific model variables to ensure fit 
        model_df = df2.dropna(subset=['A6a', 'time_of_day', 'gender', 'PCA_Profile_1'])

        # Fit the mixed linear model with userid as the grouping factor
        model = smf.mixedlm(formula, model_df, groups=model_df["userid"]).fit()
        return (model,)

    fitted_model, = _(df2, smf)
    fitted_model.summary()
    return (fitted_model,)


@app.cell
def _():
    # def _(df_lagged2, smf):
    #     # --- 3. ADVANCED MULTILEVEL MODEL ---
    #     # Target variable 'A6a' represents mood 
    #     # Fixed Effects include within-person use, between-person habits, and controls 
    #     formula = """
    #     A6a ~ use_communication_prev + use_entertainment_prev + use_finance_prev +
    #     use_gaming_prev + use_news_prev + 
    #     use_productivity_prev + use_social_prev + use_travel_prev +
    #           PCA_Profile_1 + PCA_Profile_2 + 
    #           C(time_of_day) + C(gender)
    #     """

    #     # Removing rows with missing values in the specific model variables to ensure fit 
    #     model_df = df_lagged2.dropna(subset=['A6a', 'time_of_day', 'gender', 'PCA_Profile_1']).reset_index(drop=True)

    #     # Fit the mixed linear model with userid as the grouping factor
    #     model = smf.mixedlm(formula, model_df, groups=model_df["userid"]).fit()
    #     return (model,)

    # fitted_model2, = _(df_lagged2, smf)
    # fitted_model2.summary()
    return


@app.cell
def _(fitted_model, pca, pd, plt, sns, usage_cols):

    def _(fitted_model, pca, pd, plt, sns, usage_cols):
        # --- 4. VISUALIZATION ---

        def plot_analysis_results(fitted_model, pca, usage_cols):
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))

            # Plot A: PCA Loadings
            loadings = pd.DataFrame(
                pca.components_.T, 
                columns=['Profile 1', 'Profile 2', 'Profile 3'], 
                index=[c.replace('use_', '').replace('_min', '') for c in usage_cols]
            )
            sns.heatmap(loadings, annot=True, cmap='RdBu_r', ax=axes[0])
            axes[0].set_title("App Routine PCA Profiles (Loadings)")

            # Plot B: Model Coefficients
            # Extracting coefficients, excluding the Group Var
            results_df = fitted_model.summary().tables[1].iloc[:-1] 
            results_df = results_df.astype(float)
            results_df['feat'] = results_df.index

            sns.barplot(data=results_df, x='Coef.', y='feat', ax=axes[1], palette='vlag')
            axes[1].axvline(0, color='black', lw=1)
            axes[1].set_title("Impact on Mood (Multilevel Model Coefficients)")

            plt.tight_layout()
            plt.show()

        plot_analysis_results(fitted_model, pca, usage_cols)
        print(fitted_model.summary())
        return

    _(fitted_model, pca, pd, plt, sns, usage_cols)
    return


@app.cell
def _(mo):
    mo.md(r"""
    # "Is using social media at night different for your mood than using it in the morning?"
    """)
    return


@app.cell
def _(df2, smf):
    def _(df2, smf):
        # --- 5. INTERACTION ANALYSIS: CONTEXT MATTERS ---
        # We add an interaction term (*) between Social Usage and Time of Day
        # This tests: "Does the effect of social media on mood change depending on if it's Morning, Afternoon, or Night?"
    
        formula_interaction = """
        A6a ~ use_social_min_centered * C(time_of_day) + 
              use_productivity_min_centered + usage_diversity_centered +
              use_social_min_person_mean + 
              PCA_Profile_1 + PCA_Profile_2 + PCA_Profile_3 +
              C(gender)
        """

        # Removing rows with missing values
        model_df_int = df2.dropna(subset=['A6a', 'time_of_day', 'gender', 'PCA_Profile_1'])

        # Fit the interaction model
        model_interaction = smf.mixedlm(formula_interaction, model_df_int, groups=model_df_int["userid"]).fit()
    
        print("--- Interaction Model Summary ---")
        print(model_interaction.summary())
        return model_interaction,

    _(df2, smf)
    return


@app.cell
def _(df2, np, plt, sns, usage_cols):
    def _(df2, plt, sns, usage_cols):
        # --- 6. ADDITIONAL VISUALIZATIONS ---
    
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(2, 2)

        # 1. Diurnal Mood Curve (Top Left)
        # Visualizing the "Time of Day" effect found in your model
        ax1 = fig.add_subplot(gs[0, 0])
        sns.lineplot(data=df2, x='hour_curr', y='A6a', errorbar='ci', color='navy', ax=ax1)
        ax1.set_title("Average Mood (A6a) by Hour of Day", fontsize=14)
        ax1.set_xlabel("Hour (0-24)")
        ax1.set_ylabel("Reported Mood (A6a)")
        ax1.grid(True, alpha=0.3)

        # 2. Contextual Effect: Social Media vs Mood by Time of Day (Top Right)
        # Do we see different slopes for different times?
        ax2 = fig.add_subplot(gs[0, 1])
        # We use the centered variable to match the model logic
        sns.regplot(
            data=df2[df2['time_of_day'] == '1. Morning (5-12)'], 
            x='use_social_min_centered', y='A6a', 
            scatter=False, label='Morning', color='red', ax=ax2
        )
        sns.regplot(
            data=df2[df2['time_of_day'] == '4. Night (0-1)'], 
            x='use_social_min_centered', y='A6a', 
            scatter=False, label='Night', color='blue', ax=ax2
        )
        ax2.set_title("Impact of Social Media on Mood: Morning vs. Night", fontsize=14)
        ax2.set_xlabel("Social Media Use (Centered Minutes)")
        ax2.set_ylabel("Predicted Mood")
        ax2.legend()
        ax2.set_xlim(-10, 30) # Limit x-axis to zoom in on typical usage

        # 3. Correlation Heatmap (Bottom)
        # Overview of how all raw usage categories correlate with Mood
        ax3 = fig.add_subplot(gs[1, :])
    
        # Select columns to correlate
        cols_to_corr = ['A6a', 'usage_diversity'] + [f"{c}_min" for c in usage_cols]
        corr_matrix = df2[cols_to_corr].corr()
    
        # Mask the upper triangle
        mask = np.zeros_like(corr_matrix, dtype=bool)
        mask[np.triu_indices_from(mask)] = True
    
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', vmin=-0.1, vmax=0.1, ax=ax3)
        ax3.set_title("Raw Correlation Matrix: App Categories vs. Mood", fontsize=14)

        plt.tight_layout()
        plt.show()
        return

    _(df2, plt, sns, usage_cols)
    return


if __name__ == "__main__":
    app.run()
