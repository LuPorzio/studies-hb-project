import marimo

__generated_with = "0.17.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import altair as alt
    return alt, mo, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Exploratory Data Analysis
    This notebook is devoted to exploring the data and making sense of the content of each dataset outlining the main differences between each of them.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## appuseIT_class_17_11.parquet
    """)
    return


@app.cell
def _(pd):
    # read the dataset
    appuse_df = pd.read_parquet('./data/appuseIT_class_17_11.parquet')

    # obtain info about the columns datatypes
    appuse_df.info()
    return (appuse_df,)


@app.cell
def _(appuse_df):
    appuse_df.isna().sum()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    From the output above we can see that the majority of datatypes is coherent with what they aim to represent. There are some missing data for the category, free, and content rating columns. Now we have a look at a preview of the data to check the first 10 lines of the dataset.
    """)
    return


@app.cell
def _(appuse_df):
    appuse_df["App Name"] = appuse_df["App Name"].astype(str)
    return


@app.cell
def _(appuse_df):
    appuse_df.head(8000).sort_values("tot_count", ascending = True)
    return


@app.cell
def _(mo):
    mo.md(r"""
    We notice that there is an `experimentid` listed as one of the dimensions. To understand a little bit more about the dataset we can produce some summary statistics.
    """)
    return


@app.cell
def _(appuse_df):
    appuse_df.describe().T
    return


@app.cell
def _(mo):
    mo.md(r"""
    We can summarize the (relevant) findings as follows:
    1. the experiment ran from 12-11-2025 to 12-12-2025
    2. `del_app` is a effectively a constant in this sample ($s$ = 0)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We now try to understand what exactly `tot_count` refers to.
    """)
    return


@app.cell
def _(appuse_df, pd):
    # extract date without time component from data
    appuse_df["day_month"] = appuse_df["timestamp"].apply(lambda x: pd.to_datetime(x).date())
    return


@app.cell
def _(appuse_df):
    # aggregate by userid, applicationname, day_month and get the number of observations in
    # each group
    appuse_df.groupby(["userid", "applicationname", "day_month"]).agg(tot_count = ("timestamp", "size"))\
    .reset_index().query("applicationname == 'android'")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    After many experiments it is still unclear how `tot_count` was computed. We suspect it is computed as explained in the cell above but the result does not always apply.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Cleaning Dataset
    We now proceed to deleting the observations that are not needed and perform some other cleaning procedures.
    """)
    return


@app.cell
def _(appuse_df):
    # remove the unnecessary observations related to system-level automatic processes
    appuse_df_cleaned = appuse_df.query("applicationname != 'android'")

    # sanity check
    len(appuse_df_cleaned.query("applicationname == 'android'"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## app4regress_IT_new_v2.parquet
    Same process as above.
    """)
    return


@app.cell
def _(pd):
    # read data
    appregress_df = pd.read_parquet("./data/app4regress_IT_new_v2.parquet")

    # get info about dataset
    appregress_df.info()
    return (appregress_df,)


@app.cell
def _(appregress_df):
    # check which cols have missing data
    appregress_df.isna().any()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The columns with missing data are:
    1. A4
    2. mood_std
    3. Neuroticism
    """)
    return


@app.cell
def _(appregress_df, mo):
    user_filter = mo.ui.dropdown(appregress_df["userid"])
    user_filter
    return (user_filter,)


@app.cell
def _(appregress_df, user_filter):
    appregress_df[["userid", "timestamp", "A6a", "delta_mood1"]]\
        .query(f"userid == {user_filter.value}").head(100)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Important**: from the exploration above, we now know that the `delta_mood1` column contains the difference in mood between each consecutive measurements taken for each individual. So there is no need to compute it for future applications.
    """)
    return


@app.cell
def _(appregress_df):
    appregress_df.describe().T
    return


@app.cell
def _(appregress_df):
    # cohort contains info about the age of users divided in groups
    appregress_df['cohort'].unique()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We conclude this dataset contains aggregated data from the experiment that has been already elaborated for modeling. This is a step beyond the raw data as is.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## td_ita.csv (Time Diary)
    """)
    return


@app.cell
def _(pd):
    td_df = pd.read_csv('./data/td_ita.csv', low_memory=False)

    td_df.info()
    return (td_df,)


@app.cell
def _(td_df):
    td_df.iloc[:,:]
    return


@app.cell
def _(mo):
    mo.md(r"""
    1. date_not: the timestamp when the notification was received.
    2. datein_ques: the timestamp when the question arrives.
    3. datein_answ: the timestamp when the question is answered.

    We can check how much time passes from receiving the notification and answering.
    """)
    return


@app.cell
def _(pd, td_df):
    # convert to datetime to analyze properly
    td_df['date_not'] = pd.to_datetime(td_df['date_not'])
    td_df['datein_ques'] = pd.to_datetime(td_df['datein_ques'])
    td_df['datein_answ'] = pd.to_datetime(td_df['datein_answ'])
    return


@app.cell
def _(td_df):
    # number of complete observations from the night hours
    # check how the number of 'no information' changes throughout the day
    plot_no_info = td_df.query("what == 'No information'")\
        .groupby(['hh_not'])\
        .agg(aggregated_what=('what','size')).reset_index()
    return (plot_no_info,)


@app.cell
def _(alt, plot_no_info):
    # replace _df with your data source
    _chart = (
        alt.Chart(plot_no_info)
        .mark_bar()
        .encode(
            x=alt.X(field='hh_not', type='quantitative', title='Hour'),
            y=alt.Y(field='aggregated_what', type='quantitative', title='# No information'),
            tooltip=[
                alt.Tooltip(field='hh_not', format=',.0f', title='Hour'),
                alt.Tooltip(field='aggregated_what', format=',.0f', title='# No information')
            ]
        )
        .properties(
            title='No information over days',
            height=290,
            width='container',
            config={
                'axis': {
                    'grid': False
                }
            }
        )
    )
    _chart
    return


@app.cell
def _(mo):
    mo.md(r"""
    We have considerably more information starting from 9 until around 23. Based on what discovered, we can drop some data to avoid including non informative units in the analysis.
    """)
    return


@app.cell
def _(td_df):
    # remove observations from 1 am to 5 am
    td_df_cleaned = td_df.set_index('date_not').between_time(start_time='5:00am', end_time='1:00am').reset_index()
    td_df_cleaned.head(20)
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    So far we have cleaned the data performing these steps:
    1. removed observations that are between 1am and 5am
    """)
    return


if __name__ == "__main__":
    app.run()
