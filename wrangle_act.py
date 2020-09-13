import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from helpers import clean_data, gather_data

DATA_NAMES = ["archived_tweets", "fetched_tweets", "image_predictions", "merged"]


def gather():
    """Gathers data of tweets from both the archived CSV file and JSON file.
    Loads the downloaded image predictions dataset"""

    return gather_data.load_date("resources/twitter_archive_enhanced.csv", ","), \
           gather_data.tweets_json_to_dataframe(), \
           gather_data.download_image_predictions()


def assess(df):
    """Describes dataset samples and features"""

    df.info()
    print(df.describe())
    print(df.head(5))


def clean(df, data_name):
    if data_name == DATA_NAMES[0]:
        # Convert "timestamp" and "retweeted_status_timestamp" column from string to datetime
        df = clean_data.convert_column_to_datetime(df, "timestamp")
        df = clean_data.convert_column_to_datetime(df, "retweeted_status_timestamp")

        # Create new column "stage" instead of 4 columns: doggo, floofer, pupper, puppo
        conditions = [df['doggo'] != "None",
                      df['floofer'] != "None",
                      df['pupper'] != "None",
                      df['puppo'] != "None"]
        choices = ["doggo", "floofer", "pupper", "puppo"]
        df['stage'] = np.select(conditions, choices, default=np.NaN)

        # Get specific columns from the dataset and ignore the rest
        df = clean_data.get_some_columns(df, ["tweet_id", "stage"])

    if data_name == DATA_NAMES[1]:
        # Get specific columns from the dataset and ignore the rest
        df = clean_data.get_some_columns(df, ["created_at", "id", "retweet_count", "favorite_count"])

        # Rename columns
        df = clean_data.rename_columns(df, {"id": "tweet_id"})

        # Convert "created_at" column from string to datetime
        df = clean_data.convert_column_to_datetime(df, "created_at")

    if data_name == DATA_NAMES[2]:
        # Get specific columns from the dataset and ignore the rest
        df = clean_data.get_some_columns(df, ["tweet_id", "jpg_url", "p1", "p1_conf", "p1_dog"])

        # Get only dogs and ignore non-dogs
        df = clean_data.filter_by_condition(df, df["p1_dog"] == True)

        # Remove p1_dog column because it is always True, useless column
        df = df.drop(columns=["p1_dog"])

    if data_name == DATA_NAMES[3]:
        df['year'] = df['created_at'].dt.year
        df = clean_data.rename_columns(df, {"p1": "species_prediction",
                                            "p1_conf": "prediction_conf"})

    # Remove duplicated rows
    df = df.drop_duplicates()

    # Remove rows with missing values
    df = df.dropna()

    return df


def merge_data(df1, df2, df3):
    """Merges tweets datasets and image_predictions dataset according to "tweet_id"""

    merged = pd.merge(df1, df2)
    return pd.merge(merged, df3)


def store_data(df, file_name):
    """Stores cleaned datasets to a CSV file"""

    df.to_csv("resources/{}".format(file_name), index=False)


def explore_data(df):
    """Visualizes and analyzes some statistics"""

    # Print most common dog species in 2015, 2016, 2017
    most_popular_dog_species = df.groupby(['year'])['species_prediction'].agg(pd.Series.mode)
    print("_" * 40)
    print("\nMost Common Dog Species per Year:")
    print(most_popular_dog_species)

    # Print most common dog stage in 2015, 2016, 2017
    df_copy = clean_data.filter_by_condition(df, df['stage'] != "nan")
    most_popular_dog_stage = df_copy.groupby(['year'])['stage'].agg(pd.Series.mode)
    print("_" * 40)
    print("\nMost Common Dog Stage per Year:")
    print(most_popular_dog_stage)

    # Create 4 subplots
    fig, axs = plt.subplots(nrows=2, ncols=2)

    # Explore correlation between variables
    # +ve correlation means the variable increases if the other variable increases
    # -ve correlation means the variable decreases if the other variable increases and vice versa
    sns.heatmap(df.corr(),
                annot=True,
                ax=axs[0, 0])
    axs[0, 0].set_title("Correlation Matrix")

    # Plot top 5 dog species
    sns.countplot(x="species_prediction",
                  data=df,
                  order=df.species_prediction.value_counts().iloc[:5].index,
                  ax=axs[0, 1])
    axs[0, 1].set_xticklabels(axs[0, 1].get_xticklabels(), rotation=90)
    axs[0, 1].set_title("Top 5 Dog Species")
    axs[0, 1].set_xlabel("Species")
    axs[0, 1].set_ylabel("Count")

    # Plot number of retweets per year
    sns.barplot(x="year",
                y="retweet_count",
                data=df,
                ax=axs[1, 0])
    axs[1, 0].set_title("Number of Retweets per Year")
    axs[1, 0].set_xlabel("Year")
    axs[1, 0].set_ylabel("Retweet Count")

    # Plot number of favorites for each dog stage
    sns.barplot(x="stage",
                y="favorite_count",
                data=df,
                ax=axs[1, 1])
    axs[1, 1].set_title("Favourite Count for Each Stage")
    axs[1, 1].set_xlabel("Stage")
    axs[1, 1].set_ylabel("Favorite Count")

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    df_archived_tweets, df_fetched_tweets, df_predictions = gather()

    # assess(df_archived_tweets)
    # assess(df_fetched_tweets)
    # assess(df_predictions)

    df_archived_tweets = clean(df_archived_tweets, DATA_NAMES[0])
    # assess(df_archived_tweets)

    df_fetched_tweets = clean(df_fetched_tweets, DATA_NAMES[1])
    # assess(df_fetched_tweets)

    df_predictions = clean(df_predictions, DATA_NAMES[2])
    # assess(df_predictions)

    df_merged = merge_data(df_fetched_tweets, df_archived_tweets, df_predictions)
    df_merged = clean(df_merged, DATA_NAMES[3])
    # assess(df_merged)

    store_data(df_merged, "twitter_archive_master.csv")

    explore_data(df_merged)
