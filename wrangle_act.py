import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from helpers import clean_data, gather_data

DATA_NAMES = ["enhanced_tweets", "fetched_tweets", "image_predictions", "merged"]


def gather():
    """Collects data of tweets from both the enhanced CSV file and JSON file.
    Loads the downloaded image predictions dataset"""

    return gather_data.load_date("twitter_archive_enhanced.csv", "default"), \
           gather_data.tweets_json_to_dataframe(), \
           gather_data.download_image_predictions()


def assess(df):
    """Describes dataset samples and features"""

    df.info()
    print(df.describe())
    print(df.head(5))


def clean(df, data_name):
    if data_name == DATA_NAMES[0]:
        # Change data types of specific columns after filling NULL values
        df = clean_data.convert_column_to_datetime(df, "timestamp")
        df["in_reply_to_status_id"] = df["in_reply_to_status_id"].fillna(0).astype(int)
        df["in_reply_to_user_id"] = df["in_reply_to_user_id"].fillna(0).astype(int)
        df["retweeted_status_id"] = df["retweeted_status_id"].fillna(0).astype(int)
        df["retweeted_status_user_id"] = df["retweeted_status_user_id"].fillna(0).astype(int)
        df["retweeted_status_timestamp"] = df["retweeted_status_timestamp"].fillna(0)
        df = clean_data.convert_column_to_datetime(df, "retweeted_status_timestamp")

        # Create new column "stage" instead of 4 columns: doggo, floofer, pupper, puppo
        conditions = [df['doggo'] != "None",
                      df['floofer'] != "None",
                      df['pupper'] != "None",
                      df['puppo'] != "None"]
        choices = ["doggo", "floofer", "pupper", "puppo"]
        df['stage'] = np.select(conditions, choices, default=np.NaN)

        # Delete unwanted columns
        df = df.drop(columns=["doggo", "floofer", "pupper", "puppo"])

    if data_name == DATA_NAMES[1]:
        # Rename columns
        df = clean_data.rename_columns(df, {"id": "tweet_id"})

        # Change data types of specific columns after filling NULL values
        df = clean_data.convert_column_to_datetime(df, "created_at")
        df["in_reply_to_status_id"] = df["in_reply_to_status_id"].fillna(0.0).astype(int)
        df["in_reply_to_user_id"] = df["in_reply_to_user_id"].fillna(0.0).astype(int)
        df["possibly_sensitive"] = df["possibly_sensitive"].fillna("FALSE").astype(bool)
        df["possibly_sensitive_appealable"] = df["possibly_sensitive_appealable"].fillna("FALSE").astype(bool)

        # retweeted column has False value even when retweet_count > 0, so it should be fixed
        conditions = [df['retweet_count'] > 0]
        choices = ["TRUE"]
        df['retweeted'] = np.select(conditions, choices, default=np.False_)

        # favorited column has False value even when favorite_count > 0, so it should be fixed
        conditions = [df['favorite_count'] > 0]
        choices = ["TRUE"]
        df['favorited'] = np.select(conditions, choices, default=np.False_)

        # Delete unwanted columns (having too many missing values or useless)
        df = df.drop(
            columns=["id_str", "in_reply_to_status_id_str", "in_reply_to_user_id_str", "in_reply_to_screen_name",
                     "geo", "coordinates", "place", "contributors", "retweeted_status", "quoted_status_id",
                     "quoted_status_id_str", "quoted_status"])

    if data_name == DATA_NAMES[2]:
        # Rename columns
        df = clean_data.rename_columns(df, {"jpg_url": "image_url"})

        # Get only dogs and ignore non-dogs
        df = clean_data.filter_by_condition(df, df["p1_dog"] == True)

        # Remove p1_dog column because it is always True, useless column
        df = df.drop(columns=["p1_dog"])

    if data_name == DATA_NAMES[3]:
        df['year'] = df['created_at'].dt.year
        df = clean_data.rename_columns(df, {"p1": "species_prediction",
                                            "p1_conf": "prediction_conf"})

        # Delete unwanted columns (duplicated columns, columns with only one value)
        df = df.drop(columns=["timestamp", "truncated", "is_quote_status", "possibly_sensitive",
                              "possibly_sensitive_appealable"])

    # Remove rows with missing values
    df = df.dropna()

    return df


def merge_data(df1, df2, df3):
    """Merges tweets datasets and image predictions dataset according to "tweet_id"""

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

    # Create 2 subplots
    fig, axs1 = plt.subplots(ncols=2)

    # Plot top 5 dog species
    sns.countplot(x="species_prediction",
                  data=df,
                  order=df.species_prediction.value_counts().iloc[:5].index,
                  ax=axs1[0])
    axs1[0].set_xticklabels(axs1[0].get_xticklabels(), rotation=90)
    axs1[0].set_title("Top 5 Dog Species")
    axs1[0].set_xlabel("Species")
    axs1[0].set_ylabel("Count")

    # Plot rating for each dog species
    sns.barplot(x="species_prediction",
                y="rating_numerator",
                data=df,
                order=df.species_prediction.value_counts().iloc[:5].index,
                ax=axs1[1])
    axs1[1].set_title("Rating for Each Species")
    axs1[1].set_xticklabels(axs1[1].get_xticklabels(), rotation=90)
    axs1[1].set_xlabel("Species")
    axs1[1].set_ylabel("Rating")

    plt.tight_layout()

    # Create 2 subplots
    fig2, axs2 = plt.subplots(ncols=2)

    # Plot number of retweets per year
    sns.barplot(x="year",
                y="retweet_count",
                data=df,
                ax=axs2[0])
    axs2[0].set_title("Number of Retweets per Year")
    axs2[0].set_xlabel("Year")
    axs2[0].set_ylabel("Retweet Count")

    # Plot number of favorites for each dog stage
    sns.barplot(x="stage",
                y="favorite_count",
                data=df,
                ax=axs2[1])
    axs2[1].set_title("Favourite Count for Each Stage")
    axs2[1].set_xlabel("Stage")
    axs2[1].set_ylabel("Favorite Count")

    # Explore correlation between variables
    # +ve correlation means the variable increases if the other variable increases
    # -ve correlation means the variable decreases if the other variable increases and vice versa
    # sns.heatmap(df.corr(),
    #             annot=True,
    #             ax=axs[0, 0])
    # axs[0, 0].set_title("Correlation Matrix")

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    print("\nPerforming data wrangling.. Please wait")

    df_enhanced_tweets, df_fetched_tweets, df_predictions = gather()

    # assess(df_enhanced_tweets)
    # assess(df_fetched_tweets)
    # assess(df_predictions)

    df_enhanced_tweets_cleaned = clean(df_enhanced_tweets, DATA_NAMES[0])
    # assess(df_enhanced_tweets_cleaned)

    df_fetched_tweets_cleaned = clean(df_fetched_tweets, DATA_NAMES[1])
    # assess(df_fetched_tweets_cleaned)

    df_predictions_cleaned = clean(df_predictions, DATA_NAMES[2])
    # assess(df_predictions_cleaned)

    df_merged = merge_data(df_fetched_tweets_cleaned,
                           df_enhanced_tweets_cleaned,
                           df_predictions_cleaned)
    df_merged_cleaned = clean(df_merged, DATA_NAMES[3])
    # assess(df_merged_cleaned)

    store_data(df_merged_cleaned, "twitter_archive_master.csv")

    explore_data(df_merged_cleaned)
