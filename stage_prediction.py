import numpy as np
import pandas as pd
from sklearn import svm
from helpers import clean_data, gather_data


def download_images(urls_list, tweet_ids):
    """Download all images in dataset"""

    for url, tweet_id in zip(urls_list, tweet_ids):
        gather_data.download_image(url, tweet_id)


def build_images(tweet_ids):
    """Get data (values of pixels) of each image"""

    images_data = []
    for tweet_id in tweet_ids:
        data = gather_data.build_image(f"resources/images/{tweet_id}.jpg")
        images_data.append(data)

    return images_data


def gather(df):
    """Collects images by downloading them and converting them into pixels data"""

    # TODO uncomment to re-download the images
    # download_images(list(df["image_url"]), list(df["tweet_id"]))

    data_images = {
        "tweet_id": list(df["tweet_id"]),
        "image_data": build_images(list(df["tweet_id"]))
    }
    df_images = pd.DataFrame(data_images)
    df = pd.merge(df, df_images)

    return df


def assess(df):
    """Describes dataset samples and features"""

    df.info()
    print(df.describe())
    print(df.head(5))


def clean(df):
    # Remove un-necessary column
    df = df.drop(columns="image_url")

    # Remove duplicated rows
    df = df.drop_duplicates()

    # Remove rows with missing values
    df_train = df.dropna()

    df_test = df.drop(columns="stage")
    df_test = df_test.dropna()

    return df_train, df_test


def store_data(df, file_name):
    """Stores cleaned datasets to a CSV file"""

    df.to_csv("resources/{}".format(file_name), index=False)


def classify(df_train, df_test):
    # TODO convert pixels to image features using opencv and image processing methods
    
    X_train = df_train["image_data"]
    X_test = df_test["image_data"]
    y_train = df_train["stage"]

    clf = svm.SVC()
    clf.fit(X_train, y_train)

    y_predicted = clf.predict(X_test)
    data_prediction = {
        "tweet_id": list(df_test["tweet_id"]),
        "stage": y_predicted
    }
    df_prediction = pd.DataFrame(data_prediction)
    df_test = pd.merge(df_test, df_prediction)

    return df_train.append(df_test)


def begin(df):
    print("_" * 40)
    print("\nRunning stage prediction.. Please wait")
    df = clean_data.get_some_columns(df, ["tweet_id", "image_url", "stage"])

    df = gather(df)

    store_data(df, "stage_prediction.csv")

    df_stage = gather_data.load_date("stage_prediction.csv", "default")
    assess(df_stage)
    df_train, df_test = clean(df_stage)
    # print("Train" + "_" * 40)
    # assess(df_train)
    # print("Test" + "_" * 40)
    # assess(df_test)

    store_data(classify(df_train, df_test), "test.csv")
