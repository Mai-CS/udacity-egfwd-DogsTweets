import pandas as pd
import json
import requests


def load_date(file_name, separator):
    if separator == "default":
        separator = ','

    return pd.read_csv("resources/{}".format(file_name), sep=separator)


def tweets_json_to_dataframe():
    data = []
    with open('resources/tweets_json.txt', 'r') as f:
        for line in f:
            data.append(json.loads(line))
        return pd.DataFrame.from_records(data)


def download_image_predictions():
    try:
        url = "https://d17h27t6h515a5.cloudfront.net/topher/2017/August/599fd2ad_image-predictions/" \
              "image-predictions.tsv"
        response = requests.get(url)

        with open("resources/image_predictions.txt", 'wb') as f:
            f.write(response.content)

    finally:
        return load_date("image_predictions.txt", "\t")
