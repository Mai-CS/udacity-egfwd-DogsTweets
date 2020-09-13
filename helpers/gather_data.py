import numpy as np
import pandas as pd
import json
import requests
from PIL import Image


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


def download_image(url, tweet_id):
    is_success = True
    try:
        # Download the image (with a get request)
        response = requests.get(url)
        if response.ok:
            img_data = response.content
            # Save the picture in a JPG file
            with open(f"resources/images/{tweet_id}.jpg", 'wb') as file:
                file.write(img_data)

    except:
        is_success = False

    finally:
        return is_success


def build_image(path, config_resize=(100, 50), is_bw=True):
    try:
        # Access the image
        img = Image.open(path)

        # Resizing and conversion in black and white (if necessary)
        if is_bw:
            new_img = img.resize(config_resize, Image.ANTIALIAS).convert('L')
        else:
            new_img = img.resize(config_resize, Image.ANTIALIAS)

        # Flatten the image during the return
        return np.array(new_img).flatten().tolist()

    except:
        return None
