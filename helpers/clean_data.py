import pandas as pd


def convert_column_to_datetime(df, column_name):
    df[column_name] = pd.to_datetime(df[column_name])
    return df


def rename_columns(df, old_new_columns):
    df = df.rename(columns=old_new_columns)
    return df


def get_some_columns(df, columns):
    df = df[columns]
    return df


def filter_by_condition(df, condition):
    return df[condition]
