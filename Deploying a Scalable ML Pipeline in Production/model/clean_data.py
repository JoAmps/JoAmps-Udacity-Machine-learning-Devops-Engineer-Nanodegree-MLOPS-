import pandas as pd
import logging


def load_data(path):
    try:
        df = pd.read_csv(path, index_col=[0])
        logging.info('SUCCESS: Data succesfully imported')
        return df
    except BaseException:
        logging.info('ERROR: Data not imported')


def cleaned_data(df):
    try:
        df.columns = df.columns.str.strip()
        df.drop("fnlgt", axis="columns", inplace=True)
        df.drop("education-num", axis="columns", inplace=True)
        df.drop("capital-gain", axis="columns", inplace=True)
        df.drop("capital-loss", axis="columns", inplace=True)
        df.to_csv('census_cleaned.csv')
        logging.info('SUCCESS:Data cleaned!')
        return df
    except BaseException:
        logging.info('ERROR Data could not be cleaned')


if __name__ == '__main__':
    df = load_data("census.csv")
    cleaned_data(df)
