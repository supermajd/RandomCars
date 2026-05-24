
"""features.py: Loads Ames housing data, builds the train/test split, and prepares model features."""

__author__ = "Majd Jamal"

import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from ml.config import (
    DATA_PATH, TARGET, FEATURES,
    NUMERIC_FEATURES, CATEGORICAL_FEATURES,
    RANDOM_STATE, TEST_SIZE)


def load_data(path) -> pd.DataFrame:
    """ Loads the Ames housing dataset.
    :param path: Path to the CSV file
    :return data: Raw housing data
    """

    data = pd.read_csv(path)
    
    return data
    
def split(data):
    """ Splits data into train and test sets with a fixed seed.
    :param data: Raw housing data with feature and target columns
    :return X_train, X_test, y_train, y_test: Split features and targets as DataFrames
    """

    X = data[FEATURES]
    y = data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size = TEST_SIZE,
        random_state = RANDOM_STATE)

    return X_train, X_test, y_train, y_test


def build_preprocessor() -> ColumnTransformer:
    """ Builds the column-wise preprocessing transformer.

    Numeric features are imputed with the median. Categorical features are
    imputed with the most frequent value and one-hot encoded. The encoder
    ignores unseen categories so inference does not break on new values. Imputed is
    the way that we will empty values. One hot encoding is needed because 
    categorical values does not store order or value.

    :return preprocessor: Unfitted ColumnTransformer for the model pipeline
    """

    numeric = Pipeline(steps = [
        ('imputer', SimpleImputer(strategy = 'median')),
    ])

    categorical = Pipeline(steps = [
        ('imputer', SimpleImputer(strategy = 'most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown = 'ignore')),
    ])

    preprocessor = ColumnTransformer(transformers = [
        ('num', numeric, NUMERIC_FEATURES),
        ('cat', categorical, CATEGORICAL_FEATURES),
    ])

    return preprocessor

if __name__ == '__main__':

    data = load_data(DATA_PATH)

    X_train, X_test, y_train, y_test = split(data)

    preprocessor = build_preprocessor()

