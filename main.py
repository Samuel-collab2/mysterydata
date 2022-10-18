from os import mkdir
from os.path import join
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

import loaddata_lab6 as loader


MIN_REAL_FEATURE_UNIQUE_VALUES = 20
DATASET_TRAIN_RATIO = 0.8
DATASET_LABEL = 'ClaimAmount'
DATASET_TRAIN_PATH = 'trainingset.csv'
DATASET_TEST_PATH = 'testset.csv'
OUTPUT_DIR = 'dist'
try:
    mkdir(OUTPUT_DIR)
except FileExistsError:
    pass


def load_dataset(file_path):
    return loader.load(file_path)

def split_data_label(data, label):
    return (
        data.drop(label, axis='columns', inplace=False),
        data.loc[:, label],
    )

def split_data(data, label, train_factor):
    x, y = split_data_label(data, label)
    return train_test_split(x, y, train_size=train_factor, shuffle=False)

def handle_linear_regression(data_split):
    train_x, test_x, train_y, test_y = data_split
    reg_model = LinearRegression()
    reg_model.fit(train_x, train_y)
    pred_y = reg_model.predict(test_x)
    pred_mae = np.mean(np.abs(test_y - pred_y))
    print(f'Linear regression MAE: {pred_mae:.4f}')

def handle_basic_plots(data):
    X, y = split_data_label(data, label=DATASET_LABEL)
    for column in X.columns:
        x = X.loc[:, column]
        num_column_unique_values = data[column].nunique()
        if num_column_unique_values < MIN_REAL_FEATURE_UNIQUE_VALUES:
            plot_hist(x, bins=num_column_unique_values, feature_name=column)
        else:
            plot_scatter(x, y, feature_name=column)

def handle_compound_plots(data):
    X, y = split_data_label(data, label=DATASET_LABEL)
    for column1, column2 in combinations(X.columns, r=2):
        x1 = X.loc[:, column1]
        x2 = X.loc[:, column2]
        x = x1 / x2
        plot_scatter(x, y, feature_name=f'{column1}-{column2}')

def plot_scatter(x, y, feature_name):
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    ax.set_title(f'Scatter plot of {DATASET_LABEL} over {feature_name}')
    ax.set_xlabel(feature_name)
    ax.set_ylabel(DATASET_LABEL)

    fig_path = join(OUTPUT_DIR, f'scatter_{feature_name}.png')
    fig.savefig(fig_path, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote scatter plot to {fig_path}')

def plot_hist(x, bins, feature_name):
    fig, ax = plt.subplots()
    ax.hist(x, bins)
    ax.set_title(f'Histogram of {DATASET_LABEL} over {feature_name}')
    ax.set_xlabel(feature_name)
    ax.set_ylabel(DATASET_LABEL)

    fig_path = join(OUTPUT_DIR, f'hist_{feature_name}.png')
    fig.savefig(fig_path, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote histogram to {fig_path}')

def main():
    data_condensed = pd.read_csv(DATASET_TRAIN_PATH)
    data_expanded = load_dataset(DATASET_TRAIN_PATH)

    handle_basic_plots(data_condensed)
    handle_compound_plots(data_condensed)

    data_split = split_data(data_expanded,
        label=DATASET_LABEL,
        train_factor=DATASET_TRAIN_RATIO)
    handle_linear_regression(data_split)

if __name__ == '__main__':
    main()
