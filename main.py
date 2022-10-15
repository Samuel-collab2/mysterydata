from os import mkdir
from os.path import join
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


TRAIN_DATASET_PATH = 'trainingset.csv'
TEST_DATASET_PATH = 'testset.csv'
DATASET_LABEL = 'ClaimAmount'
TRAIN_FACTOR = 0.8
OUTPUT_DIR = 'dist'
try:
    mkdir(OUTPUT_DIR)
except FileExistsError:
    pass


def load_dataset(file_path):
    return pd.read_csv(file_path)

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
        plot_scatter(x, y, feature_name=column)

def plot_scatter(x, y, feature_name):
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    ax.set_title(f'{DATASET_LABEL} over {feature_name}')
    ax.set_xlabel(feature_name)
    ax.set_ylabel(DATASET_LABEL)

    fig_path = join(OUTPUT_DIR, f'plot_{feature_name}.png')
    fig.savefig(fig_path, bbox_inches='tight')
    plt.close(fig)

def main():
    data = load_dataset(TRAIN_DATASET_PATH)
    data_split = split_data(data,
        label=DATASET_LABEL,
        train_factor=TRAIN_FACTOR)
    handle_basic_plots(data)
    handle_linear_regression(data_split)

if __name__ == '__main__':
    main()
