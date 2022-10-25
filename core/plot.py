from os import mkdir
from os.path import join
import matplotlib.pyplot as plt
from itertools import combinations

from .constants import MIN_REAL_FEATURE_UNIQUE_VALUES, OUTPUT_DIR

def handle_basic_plots(data, label_name):
    features, label = data
    for column in features.columns:
        x = features.loc[:, column]
        num_column_unique_values = features[column].nunique()
        if num_column_unique_values < MIN_REAL_FEATURE_UNIQUE_VALUES:
            plot_hist(x, bins=num_column_unique_values, feature_name=column, label_name=label_name)
        else:
            plot_scatter(x, label, feature_name=column, label_name=label_name)

def handle_compound_plots(data, label_name):
    features, label = data
    for column1, column2 in combinations(features.columns, r=2):
        x1 = features.loc[:, column1]
        x2 = features.loc[:, column2]
        x = x1 / x2
        plot_scatter(x, label, feature_name=f'{column1}-{column2}', label_name=label_name)

def plot_scatter(x, y, feature_name, label_name):
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    ax.set_title(f'Scatter plot of {label_name} over {feature_name}')
    ax.set_xlabel(feature_name)
    ax.set_ylabel(label_name)

    _write_plot(fig, f'scatter_{feature_name}.png')
    plt.close(fig)


def plot_hist(x, bins, feature_name, label_name):
    fig, ax = plt.subplots()
    ax.hist(x, bins)
    ax.set_title(f'Histogram of {label_name} over {feature_name}')
    ax.set_xlabel(feature_name)
    ax.set_ylabel(label_name)

    _write_plot(fig, f'hist_{feature_name}.png')
    plt.close(fig)

def _write_plot(fig, filename):
    try:
        mkdir(OUTPUT_DIR)
    except FileExistsError:
        pass

    fig_path = join(OUTPUT_DIR, filename)
    fig.savefig(fig_path, bbox_inches='tight')

    print(f'Wrote plot to {fig_path}')
