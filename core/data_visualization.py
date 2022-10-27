from os import mkdir
from os.path import join
import matplotlib.pyplot as plt
from itertools import combinations

from core.constants import MIN_REAL_FEATURE_UNIQUE_VALUES, OUTPUT_DIR

def handle_basic_plots(features, label):
    for column in features.columns:
        x = features.loc[:, column]
        num_column_unique_values = features[column].nunique()
        if num_column_unique_values < MIN_REAL_FEATURE_UNIQUE_VALUES:
            plot_hist(x, bins=num_column_unique_values, x_axis=column, y_axis=label.name, file_name=f'hist_{column}')
        else:
            plot_scatter(x, label, x_axis=column, y_axis=label.name, file_name=f'scatter_{column}')

def handle_compound_plots(features, label):
    for column1, column2 in combinations(features.columns, r=2):
        x1 = features.loc[:, column1]
        x2 = features.loc[:, column2]
        x = x1 / x2

        compound_column = f'{column1}-{column2}'
        plot_scatter(x, label, x_axis=compound_column, y_axis=label.name, file_name=f'scatter_{compound_column}')
        plot_scatter(x1, x2, x_axis=column1, y_axis=column2, file_name=f'correlation_{compound_column}')

def _create_plot(set_plot, title, x_axis, y_axis, file_name):
    fig, ax = plt.subplots()
    set_plot(ax)
    ax.set_title(title)
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)

    _write_plot(fig, f'{file_name}.png')
    plt.close(fig)

def plot_scatter(x, y, x_axis, y_axis, file_name):
    _create_plot(
        lambda ax: ax.scatter(x, y),
        f'Scatter plot of {y_axis} over {x_axis}',
        x_axis,
        y_axis,
        file_name,
    )

def plot_line(x, y, x_axis, y_axis, file_name):
    _create_plot(
        lambda ax: ax.plot(x, y),
        f'Line plot of {y_axis} over {x_axis}',
        x_axis,
        y_axis,
        file_name,
    )

def plot_hist(x, bins, x_axis, y_axis, file_name):
    _create_plot(
        lambda ax: ax.hist(x, bins),
        f'Histogram of {y_axis} over {x_axis}',
        x_axis,
        y_axis,
        file_name,
    )

def _write_plot(fig, file):
    try:
        mkdir(OUTPUT_DIR)
    except FileExistsError:
        pass

    fig_path = join(OUTPUT_DIR, file)
    fig.savefig(fig_path, bbox_inches='tight')

    print(f'Wrote plot to {fig_path}')
