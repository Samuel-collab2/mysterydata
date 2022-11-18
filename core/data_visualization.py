from os import mkdir
from os.path import join
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations

from core.constants import OUTPUT_DIR

def _write_plot(fig, file):
    try:
        mkdir(OUTPUT_DIR)
    except FileExistsError:
        pass

    fig_path = join(OUTPUT_DIR, file)
    fig.savefig(fig_path, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote plot to {fig_path}')

def _create_plot(set_plot, title, x_axis, y_axis, file_name):
    fig, ax = plt.subplots()
    set_plot(ax)
    ax.set_title(title)
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)

    _write_plot(fig, f'{file_name}.png')
    plt.close(fig)

def _get_classes(x, y, label):
    class_values = label.unique()
    classes = {}
    for class_value in class_values:
        classes[class_value] = [
            (x.iloc[index], y.iloc[index]) for index, label_value
            in enumerate(label)
            if label_value == class_value
        ]

    return classes

def plot_classification(class_names, x_classes, y_classes, x_axis, y_axis, file_name):
    def configure_plot(ax):
        for index, (x_class, y_class) in enumerate(zip(x_classes, y_classes)):
            ax.scatter(x_class, y_class, s=5, alpha=0.2, zorder=index)

        ax.legend(class_names)

    _create_plot(
        configure_plot,
        f'Scatter plot of {y_axis} over {x_axis} with classification',
        x_axis,
        y_axis,
        file_name,
    )

def plot_scatter(x, y, x_axis, y_axis, file_name):
    _create_plot(
        lambda ax: ax.scatter(x, y, s=5),
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

def plot_heatmap(x, y, x_axis, y_axis, num_bins, file_name):
    fig, ax = plt.subplots(1, 2)

    features = pd.concat((
        pd.DataFrame(),
        pd.Series(x.values),
        pd.Series(y.values),
    ), axis='columns')
    features.columns = (0, 1)
    heatmap_features = features.groupby([
        pd.cut(features[1], num_bins),
        pd.cut(features[0], num_bins),
    ]).mean().unstack()
    heatmap_min = heatmap_features.min().min()
    heatmap_max = heatmap_features.max().max()

    def subplot_heatmap(features, ax):
        sns.heatmap(features,
            ax=ax,
            vmin=heatmap_min,
            vmax=heatmap_max,
            cmap='Blues',
            square=True)
        ax.set_xlabel(x_axis)
        ax.set_ylabel(y_axis)
        ax.invert_yaxis()

    subplot_heatmap(heatmap_features[0], ax[0])
    ax[0].set_title('Heatmap for rejected claims')

    subplot_heatmap(heatmap_features[1], ax[1])
    ax[1].set_title('Heatmap for accepted claims')

    fig.set_figwidth(16)
    _write_plot(fig, f'{file_name}.png')


def generate_scatter_plots(features, label):
    for column in features.columns:
        x = features.loc[:, column]
        plot_scatter(x, label, x_axis=column, y_axis=label.name, file_name=f'scatter_{column}')

def generate_histogram_plots(features):
    for column in features.columns:
        x = features.loc[:, column]
        plot_hist(x, bins=features[column].nunique(), x_axis=column, y_axis='count', file_name=f'hist_{column}')

def generate_compound_plots(features, label):
    for column1, column2 in combinations(features.columns, r=2):
        x = features.loc[:, column1]
        y = features.loc[:, column2]

        x_axis = f'{column1}-{column2}'
        plot_scatter(x / y, label, x_axis=x_axis, y_axis=label.name, file_name=f'compound_{x_axis}')

def generate_correlation_plots(features):
    for column1, column2 in combinations(features.columns, r=2):
        x = features.loc[:, column1]
        y = features.loc[:, column2]

        plot_scatter(x, y, x_axis=column1, y_axis=column2, file_name=f'correlation_{column1}-{column2}')

def generate_correlation_heatmaps(features, label):
    features = features.drop('rowIndex', axis='columns', inplace=False)

    for column1, column2 in combinations(features.columns, r=2):
        x = features.loc[:, column1]
        y = features.loc[:, column2]

        # classes = _get_classes(x, y, label)
        plot_heatmap(
            x,
            y,
            x_axis=column1,
            y_axis=column2,
            num_bins=7,
            file_name=f'heatmap_{column1}-{column2}')

def generate_classification_plots(features, label):
    for column1, column2 in combinations(features.columns, r=2):
        x = features.loc[:, column1]
        y = features.loc[:, column2]

        classes = _get_classes(x, y, label)
        plot_classification(
            classes.keys(),
            *zip(*(zip(*points) for points in classes.values())),
            x_axis=column1,
            y_axis=column2,
            file_name=f'classification_{column1}-{column2}'
        )
