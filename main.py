import sys
import time

from core.loader import load_train_dataset, load_standardized_train_dataset, load_test_dataset
from core.processing import expand_dataset, split_training_test, split_claims_accept_reject, separate_features_label
from core.data_visualization import generate_data_visualization_plots
from core.constants import DATASET_LABEL_NAME, DATASET_TRAIN_RATIO, MENU_EXIT, MENU_RETURN
from core.data_analysis import perform_linear_regression_analysis, perform_polynomial_complexity_analysis, \
    perform_lasso_lambda_analysis, perform_ridge_lambda_analysis, perform_feature_correlation_analysis
from library.option_input import OptionInput

def menu_delay():
    """
    Flushes terminal and waits a bit.
    """
    sys.stdout.flush()
    time.sleep(0.5)

def run_menu(title, options):
    exit_menu = None
    while exit_menu is None:
        menu_delay()
        print(f'--- {title} ---')
        input = OptionInput('Select option', options, lambda option: option[0])
        _, option = input.get_input()
        exit_menu = option()
    return exit_menu

def run_dataset_menu():
    return run_menu('Load dataset', [
        ('Train dataset', load_train_dataset),
        ('Standardized train dataset', load_standardized_train_dataset),
        ('Test dataset', load_test_dataset),
    ])

def main():
    dataset_raw = run_dataset_menu()

    raw_data = separate_features_label(dataset_raw, DATASET_LABEL_NAME)
    raw_features, raw_label = raw_data

    dataset = expand_dataset(dataset_raw)
    features, label = separate_features_label(dataset, DATASET_LABEL_NAME)

    accept_data, reject_data = split_claims_accept_reject(features, label)
    accept_features, accept_label = accept_data
    reject_features, reject_label = reject_data

    def run_dev_test():
        # Intended for temporary development tests
        # Run whatever you want here
        pass

    def analysis_menu(features, label):
        train_data, test_data = split_training_test(
            features,
            label,
            DATASET_TRAIN_RATIO
        )

        run_menu('Data Analysis Menu', [
            ('Perform feature correlation analysis', lambda: perform_feature_correlation_analysis(features)),
            ('Perform linear regression analysis', lambda: perform_linear_regression_analysis(train_data, test_data)),
            ('Perform polynomial complexity analysis', lambda: perform_polynomial_complexity_analysis(train_data, test_data)),
            ('Perform lasso lambda analysis', lambda: perform_lasso_lambda_analysis(train_data, test_data)),
            ('Perform ridge lambda analysis', lambda: perform_ridge_lambda_analysis(train_data, test_data)),
            MENU_RETURN
        ])

    def data_selection_menu(title, next_menu):
        run_menu(title, [
            ('Expanded data', lambda: next_menu(features, label)),
            ('Accepted claim data', lambda: next_menu(accept_features, accept_label)),
            ('Rejected claim data', lambda: next_menu(reject_features, reject_label)),
            ('Raw data', lambda: next_menu(raw_features, raw_label)),
            MENU_RETURN
        ])

    def main_menu():
        run_menu('Main Menu', [
            ('Run dev test', run_dev_test),
            ('Generate data visualization plots', lambda: generate_data_visualization_plots(raw_features, raw_label)),
            ('Perform data analysis', lambda: data_selection_menu("Select Analysis Data", analysis_menu)),
            MENU_EXIT
        ])

    main_menu()


if __name__ == '__main__':
    main()
