from core.loader import load_train_dataset
from core.processing import expand_dataset, split_training_test, split_claims_accept_reject, separate_features_label
from core.data_visualization import handle_basic_plots, handle_compound_plots
from core.constants import DATASET_LABEL_NAME, DATASET_TRAIN_RATIO, MENU_EXIT, MENU_RETURN
from core.data_analysis import perform_linear_regression_analysis, perform_polynomial_complexity_analysis, perform_lasso_lambda_analysis, perform_ridge_lambda_analysis
from library.option_input import OptionInput

def run_menu(title, options):
    exit_menu = None
    while not exit_menu:
        print(f'--- {title} ---')
        input = OptionInput("Select option", options, lambda option: option[0])
        _, option = input.get_input()
        exit_menu = option()

def generate_data_plots(features, label):
    handle_basic_plots(features, label)
    handle_compound_plots(features, label)

def perform_model_analysis(train_data, test_data):
    perform_linear_regression_analysis(train_data, test_data)
    input('Linear regression analysis complete, press any key to continue...')
    perform_polynomial_complexity_analysis(train_data, test_data)
    input('Polynomial complexity analysis complete, press any key to continue...')
    perform_lasso_lambda_analysis(train_data, test_data)
    input('Lasso lambda analysis complete, press any key to continue...')
    perform_ridge_lambda_analysis(train_data, test_data)
    input('Ridge lambda analysis complete, press any key to continue...')

def run_dev_test():
    pass

def main():
    dataset_raw = load_train_dataset()
    raw_data = separate_features_label(dataset_raw, DATASET_LABEL_NAME)
    raw_features, raw_label = raw_data

    dataset = expand_dataset(dataset_raw)
    features, label = separate_features_label(dataset, DATASET_LABEL_NAME)

    accept_data, reject_data = split_claims_accept_reject(features, label)
    accept_features, accept_label = accept_data
    reject_features, reject_label = reject_data

    def data_selection_menu(title, next_menu):
        run_menu(title, [
            ('Expanded processed dataset', lambda: next_menu(features, label)),
            ('Accepted claim dataset', lambda: next_menu(accept_features, accept_label)),
            ('Rejected claim dataset', lambda: next_menu(reject_features, reject_label)),
            ('Raw dataset', lambda: next_menu(raw_features, raw_label)),
            MENU_RETURN
        ])

    def analysis_menu(features, label):
        train_data, test_data = split_training_test(
            features,
            label,
            DATASET_TRAIN_RATIO
        )

        run_menu('Data Analysis Menu', [
            ('Perform all analysis', lambda: perform_model_analysis(train_data, test_data)),
            ('Perform Linear regression analysis', lambda: perform_linear_regression_analysis(train_data, test_data)),
            ('Perform Polynomial complexity analysis', lambda: perform_polynomial_complexity_analysis(train_data, test_data)),
            ('Perform Lasso lambda analysis', lambda: perform_lasso_lambda_analysis(train_data, test_data)),
            ('Perform Ridge lambda analysis', lambda: perform_ridge_lambda_analysis(train_data, test_data)),
            MENU_RETURN
        ])

    def main_menu():
        run_menu('Insurance Claim Model Main Menu', [
            ('Run dev test', lambda: run_dev_test()),
            ('Generate Data Visualization Plots', lambda: generate_data_plots(raw_features, raw_label)),
            ('Perform Data Analysis', lambda: data_selection_menu("Select Analysis Data", analysis_menu)),
            MENU_EXIT
        ])

    main_menu()


if __name__ == '__main__':
    main()
