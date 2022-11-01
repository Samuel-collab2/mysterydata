from core.constants import DATASET_LABEL_NAME, DATASET_TRAIN_RATIO, MENU_EXIT, MENU_RETURN
from core.data_analysis import perform_linear_regression_analysis, perform_polynomial_complexity_analysis, \
    perform_lasso_lambda_analysis, perform_ridge_lambda_analysis, perform_feature_correlation_analysis
from core.data_visualization import generate_data_visualization_plots
from core.loader import load_train_dataset, load_standardized_train_dataset
from core.predict import predict_submission1_ridge, predict_submission1_propagation
from core.preprocessing import split_training_test, split_claims_accept_reject, \
    separate_features_label, convert_label_binary, expand_dataset
from library.option_input import OptionInput
from main_induction import perform_decision_tree_induction


def run_menu(title, options):
    exit_menu = None
    while exit_menu is None:
        print(f'--- {title} ---')
        option_input = OptionInput('Select option', options, lambda option: option[0])
        _, option = option_input.get_input()
        exit_menu = option()

    return exit_menu

def run_dataset_menu():
    return run_menu('Load dataset', [
        ('Train dataset', load_train_dataset),
        ('Train dataset - Standardized', load_standardized_train_dataset),
    ])

def main():
    dataset_raw = run_dataset_menu()

    raw_data = separate_features_label(dataset_raw, DATASET_LABEL_NAME)
    raw_features, raw_label = raw_data

    dataset = expand_dataset(dataset_raw)
    features, label = separate_features_label(dataset, DATASET_LABEL_NAME)

    binary_label = convert_label_binary(label)

    accept_data, reject_data = split_claims_accept_reject(features, label)
    accept_features, accept_label = accept_data
    reject_features, reject_label = reject_data

    def run_dev_test():
        # Intended for temporary development tests
        # Run whatever you want here
        pass

    def prediction_menu():
        run_menu("Model prediction menu", [
            ('Submission 1 - Ridge', lambda: predict_submission1_ridge(dataset_raw)),
            ('Submission 1 - Propagation', lambda: predict_submission1_propagation(dataset_raw)),
            MENU_RETURN
        ])

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
            ('Perform decision tree induction (Raw data)', lambda: perform_decision_tree_induction(dataset_raw)),
            MENU_RETURN
        ])

    def data_selection_menu(title, on_select):
        run_menu(title, [
            ('Expanded - Data', lambda: on_select(features, label)),
            ('Expanded - Binary label data', lambda: on_select(features, binary_label)),
            ('Expanded - Accepted claim data', lambda: on_select(accept_features, accept_label)),
            ('Expanded - Rejected claim data', lambda: on_select(reject_features, reject_label)),
            ('Raw - Data', lambda: on_select(raw_features, raw_label)),
            ('Raw - Binary label data', lambda: on_select(raw_features, binary_label)),
            MENU_RETURN
        ])

    def main_menu():
        run_menu('Main Menu', [
            (
                'Run dev test',
                run_dev_test
            ),
            (
                'Generate data visualization plots',
                lambda: data_selection_menu("Select Visualization Data", generate_data_visualization_plots)
            ),
            (
                'Perform data analysis',
                lambda: data_selection_menu("Select Analysis Data", analysis_menu)
            ),
            (
                'Run model prediction',
                prediction_menu
            ),
            MENU_EXIT
        ])

    main_menu()


if __name__ == '__main__':
    main()
