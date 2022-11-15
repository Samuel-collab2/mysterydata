from core.constants import DATASET_LABEL_NAME, DATASET_TRAIN_RATIO, MENU_EXIT, MENU_RETURN, \
    SIGNIFICANT_FEATURE_SET_COUNTS
from core.constants_feature_set import SIGNIFICANT_RIDGE_COLUMNS, SIGNIFICANT_BINARY_LABEL_COLUMNS, \
    SIGNIFICANT_FORWARD_STEPWISE_COLUMNS
from core.data_analysis import perform_linear_regression_analysis, perform_polynomial_complexity_analysis, \
    perform_lasso_lambda_analysis, perform_ridge_lambda_analysis, perform_feature_correlation_analysis
from core.data_visualization import generate_classification_plots, \
    generate_scatter_plots, generate_histogram_plots, generate_compound_plots, generate_correlation_plots
from core.loader import load_train_dataset, load_standardized_train_dataset, load_determining_dataset
from core.predict import predict_submission1_ridge, predict_submission1_propagation, predict_submission2, \
    predict_submission3
from core.preprocessing import split_training_test, split_claims_accept_reject, \
    separate_features_label, convert_label_binary, get_categorical_columns, expand_dataset_deterministic
from library.option_input import OptionInput
from main_induction import perform_induction_tests


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

    raw_features, raw_label = separate_features_label(dataset_raw, DATASET_LABEL_NAME)

    categorical_columns = get_categorical_columns(raw_features)
    dataset_expanded = expand_dataset_deterministic(dataset_raw, load_determining_dataset(), categorical_columns)

    expanded_features, expanded_label = separate_features_label(dataset_expanded, DATASET_LABEL_NAME)

    binary_label = convert_label_binary(expanded_label)

    (accept_features, accept_label), (reject_features, reject_label) = split_claims_accept_reject(
        expanded_features,
        expanded_label
    )

    def run_dev_test():
        # Intended for temporary development tests
        # Run whatever you want here
        pass

    def prediction_menu():
        run_menu('Model prediction menu', [
            ('Submission 1: Ridge', lambda: predict_submission1_ridge(dataset_raw)),
            ('Submission 1: Propagation', lambda: predict_submission1_propagation(dataset_raw)),
            ('Submission 2', lambda: predict_submission2(dataset_raw)),
            ('Submission 3', lambda: predict_submission3(dataset_raw)),
            MENU_RETURN
        ])

    def analysis_menu(features, label):
        train_data, test_data = split_training_test(
            features,
            label,
            DATASET_TRAIN_RATIO
        )

        run_menu('Data analysis menu', [
            ('Perform feature correlation analysis', lambda: perform_feature_correlation_analysis(features)),
            ('Perform linear regression analysis', lambda: perform_linear_regression_analysis(train_data, test_data)),
            ('Perform polynomial complexity analysis', lambda: perform_polynomial_complexity_analysis(train_data, test_data)),
            ('Perform lasso lambda analysis', lambda: perform_lasso_lambda_analysis(train_data, test_data)),
            ('Perform ridge lambda analysis', lambda: perform_ridge_lambda_analysis(train_data, test_data)),
            ('Perform induction tests (Raw data)', lambda: perform_induction_tests(dataset_raw)),
            MENU_RETURN
        ])

    def visualization_menu(features, label):
        run_menu('Data visualization menu', [
            ('Generate scatter plots', lambda: generate_scatter_plots(features, label)),
            ('Generate histogram plots', lambda: generate_histogram_plots(features)),
            ('Generate compound plots', lambda: generate_compound_plots(features, label)),
            ('Generate correlation plots', lambda: generate_correlation_plots(features)),
            ('Generate classification plots', lambda: generate_classification_plots(features, label)),
            MENU_RETURN
        ])

    def featureset_menu(features, label, on_select):
        def select_features(features, label, feature_columns):
            return lambda: feature_count_menu(lambda count: on_select(
                features.loc[:, feature_columns[:count]],
                label
            ))

        def feature_count_menu(on_select):
            def get_option(count):
                return f'{count}', lambda: on_select(count)

            run_menu("Select feature set count", [
                *[
                    get_option(count) for count
                    in SIGNIFICANT_FEATURE_SET_COUNTS
                ],
                MENU_RETURN,
            ])

        run_menu("Select feature set", [
            ('All', lambda: on_select(features, label)),
            ('Significant: Ridge', select_features(features, label, SIGNIFICANT_RIDGE_COLUMNS)),
            ('Significant: Ridge binary label', select_features(features, label, SIGNIFICANT_BINARY_LABEL_COLUMNS)),
            ('Significant: Forward stepwise', select_features(features, label, SIGNIFICANT_FORWARD_STEPWISE_COLUMNS)),
            MENU_RETURN
        ])

    def data_selection_menu(title, on_select):
        def select_featureset(features, label):
            return lambda: featureset_menu(features, label, on_select)

        run_menu(title, [
            ('Expanded: Data', select_featureset(expanded_features, expanded_label)),
            ('Expanded: Binary label data', select_featureset(expanded_features, binary_label)),
            ('Expanded: Accepted claim data', select_featureset(accept_features, accept_label)),
            ('Expanded: Rejected claim data', select_featureset(reject_features, reject_label)),
            ('Raw: Data', lambda: on_select(raw_features, raw_label)),
            ('Raw: Binary label data', lambda: on_select(raw_features, binary_label)),
            ('Raw: Categorical', lambda: on_select(raw_features.loc[:, categorical_columns], raw_label)),
            MENU_RETURN
        ])

    def main_menu():
        run_menu('Main menu', [
            (
                'Run dev test',
                run_dev_test
            ),
            (
                'Generate data visualization plots',
                lambda: data_selection_menu('Select visualization data', visualization_menu)
            ),
            (
                'Perform data analysis',
                lambda: data_selection_menu('Select analysis data', analysis_menu)
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
