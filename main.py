from core.loader import load_train_dataset
from core.processing import expand_dataset, split_training_test, split_claims_accept_reject, separate_features_label
from core.data_visualization import handle_basic_plots, handle_compound_plots
from core.constants import DATASET_LABEL_NAME, DATASET_TRAIN_RATIO, POLYNOMIAL_ANALYSIS_DEGREES, CROSS_VALIDATION_SETS
from core.data_analysis import calculate_linear_regression_error, calculate_polynomial_complexity_errors
from library.option_input import OptionInput

def run_menu(options):
    exit_menu = None
    while not exit_menu:
        print('--- Insurance Claim Model Menu ---')
        input = OptionInput('Select Option', options, lambda option: option[0])
        _, option = input.get_input()
        exit_menu = option()

def generate_data_plots(features, label):
    handle_basic_plots(features, label)
    handle_compound_plots(features, label)

def perform_model_analysis(train_data, test_data):
    print('Calculating linear regression mae...')
    linear_regression_mae = calculate_linear_regression_error(*train_data, *test_data)
    print(f'Linear regression MAE: {linear_regression_mae:.4f}')

    degrees = range(1, POLYNOMIAL_ANALYSIS_DEGREES + 1)
    cv_sets = CROSS_VALIDATION_SETS
    print(f'Calculating polynomial regression mae for degrees={degrees} across {cv_sets} cross-validation sets...')
    for degree, train_error, cv_error in calculate_polynomial_complexity_errors(*train_data, degrees=degrees, cv_sets=cv_sets):
        print(f'Polynomial regression degree={degree}')
        print(f'|_ Train error: {train_error:.4f}')
        print(f'|_ Validation error: {cv_error:.4f}')

def run_dev_test(raw_data, train_data, test_data):
    # Intended for temporary development tests
    # Run whatever you want here
    pass

def main():
    dataset_raw = load_train_dataset()
    raw_data = separate_features_label(dataset_raw, DATASET_LABEL_NAME)
    raw_features, raw_label = raw_data

    dataset = expand_dataset(dataset_raw)
    features, label = separate_features_label(dataset, DATASET_LABEL_NAME)

    accept_data, _ = split_claims_accept_reject(features, label)
    accept_features, accept_label = accept_data

    train_data, test_data = split_training_test(
        accept_features,
        accept_label,
        train_factor=DATASET_TRAIN_RATIO,
    )

    run_menu([
        ('Run dev test', lambda: run_dev_test(raw_data, train_data, test_data)),
        ('Generate Data Plots', lambda: generate_data_plots(raw_features, raw_label)),
        ('Perform Data Analysis', lambda: perform_model_analysis(train_data, test_data)),
        ('Exit', lambda: True)
    ])


if __name__ == '__main__':
    main()
