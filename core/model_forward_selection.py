import pandas as pd
import core.processing as proc
from sklearn.linear_model import LinearRegression
import numpy as np
import sys


LARGEST_FLOAT = sys.float_info.max


def get_my_data_bb(train_factor=0.75):
    # load data
    data = pd.read_csv('../trainingset.csv', low_memory=False)

    # apply one hot encoding to all categorical values
    expanded_dataset = proc.expand_dataset(data)

    # DataFrame, and series.Series << feature and label types
    split_data = proc.separate_features_label_and_remove_rowIndex_col(expanded_dataset, 'ClaimAmount')

    # split samples by claim_amount > 0 <split_data_by_claim[0]>, and ClaimAmount == 1 <split_data_by_claim[1]>
    split_data_by_claim = proc.split_claims_accept_reject(split_data[0], split_data[1])
    split_data_train_test = proc.split_training_test(split_data_by_claim[0][0], split_data_by_claim[0][1], train_factor)

    return split_data_train_test


def forward_stepwise(total_features=10):
    # [0] = training.   [0][0] = features.  [0][1] = labels
    # [1] = test.       [1][0] = ...
    split_data_train_test = get_my_data_bb(0.75)
    training_features = split_data_train_test[0][0]
    training_labels = split_data_train_test[0][1]
    test_features = split_data_train_test[1][0]
    test_labels = split_data_train_test[1][1]

    # This copy of dataset will be stored separately for subset creating usage (inside for loop)
    training_features_origin = training_features
    test_features_origin = test_features

    # Apply forward stepwise selection to generate <total_features> most significant features (mae wise)
    number_of_features = training_features.shape[1]
    my_array_of_features = []
    for i in range(0, total_features):
        best_mae_and_feature = [LARGEST_FLOAT, "filler_feature_name"]
        for y in range(0, number_of_features):
            test_features_categories = []
            test_features_categories.extend(my_array_of_features)
            test_features_categories.append(training_features.columns[y])

            data_train_subset = training_features_origin[test_features_categories]
            data_test_subset = test_features_origin[test_features_categories]

            # calculate mae with chosen features
            lin_reg = LinearRegression()
            lin_reg.fit(data_train_subset, training_labels)
            price_pred = lin_reg.predict(data_test_subset)

            # calculate RSS with chosen features
            res_sum_sq = np.sum(np.square(test_labels - price_pred))
            # print(res_sum_sq)

            # update 'logger' on which feature added is improving rss calculations
            if best_mae_and_feature[0] > res_sum_sq:
                best_mae_and_feature[0] = res_sum_sq
                best_mae_and_feature[1] = training_features.columns[y]

            # # calculate mae with chosen features
            # mae = np.mean(np.absolute(test_labels - price_pred))
            # # print(mae)
            #
            # # update 'logger' on which feature added is improving mae calculations
            # if best_mae_and_feature[0] > mae:
            #     best_mae_and_feature[0] = mae
            #     best_mae_and_feature[1] = training_features.columns[y]

        number_of_features = number_of_features - 1

        # drop selected feature from BOTH training and test datasets (or order for above inner loop's functionality)
        print(" * * * * * ")
        print("added feature to list + improved optimality function score: %s || %f"
              % (best_mae_and_feature[1], best_mae_and_feature[0]))
        training_features = training_features.drop(best_mae_and_feature[1], axis='columns', inplace=False)
        test_features = test_features.drop(best_mae_and_feature[1], axis='columns', inplace=False)

        # update 'my_array_of_features'!
        my_array_of_features.append(best_mae_and_feature[1])

    print("\nFinal chosen features")
    print(my_array_of_features)
    # calculate mae with final chosen features
    data_train_subset = training_features_origin[my_array_of_features]
    data_test_subset = test_features_origin[my_array_of_features]
    lin_reg = LinearRegression()
    lin_reg.fit(data_train_subset, training_labels)
    price_pred = lin_reg.predict(data_test_subset)
    mae = np.mean(np.absolute(test_labels - price_pred))
    print("MAE optimality function score for final chosen features: %f" % mae)

    # TRAINING and TESTING SUBSET
    # my_arr = [training_features.columns[0], training_features.columns[20]]
    # df = training_features[my_arr]
    # lin_reg = LinearRegression()
    # lin_reg.fit(df, training_labels)
    #
    # my_arr2 = [test_features.columns[0], test_features.columns[20]]
    # df2 = test_features[my_arr2]
    # price_pred = lin_reg.predict(df2)
    # mae = np.mean(np.absolute(test_labels - price_pred))
    # print(mae)

    # SUBSET
    # my_arr = [training_features.columns[0], training_features.columns[20]]
    # df = training_features[my_arr]
    # print(df)

    # DROP
    # print(training_features)
    # training_features = training_features.drop('feature1', axis='columns', inplace=False)
    # print(training_features)

    # INSERTING
    # df = pd.DataFrame()
    # df.insert(0, 'feature1', split_data_train_test[0][0]["feature1"])


forward_stepwise()
