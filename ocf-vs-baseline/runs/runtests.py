import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

import os
import time
from datetime import datetime
import matplotlib.pyplot as plt

from constants import *
from datadistances import *
from dataop import my_from_csv, pretty_confusion_matrix, confusion_matrix_into_four_values, \
    get_description_dataset, compute_roc_auc, plot_roc_curve, save_test_results_into_dataset, sorted_by_increasing_size, \
    my_to_csv
from distances import get_distance_function
from my_scaling import my_scaling_dataset_test
from ocf_plot import create_ocf_plot
from to_file import *

if __name__ == "__main__":
    all_dataset_names = sorted_by_increasing_size(get_all_datasets_names())

    cpt_dataset = 0
    for dataset_name in all_dataset_names:
        cpt_dataset += 1
        print(f'Processing the CLASS dataset {dataset_name}...')

        # If not all samplings have been trained for this dataset, we pass to the next dataset
        if not check_if_all_sampling_have_been_OCFHammingtrained(dataset_name):
            print(f'Some (or all) samplings for {dataset_name} have not been trained yet! Pass to the next dataset...')
            continue

        # For every sampling, compute the OCFed test dataset, and write in a log file the results:
        # the confusion matrix, and the value for each metric

        # List all sampling directories for the corresponding dataset
        all_sampling_directories = get_all_sampling_dir_paths(dataset_name)

        # test if there is at least one sampling, otherwise pass to the next dataset
        if not all_sampling_directories:
            print(f'No sampling for dataset {dataset_name}! Pass to the next one...')
            continue

        TP = 0
        FP = 0
        TN = 0
        FN = 0
        # Creating a dictionary with keys from ALL_METRIC_NAMES and values initialized to 0 to sum the value of
        # all metrics over all samplings, as well as the roc_auc
        metric_total_values = {metric: 0 for metric in ALL_METRIC_NAMES}
        metric_total_values[METRIC_ROCAUC] = 0

        dataset_results = {MODEL_NAME: MODEL_NAME_OCF_HAMMING}

        for sampling_dir_name in all_sampling_directories:
            id_sampling = get_id_sampling_from_dir_path(sampling_dir_name)

            print(f'Processing the test dataset {get_test_withOCF_file_name(dataset_name, id_sampling)}...')
            # We first check if that sampling has been trained! If not, we pass to the next one
            assert check_if_sampling_is_OCFHammingtrained(dataset_name, id_sampling)

            #test_dataset_OCF_file_name_without_extension = get_dataset_dir_path(
            #    dataset_name) + '/' + sampling_dir_name + '/' + sampling_dir_name + DATASET_SAMPLING_TEST_NAME + DATASET_OCF_NAME + SCALING_HAMMING_NAME
            #test_dataset_OCF_file_name = test_dataset_OCF_file_name_without_extension + '.csv'

            # We will also need the training dataset to test the test dataset
            # Remember that the classifier itself is characterized by the training dataset, a distance function,
            # and a shift value to convert the distance into an OCF value
            #train_dataset_file_name = get_dataset_dir_path(
            #    dataset_name) + '/' + sampling_dir_name + '/' + sampling_dir_name + DATASET_SAMPLING_TRAIN_NAME + DATASET_OCF_NAME + SCALING_HAMMING_NAME + '.csv'
            df_train = my_from_csv(get_sampling_train_ocfhamming_file_path(dataset_name, id_sampling))
            df_test = my_from_csv(get_sampling_test_ocfhamming_file_path(dataset_name, id_sampling))
            # Now creating the OCFed test dataset
            # We first read the initial test dataset

            print(f'Creating the OCFed test dataset {get_sampling_test_ocfhamming_file_path(dataset_name, id_sampling)}...')
            # We retrieve the shift value to withdraw from a distance to get the OCF value for each row in the test set
            with open(get_OCF_to_distance_shift_value_file_name(dataset_name, id_sampling), 'r') as shift_value_file:
                shift_value = int(shift_value_file.read())

            df_type = my_from_csv(get_dataset_feature_type_list(dataset_name))
            df_test_with_distances = get_dataset_with_distances_scaled(df_test, df_type, ext_dataset=df_train)
            df_test_with_ocf, _ = convert_from_distances_to_ocf_scaled(df_test_with_distances, df_type, shift_value)
            # We save the OCFed test dataset in a file
            my_to_csv(df_test_with_ocf, get_test_withOCF_file_name(dataset_name, id_sampling))

            print(f'OCFed test dataset {get_test_withOCF_file_name(dataset_name, id_sampling)} created!')

            # We retrieve the threshold
            with open(get_best_threshold_file_name(dataset_name, id_sampling), 'r') as best_thr_file:
                thr = int(best_thr_file.read())

            # We save the OCF with threshold into a plot
            create_ocf_plot(df_test_with_ocf, get_fig_test_with_threshold_file_name(dataset_name, id_sampling),
                            get_fig_title(dataset_name, id_sampling), threshold=thr)
            print(f'OCF plot created!')

            # We create the contents for the log file: we compute the confusion matrix and write the value
            # for each metric (f1 score, accuracy, precision, recall, roc auc)
            y_true = df_test_with_ocf[TARGET_DATA_COLUMN_NAME].values
            y_ocf_values = df_test_with_ocf[OCF_DATA_COLUMN_NAME].values
            y_pred_values = [1 if val <= thr else 0 for val in df_test_with_ocf[OCF_DATA_COLUMN_NAME]]

            # compute the roc auc
            roc_auc, fpr, tpr = compute_roc_auc(y_true, y_ocf_values)
            # Plot and save the ROC curve
            plot_roc_curve(fpr, tpr, roc_auc, get_fig_test_roc_file_name(dataset_name, id_sampling))

            conf_matrix_four_values = confusion_matrix_into_four_values(y_true, y_pred_values)
            # TP, FP, TN, FN
            current_TP = conf_matrix_four_values[0]
            current_FP = conf_matrix_four_values[1]
            current_TN = conf_matrix_four_values[2]
            current_FN = conf_matrix_four_values[3]
            TP = TP + current_TP
            FP = FP + current_FP
            TN = TN + current_TN
            FN = FN + current_FN

            # for log only
            log_message_test_results = [FILLED_LINE]
            for message in get_description_dataset(df_test_with_ocf, 'Test dataset'):
                log_message_test_results.append(message)
            log_message_test_results.append(f'Confusion matrix:\n{pretty_confusion_matrix(conf_matrix_four_values)}')
            for metric_name in ALL_METRIC_NAMES:
                # We list the value for each metric given the test results
                value = calculate_metric(y_true, y_pred_values, metric_name)
                metric_total_values[metric_name] += value
                log_message_test_results.append(f'Metric {metric_name}: {value}')
            metric_total_values[METRIC_ROCAUC] += roc_auc
            log_message_test_results.append(f'ROC AUC: {roc_auc}')
            log_message_test_results.append(FILLED_LINE)
            print(log_message_test_results)
            # We write the results in the log file
            with open(get_sampling_test_log_file_name(dataset_name, id_sampling), 'a') as log_file:
                for message in log_message_test_results:
                    log_file.write(datetime.now().strftime("[%Y-%m-%d %H:%M:%S]") + ' ' + message + '\n')

        # And now we want the average quality for the dataset among all samplings for this dataset
        # The variables TP, FP, TN, FN contain all info

        global_log_message_test_results = [FILLED_LINE]
        for message in get_description_dataset(my_from_csv(get_dataset_file_path(dataset_name)), 'Dataset'):
            global_log_message_test_results.append(message)
        global_log_message_test_results.append(f'Confusion matrix:\n{pretty_confusion_matrix([TP, FP, TN, FN])}')
        dataset_results.update({TOTAL_TP: TP, TOTAL_TN: TN, TOTAL_FP: FP, TOTAL_FN: FN})

        # Calculate global precision
        # precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        # Calculate global recall
        # recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        # Calculate global F1-score
        # f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        # Calculate global accuracy
        # accuracy = (TP + TN) / (TP + FP + TN + FN) if (TP + FP + TN + FN) > 0 else 0.0
        # Calculate global mcc
        # mcc = ((TP * TN) - (FP * FN)) / (((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) ** 0.5) if (((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) ** 0.5) > 0 else 0.0
        # Calculate avg of all metrics
        metric_total_values = {metric_name: value / len(all_sampling_directories) for metric_name, value in
                               metric_total_values.items()}

        for metric_name, value in metric_total_values.items():
            global_log_message_test_results.append(f'Avg metric {metric_name}: {value}')
            dataset_results[METRIC_NAME_TO_AVG_METRIC_FEATURE_NAME[metric_name]] = value

        global_log_message_test_results.append(FILLED_LINE)

        print(global_log_message_test_results)

        # We write the results in the log file
        with open(get_dataset_log_file_name(dataset_name), 'a') as log_file:
            for message in global_log_message_test_results:
                log_file.write(datetime.now().strftime("[%Y-%m-%d %H:%M:%S]") + ' ' + message + '\n')

        # We save all resulting tests in a dataset
        save_test_results_into_dataset(get_all_results_file_name(dataset_name), dataset_results)
