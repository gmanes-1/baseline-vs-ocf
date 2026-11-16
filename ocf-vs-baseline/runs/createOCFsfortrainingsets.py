import time
from datetime import datetime
import matplotlib.pyplot as plt

from datadistances import *
from dataop import my_from_csv, sorted_by_increasing_size, get_description_dataset
from ocf_plot import create_ocf_plot
from to_file import *


# def create_ocf_plot_old(dataset_ocf, fig_file, fig_title, threshold=None):
#     # Separate positive and negative data
#     positive_dataset = dataset_ocf[dataset_ocf[TARGET_DATA_COLUMN_NAME] == 1]
#     negative_dataset = dataset_ocf[dataset_ocf[TARGET_DATA_COLUMN_NAME] == 0]
#
#     plt.figure(figsize=(6, 8))
#     plt.scatter(1 - negative_dataset[TARGET_DATA_COLUMN_NAME], negative_dataset[OCF_DATA_COLUMN_NAME], color='red', label='neg')
#     plt.scatter(1 - positive_dataset[TARGET_DATA_COLUMN_NAME], positive_dataset[OCF_DATA_COLUMN_NAME], color='blue', label='pos')
#     if threshold:
#         # Adding a horizontal line with gray dashed style
#         plt.axhline(y=threshold, color='gray', linestyle='--')
#     else:
#         threshold = 0
#     plt.xlabel('pos vs. neg')
#     plt.grid(axis='x')
#     plt.xticks([0, 1], ['pos', 'neg'])
#     plt.ylabel(OCF_DATA_COLUMN_NAME)
#     plt.title(f'{fig_title} ({get_nb_rows(positive_dataset)}pos/{get_nb_rows(negative_dataset)}neg)', fontsize=8)
#
#     plt.xlim(-1, 2)  # Setting wider x-axis limits to create space
#     plt.ylim(0, max(round(1.1 * dataset_ocf[OCF_DATA_COLUMN_NAME].max()) + 1, threshold + 1))
#     plt.grid(True)
#     plt.tight_layout()
#
#     plt.savefig(fig_file)
#     plt.close()


# def create_ocf_plot(dataset_ocf, fig_file, fig_title, threshold=None):
#     # Separate positive and negative data
#     positive_dataset = dataset_ocf[dataset_ocf[TARGET_DATA_COLUMN_NAME] == 1]
#     negative_dataset = dataset_ocf[dataset_ocf[TARGET_DATA_COLUMN_NAME] == 0]
#
#     # Add a new column for the x-axis position using .loc to avoid SettingWithCopyWarning
#     positive_dataset = positive_dataset.copy()  # Ensure it's a separate DataFrame
#     negative_dataset = negative_dataset.copy()  # Ensure it's a separate DataFrame
#
#     # Add jitter to spread dots horizontally, centering positive instances around 0 and negative around 1
#     positive_dataset.loc[:, 'jittered_pos_neg'] = np.random.normal(0, FIG_DOT_STANDARD_DEVIATION,
#                                                             size=len(positive_dataset))  # Jitter around 0 (positive)
#     negative_dataset.loc[:, 'jittered_pos_neg'] = np.random.normal(1, FIG_DOT_STANDARD_DEVIATION,
#                                                             size=len(negative_dataset))  # Jitter around 1 (negative)
#
#     plt.figure(figsize=(6, 8))
#
#     # Scatter plot with jittered positions for both positive and negative instances
#     plt.scatter(positive_dataset['jittered_pos_neg'], positive_dataset[OCF_DATA_COLUMN_NAME], color='blue', alpha=FIG_DOT_TRANSPARENCY,
#                 label='pos')
#     plt.scatter(negative_dataset['jittered_pos_neg'], negative_dataset[OCF_DATA_COLUMN_NAME], color='red', alpha=FIG_DOT_TRANSPARENCY,
#                 label='neg')
#
#     # Add threshold line if it exists
#     if threshold:
#         plt.axhline(y=threshold, color='gray', linestyle='--')
#     else:
#         threshold = 0
#
#     # Add labels, title, grid, and other plot formatting
#     plt.xlabel('pos vs. neg')
#     plt.grid(axis='x')
#     plt.xticks([0, 1], ['pos', 'neg'])  # Correct x-tick labels
#     plt.ylabel(OCF_DATA_COLUMN_NAME)
#     plt.title(f'{fig_title} ({get_nb_rows(positive_dataset)}pos/{get_nb_rows(negative_dataset)}neg)', fontsize=FIG_TITLE_FONT_SIZE)
#
#     # # Adjust limits for better visualization
#     # plt.xlim(-1, 2)  # Wider x-axis limits to create space
#     # plt.ylim(0, max(round(1.1 * dataset_ocf[OCF_DATA_COLUMN_NAME].max()) + 1, threshold + 1))
#     min_y = min(dataset_ocf[OCF_DATA_COLUMN_NAME].min(), threshold)
#     max_y = dataset_ocf[OCF_DATA_COLUMN_NAME].max()
#
#     # Calculate range and adjust ylim to have equal space above max and below min
#     y_range = max_y - min_y
#     plt.ylim(min_y - y_range, max_y + y_range)
#
#     plt.xlim(-1, 2)  # Wider x-axis limits to create space
#
#     plt.grid(True)
#     plt.tight_layout()
#
#     # Save and close the figure
#     plt.savefig(fig_file)
#     plt.close()


def create_metric_plot(list_all_thresholds, list_all_metric_values, fig_file, fig_title):
    plt.plot(list_all_thresholds, list_all_metric_values)

    # Adding labels and title
    plt.xlabel(f'Threshold')
    plt.ylabel(f'Metric value')
    plt.title(fig_title)

    # Show the plot
    plt.savefig(fig_file)
    plt.close()


# This function lists all possible thresholds separating +/- instances: from the min OCF value to the max OCF value
# found in the dataset
def get_list_thresholds(dataset):
    # dataset assumes to have 'OCF value' computed for each row
    min_threshold = get_min_value(dataset, OCF_DATA_COLUMN_NAME)
    max_threshold = get_max_value(dataset, OCF_DATA_COLUMN_NAME)

    return [x + min_threshold for x in range(max_threshold - min_threshold + 1)]


# For each threshold in list_all_thresholds, compute the metric and store it in another array list_all_metric_values
def get_list_thresholds_to_metric_values(dataset, list_all_thresholds, metric_name):
    list_all_metric_values = [0 for i in range(len(list_all_thresholds))]

    for i in range(len(list_all_thresholds)):
        thr = list_all_thresholds[i]
        y_true = dataset[TARGET_DATA_COLUMN_NAME].values
        y_pred_values = [1 if val <= thr else 0 for val in dataset[OCF_DATA_COLUMN_NAME]]
        list_all_metric_values[i] = calculate_metric(y_true, y_pred_values, metric_name)

    return list_all_metric_values


# Here the best threshold is computed very arbitrarily: we just take the value in between the min and the max
# values in the list of all 'best thresholds', i.e., those associated with a maximal metric
# Also return the actual value for the metric
def get_best_threshold_value(list_all_thresholds, list_all_metric_values):
    max_value = max(list_all_metric_values)  # Find the maximum value in the list
    max_indices = [index for index, value in enumerate(list_all_metric_values) if value == max_value]
    # Select the index in the middle
    return [list_all_thresholds[max_indices[len(max_indices) // 2]], max_value]


# We want to save only ONE threshold used to get a binary classifier
# This threshold will be saved in 'file overall best threshold'
# For now, we just choose a weighted average prioritizing
# Youden > MCC > Jaccard > f1_score > accuracy > precision > recall
# (current weight: 64 / 32 / 16 / 8 / 4 / 2 / 1)
# This function return the normalized aggregated list of values
def get_list_all_overall_metric_values(list_all_thresholds, list_all_metric_values_for_all_metrics):
    # We create a list of aggregated values
    list_all_overall_metric_values = [0] * len(list_all_thresholds)
    for metric in list_all_metric_values_for_all_metrics:
        if metric[0] == METRIC_YOUDEN:
            multiplier = 64
        elif metric[0] == METRIC_MCC:
            multiplier = 32
        elif metric[0] == METRIC_JACCARD:
            multiplier = 16
        elif metric[0] == METRIC_F1_SCORE:
            multiplier = 8
        elif metric[0] == METRIC_ACCURACY:
            multiplier = 4
        elif metric[0] == METRIC_PRECISION:
            multiplier = 2
        else:
            multiplier = 1
        list_all_metric_values = metric[1]
        list_all_overall_metric_values = [list_all_overall_metric_values[i] + (list_all_metric_values[i] * multiplier)
                                          for i in range(len(list_all_thresholds))]

    return [list_all_overall_metric_values[i] / 127 for i in range(len(list_all_overall_metric_values))]


if __name__ == "__main__":
    all_dataset_names = sorted_by_increasing_size(get_all_datasets_names())

    #all_dataset_names = ['161-Mammographic Mass_961r_5f']

    print(f'nb of datasets: {len(all_dataset_names)}:')
    for ds_name in all_dataset_names:
        print(ds_name)

    with open(get_all_datasets_log_file_name(), 'a') as log_file:
        log_file.write(datetime.now().strftime("[%Y-%m-%d %H:%M:%S]") + ' ' + FILLED_LINE + '\n')
        log_file.write(datetime.now().strftime("[%Y-%m-%d %H:%M:%S]") + f' Nb of datasets: {len(all_dataset_names)}\n')

    cpt_dataset = 0
    for dataset_name in all_dataset_names:
        cpt_dataset += 1
        print(f'Processing #{cpt_dataset}/#{len(all_dataset_names)} CLASS dataset {dataset_name}...')

        # The following few lines are only for the log message scoped to the dataset
        start_time_dataset = time.time()
        message_log_dataset = [FILLED_LINE]
        for message in get_description_dataset(my_from_csv(get_dataset_file_path(dataset_name)), 'Dataset'):
            message_log_dataset.append(message)

        # For every sampling directory, compute the OCFed training dataset
        # List all sampling directory paths for the corresponding dataset
        all_sampling_directories = get_all_sampling_dir_paths(dataset_name)

        for sampling_dir_name in all_sampling_directories:
            start_time_sampling = time.time()

            id_sampling = get_id_sampling_from_dir_path(sampling_dir_name)

            print(f'Processing the train dataset {get_sampling_train_ocfhamming_file_path(dataset_name, id_sampling)}...')
            # If the sampling has already been trained, skip it
            if os.path.exists(get_train_withOCF_file_name(dataset_name, id_sampling)) and os.path.exists(get_best_threshold_file_name(dataset_name, id_sampling)):
                print(f'Already trained! (files {get_train_withOCF_file_name(dataset_name, id_sampling)} and '
                      f'{get_fig_train_best_threshold_file_name(dataset_name, id_sampling)} already exist)\n'
                      f'--. only (re-)draw some curves and the OCFs into plots')
                # We read the train dataset with OCF
                df_train_with_ocf = my_from_csv(get_train_withOCF_file_name(dataset_name, id_sampling))
                # We retrieve the threshold
                with open(get_best_threshold_file_name(dataset_name, id_sampling), 'r') as best_thr_file:
                    best_thr = int(best_thr_file.read())
            else:
                # We first read the training dataset (it should be already scaled for OCFHamming)
                df_train_initial = my_from_csv(get_sampling_train_ocfhamming_file_path(dataset_name, id_sampling))

                print(f'Creating the OCFed train dataset {get_train_withOCF_file_name(dataset_name, id_sampling)}...')
                # We augment the training dataset with a new column representing the distance between the data
                # corresponding to the current row and the full dataset

                # in the case of binarized datasets, we don't need scaling info
                # otherwise we need and we use another way to compute distances
                # Then we augment the training dataset recently augmented with distances with a new column: the OCF value.
                # This value corresponds to distance - min_value, where min_value is NOT the minimal distance found in
                # the dataset itself, but the minimal distance that any world could have to the dataset
                # (this minimal distance is computed in polynomial time)
                # .csv file of type
                df_type = my_from_csv(get_dataset_feature_type_list(dataset_name))

                # computing the distances and ocf
                df_train_with_distances = get_dataset_with_distances_scaled(df_train_initial, df_type, ext_dataset=None)
                df_train_with_ocf, shift_value = convert_from_distances_to_ocf_scaled(df_train_with_distances, df_type)

                df_train_with_ocf.to_csv(get_train_withOCF_file_name(dataset_name, id_sampling), index=False)

                # We need to save in a file the shift value to withdraw from a distance to get the OCF value for later tests
                with open(get_OCF_to_distance_shift_value_file_name(dataset_name, id_sampling), 'w') as shift_value_file:
                    shift_value_file.write(str(shift_value) + '\n')
                print(f'OCFed dataset {get_train_withOCF_file_name(dataset_name, id_sampling)} created!')

                # At this stage all distances and OCF values have been computed and are stored into df_train_with_ocf

                # Storing some log message describing the OCFed train dataset
                message_log_sampling = [FILLED_LINE]
                for message in get_description_dataset(df_train_with_ocf, 'Training dataset'):
                    message_log_sampling.append(message)

                # At this point we have created an OCFed dataset and plotted the OCF graph (without threshold)
                # To get an actual classifier, we want to compute the best threshold, which we choose according
                # to the metric youden

                print(f'Computing the best threshold value according to the metric {METRIC_NAME_USED_FOR_BEST_THRESHOLD}')

                list_all_thresholds = get_list_thresholds(df_train_with_ocf)
                list_all_metric_values = get_list_thresholds_to_metric_values(df_train_with_ocf, list_all_thresholds,
                                                                              METRIC_NAME_USED_FOR_BEST_THRESHOLD)
                pair_best_thr_value = get_best_threshold_value(list_all_thresholds, list_all_metric_values)
                best_thr = pair_best_thr_value[0]

                # We draw the curve f(list_all_thresholds) = list_all_metric_values
                create_metric_plot(list_all_thresholds, list_all_metric_values, get_fig_train_curve_thresholds_values_file_name(dataset_name, id_sampling),
                                   get_fig_title(dataset_name, id_sampling, METRIC_NAME_USED_FOR_BEST_THRESHOLD))

                # We save the best threshold in a file
                with open(get_best_threshold_file_name(dataset_name, id_sampling), 'w') as best_thr_file:
                    best_thr_file.write(str(best_thr) + '\n')

                # Write a log saving the metric name, best threshold and corresponding value
                message_log_sampling.append(f'Metric {METRIC_NAME_USED_FOR_BEST_THRESHOLD}, '
                                            f'best threshold/value: {best_thr}/{pair_best_thr_value[1]}')

                end_time_sampling = time.time()
                execution_time = end_time_sampling - start_time_sampling
                message_time_taken = f'Time taken: {execution_time:.0f} seconds'
                message_log_sampling.append(message_time_taken)
                message_log_sampling.append(FILLED_LINE)
                print(message_log_sampling)
                print(f'Training dataset {get_sampling_name(dataset_name, id_sampling)} trained!')
                with open(get_sampling_train_log_file_name(dataset_name, id_sampling), 'a') as log_file:
                    for message in message_log_sampling:
                        log_file.write(datetime.now().strftime("[%Y-%m-%d %H:%M:%S]") + ' ' + message + '\n')

            # draw all curves and plots
            # We create a figure of the OCFed training dataset (without threshold)
            create_ocf_plot(df_train_with_ocf,
                            get_fig_train_no_threshold_file_name(dataset_name, id_sampling),
                            get_fig_title(dataset_name, id_sampling))

            # We create the OCF plot with the threshold
            create_ocf_plot(df_train_with_ocf, get_fig_train_best_threshold_file_name(dataset_name, id_sampling),
                            get_fig_title(dataset_name, id_sampling, METRIC_NAME_USED_FOR_BEST_THRESHOLD), threshold=best_thr)

        end_time_dataset = time.time()
        execution_time = end_time_dataset- start_time_dataset
        message_time_taken = f'Time taken for training all samplings: {execution_time:.0f} seconds'
        with open(get_dataset_log_file_name(dataset_name), 'a') as log_file:
            log_file.write(datetime.now().strftime("[%Y-%m-%d %H:%M:%S]") + ' ' + message_time_taken + '\n')
        with open(get_all_datasets_log_file_name(), 'a') as log_file:
            log_file.write(datetime.now().strftime("[%Y-%m-%d %H:%M:%S]") + f' Dataset #{cpt_dataset}/#{len(all_dataset_names)}: {dataset_name} - {message_time_taken}\n')
