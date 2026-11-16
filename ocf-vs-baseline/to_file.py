from constants import *
import os
import re
from pathlib import Path


def get_datasets_dir_path():
    return os.environ[DATASET_ROOT_PATH_ENV_VAR_NAME]


def get_all_results_plot_dir_path():
    return get_datasets_dir_path() + '/' + ALL_RESULTS_DIR_NAME


def get_all_datasets_names():
    return [entry for entry in os.listdir(get_datasets_dir_path()) if
            os.path.isdir(os.path.join(get_datasets_dir_path(), entry)) and
            get_all_results_plot_dir_path() not in os.path.join(get_datasets_dir_path(), entry)]


def get_dataset_dir_path(dataset_name):
    return get_datasets_dir_path() + '/' + dataset_name


def get_dataset_file_path(dataset_name):
    return get_dataset_dir_path(dataset_name) + '/' + dataset_name + '.csv'


def is_sampling_dir_path_name(dir_path):
    return re.search(DATASET_SAMPLING_NAME + r"-([0-9]+)", dir_path)


def get_sampling_dir_path(dataset_name, id_sampling, training_set_size=None):
    if training_set_size is None:
        pattern = DATASET_SAMPLING_NAME + '-' + str(id_sampling) + '-'

        matching_directories = [dir_name for dir_name in os.listdir(get_dataset_dir_path(dataset_name)) if
                                os.path.isdir(
                                    os.path.join(get_dataset_dir_path(dataset_name), dir_name)) and pattern in dir_name]

        if len(matching_directories) == 1:
            return get_dataset_dir_path(dataset_name) + '/' + matching_directories[0]
        elif len(matching_directories) > 1:
            raise ValueError("Multiple directories match the criteria.")
        else:
            raise ValueError("No directory matches the criteria.")
    else:
        return (get_dataset_dir_path(dataset_name) + '/' + dataset_name + DATASET_SAMPLING_NAME + '-'
                + str(id_sampling) + '-' + f'{training_set_size}_{100 - training_set_size}')


# to use only when we want the *name* of the sampling, not the full path of the corresponding directory
def get_sampling_name(dataset_name, id_sampling):
    return get_sampling_dir_path(dataset_name, id_sampling).split('/')[-1]


def get_id_sampling_from_dir_path(dir_path):
    match = re.search(DATASET_SAMPLING_NAME + r"-([0-9]+)", dir_path)
    if match:
        return int(match.group(1))  # Extract the number and convert it to an integer
    return None


# This procedure is generic and is used in further procedures
def get_sampling_file_path(dataset_name, id_sampling, training, imputed_and_scaled, one_hot_encoded, ocfHamming):
    sampling_dir_name = get_sampling_dir_path(dataset_name, id_sampling)
    if training:
        sampling_file_name = sampling_dir_name.split('/')[-1] + DATASET_SAMPLING_TRAIN_NAME
    else:
        sampling_file_name = sampling_dir_name.split('/')[-1] + DATASET_SAMPLING_TEST_NAME

    if imputed_and_scaled:
        sampling_file_name += DATASET_STANDARDIMPUTEDSCALED
    if one_hot_encoded:
        sampling_file_name += DATASET_ONEHOTENCODED
    if ocfHamming:
        sampling_file_name += DATASET_SCALED_FOR_OCFHAMMING

    sampling_file_name += '.csv'

    return sampling_dir_name + '/' + sampling_file_name


def get_sampling_train_original_file_path(dataset_name, id_sampling):
    return get_sampling_file_path(dataset_name, id_sampling, training=True, imputed_and_scaled=False,
                                  one_hot_encoded=False, ocfHamming=False)


def get_sampling_test_original_file_path(dataset_name, id_sampling):
    return get_sampling_file_path(dataset_name, id_sampling, training=False, imputed_and_scaled=False,
                                  one_hot_encoded=False, ocfHamming=False)


def get_sampling_train_existingmodel_file_path(dataset_name, id_sampling):
    return get_sampling_file_path(dataset_name, id_sampling, training=True, imputed_and_scaled=True,
                                  one_hot_encoded=True, ocfHamming=False)


def get_sampling_test_existingmodel_file_path(dataset_name, id_sampling):
    return get_sampling_file_path(dataset_name, id_sampling, training=False, imputed_and_scaled=True,
                                  one_hot_encoded=True, ocfHamming=False)


def get_sampling_train_ocfhamming_file_path(dataset_name, id_sampling):
    return get_sampling_file_path(dataset_name, id_sampling, training=True, imputed_and_scaled=True,
                                  one_hot_encoded=False, ocfHamming=True)


def get_sampling_test_ocfhamming_file_path(dataset_name, id_sampling):
    return get_sampling_file_path(dataset_name, id_sampling, training=False, imputed_and_scaled=True,
                                  one_hot_encoded=False, ocfHamming=True)


def get_sampling_scaling_parameters_file_path(dataset_name, id_sampling):
    sampling_dir_name = get_sampling_dir_path(dataset_name, id_sampling)

    return sampling_dir_name + '/' + sampling_dir_name.split('/')[-1] + DATASET_SCALING_PARAMETERS + '.csv'


def get_sampling_train_log_file_name(dataset_name, id_sampling):
    return get_train_withOCF_base_file_name(dataset_name, id_sampling) + LOG_NAME + '.txt'


def get_sampling_test_log_file_name(dataset_name, id_sampling):
    return get_test_withOCF_base_file_name(dataset_name, id_sampling) + LOG_NAME + '.txt'


def get_all_sampling_dir_paths(dataset_name):
    result = [dir_name for dir_name in os.listdir(get_dataset_dir_path(dataset_name)) if
              os.path.isdir(os.path.join(get_dataset_dir_path(dataset_name),
                                         dir_name)) and is_sampling_dir_path_name(dir_name)]
    return sorted(result) if result else None


def get_max_id_sample(dataset_name):
    all_sampling_dir_paths = get_all_sampling_dir_paths(dataset_name)

    if all_sampling_dir_paths:
        return max([get_id_sampling_from_dir_path(dir_path) for dir_path in all_sampling_dir_paths])
    else:
        return 0

    # List directory names
    # directory_names = os.listdir(f'{get_datasets_dir_path()}/{dataset_name}/')
    # pattern = DATASET_SAMPLING_NAME + r"-([0-9]+)"
    # substring_list = []
    # for directory in directory_names:
    #    match = re.search(pattern, directory)
    #    if match:
    #        extracted_number = int(match.group(1))  # Extracting the integer part
    #        substring_list.append(extracted_number)
    # if not substring_list:
    #    max_id = 0
    # else:
    #    max_id = max(substring_list)
    #
    # return max_id + 1


# checks if the given sampling has been trained according to our own OCFHamming model
def check_if_sampling_is_OCFHammingtrained(dataset_name, id_sampling):
    return Path(get_train_withOCF_file_name(dataset_name, id_sampling)).exists()


# checks if at least one of the existing samplings have been trained according to our own OCFHamming model
def check_if_at_least_one_sampling_is_OCFHammingtrained(dataset_name):
    all_sampling_dir_paths = get_all_sampling_dir_paths(dataset_name)
    if not all_sampling_dir_paths:
        # True by vacuity
        return True
    all_sampling_ids = [get_id_sampling_from_dir_path(sampling_dir_path) for sampling_dir_path in
                        all_sampling_dir_paths]
    for id_sampling in all_sampling_ids:
        if check_if_sampling_is_OCFHammingtrained(dataset_name, id_sampling):
            return True
    return False


# checks if ALL existing samplings have been trained according to our own OCFHamming model
def check_if_all_sampling_have_been_OCFHammingtrained(dataset_name):
    all_sampling_dir_paths = get_all_sampling_dir_paths(dataset_name)
    if not all_sampling_dir_paths:
        # True by vacuity
        return True
    all_sampling_ids = [get_id_sampling_from_dir_path(sampling_dir_path) for sampling_dir_path in
                        all_sampling_dir_paths]
    for id_sampling in all_sampling_ids:
        if not check_if_sampling_is_OCFHammingtrained(dataset_name, id_sampling):
            return False
    return True


def get_all_results_file_name(dataset_name):
    return get_dataset_dir_path(dataset_name) + '/' + dataset_name + ALL_RESULTS + '.csv'


def get_all_datasets_log_file_name():
    return get_datasets_dir_path() + '/' + LOG_NAME + '.txt'


def get_dataset_log_file_name(dataset_name):
    return get_dataset_dir_path(dataset_name) + '/' + dataset_name + LOG_NAME + '.txt'


def get_dataset_feature_type_list(dataset_name):
    return get_dataset_dir_path(dataset_name) + '/' + dataset_name + DATASET_TYPE + '.csv'


# for generic use only within this file
def get_train_withOCF_base_file_name(dataset_name, id_sampling):
    sampling_dir_name = get_sampling_dir_path(dataset_name, id_sampling)

    return sampling_dir_name + '/' + sampling_dir_name.split('/')[-1] + DATASET_SAMPLING_TRAIN_NAME + DATASET_OCF_NAME


def get_train_withOCF_file_name(dataset_name, id_sampling):
    return get_train_withOCF_base_file_name(dataset_name, id_sampling) + '.csv'


def get_OCF_to_distance_shift_value_file_name(dataset_name, id_sampling):
    return get_train_withOCF_base_file_name(dataset_name, id_sampling) + SHIFT_VALUE_NAME + '.txt'


### about figures
def get_fig_train_no_threshold_file_name(dataset_name, id_sampling):
    return get_train_withOCF_base_file_name(dataset_name, id_sampling) + OCF_FIG_NAME_WITHOUT_THRESHOLD + '.png'


def get_fig_train_curve_thresholds_values_file_name(dataset_name, id_sampling):
    return get_train_withOCF_base_file_name(dataset_name, id_sampling) + CURVE_FIG_NAME + '.png'


def get_fig_train_best_threshold_file_name(dataset_name, id_sampling):
    return get_train_withOCF_base_file_name(dataset_name, id_sampling) + BEST_THRESHOLD_NAME + '.png'


def get_fig_title(dataset_name, id_sampling, metric_name=None):
    if metric_name:
        return get_sampling_name(dataset_name, id_sampling) + '-' + metric_name
    else:
        return get_sampling_name(dataset_name, id_sampling)


### about thresholds
def get_best_threshold_file_name(dataset_name, id_sampling):
    return get_train_withOCF_base_file_name(dataset_name, id_sampling) + BEST_THRESHOLD_NAME + '.txt'


# for generic use only within this file
def get_test_withOCF_base_file_name(dataset_name, id_sampling):
    sampling_dir_name = get_sampling_dir_path(dataset_name, id_sampling)

    return sampling_dir_name + '/' + sampling_dir_name.split('/')[-1] + DATASET_SAMPLING_TEST_NAME + DATASET_OCF_NAME


def get_test_withOCF_file_name(dataset_name, id_sampling):
    return get_test_withOCF_base_file_name(dataset_name, id_sampling) + '.csv'


### about figures
def get_fig_test_no_threshold_file_name(dataset_name, id_sampling):
    return get_test_withOCF_base_file_name(dataset_name, id_sampling) + OCF_FIG_NAME_WITHOUT_THRESHOLD + '.png'


def get_fig_test_with_threshold_file_name(dataset_name, id_sampling):
    return get_test_withOCF_base_file_name(dataset_name, id_sampling) + WITH_THRESHOLD_NAME + '.png'


def get_fig_test_roc_file_name(dataset_name, id_sampling):
    return get_test_withOCF_base_file_name(dataset_name, id_sampling) + ROC_FIG_NAME + '.png'
