import os
import pandas as pd

from constants import *
from sklearn.model_selection import train_test_split
from dataop import my_from_csv, my_to_csv, sorted_by_increasing_size
from my_scaling import impute_and_scale_sampling, one_hot_encode_sampling, my_scaling_dataset_train, \
    my_scaling_dataset_test
from to_file import *


def create_samplings(dataset_name, training_set_size, nb_samplings):
    dataset = my_from_csv(get_dataset_file_path(dataset_name))

    max_id = get_max_id_sample(dataset_name)
    for cpt in range(nb_samplings):
        sampling_id = max_id + cpt
        train_dataset, test_dataset = train_test_split(dataset, test_size=(
                    (100 - training_set_size) / 100), stratify=dataset[TARGET_DATA_COLUMN_NAME])
        directory_name = get_sampling_dir_path(dataset_name, sampling_id, training_set_size=training_set_size)
        os.makedirs(directory_name, exist_ok=True)

        # Here the training / test sets are not scaled, not imputed, not one-hot-encoded
        my_to_csv(train_dataset, get_sampling_train_original_file_path(dataset_name, sampling_id))
        my_to_csv(test_dataset, get_sampling_test_original_file_path(dataset_name, sampling_id))

        # Now the training / test sets are imputed and scaled
        imputedscaled_train_dataset, imputedscaled_test_dataset = impute_and_scale_sampling(train_dataset, test_dataset)

        # Then they are one-hot-encoded
        encoded_train_dataset, encoded_test_dataset = one_hot_encode_sampling(imputedscaled_train_dataset, imputedscaled_test_dataset)
        my_to_csv(encoded_train_dataset, get_sampling_train_existingmodel_file_path(dataset_name, sampling_id))
        my_to_csv(encoded_test_dataset, get_sampling_test_existingmodel_file_path(dataset_name, sampling_id))

        # Now we want to adapt the samplings for our own scaled Hamming distance model
        # We use as a base the training / test datasets that are imputed and scaled above
        # We first retrieve the type of each feature
        df_type = pd.read_csv(get_dataset_dir_path(dataset_name) + '/' + dataset_name + DATASET_TYPE + '.csv')
        # Below, my_scaled_df is the scaled train dataset, and scale_info_df is the dataframe storing all scaling info
        my_scaled_df_train, scale_info_df = my_scaling_dataset_train(imputedscaled_train_dataset, df_type)
        # Here, we save the scale_info_df (suffixed by -scaling-param) .csv file for debugging
        sampling_dir_name = get_sampling_dir_path(dataset_name, sampling_id)
        my_to_csv(scale_info_df, get_sampling_scaling_parameters_file_path(dataset_name, sampling_id))
        # And we save the scaled train dataset
        my_to_csv(my_scaled_df_train, get_sampling_train_ocfhamming_file_path(dataset_name, sampling_id))
        # Now on to the test dataset
        # rescale the test dataset accordingly
        my_scaled_df_test = my_scaling_dataset_test(imputedscaled_test_dataset, scale_info_df)
        my_to_csv(my_scaled_df_test, get_sampling_test_ocfhamming_file_path(dataset_name, sampling_id))

        print(f'sampling generated for {dataset_name}: #{sampling_id}')


if __name__ == "__main__":
    # flag to put at True if we want to generate samplings even when they already exist
    force_generation = False

    # nb_samplings = the number of batches
    nb_desired_samplings = 10

    # training_set_size = the proportion training set size / test set size
    training_dataset_size = 90

    all_dataset_names = sorted_by_increasing_size(get_all_datasets_names())
    nb_datasets = len(all_dataset_names)

    cpt = 0
    for ds_name in all_dataset_names:
        cpt += 1
        print(f'Processing #{cpt}/#{nb_datasets} CLASS dataset {ds_name}...')
        if force_generation:
            create_samplings(ds_name, training_dataset_size, nb_desired_samplings)
        else:
            all_existing_sampling_directories = get_all_sampling_dir_paths(ds_name)
            nb_existing_sampling_directories = len(all_existing_sampling_directories) if all_existing_sampling_directories else 0
            if nb_existing_sampling_directories < nb_desired_samplings:
                create_samplings(ds_name, training_dataset_size, nb_desired_samplings - nb_existing_sampling_directories)
            else:
                print(f'Already {nb_existing_sampling_directories} samplings exist (desired number: {nb_desired_samplings})')
