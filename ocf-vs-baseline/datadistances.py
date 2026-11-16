from constants import *
import distances
import dataop
import statistics
import pandas as pd
import numpy as np

from dataop import get_column_value_dataset_as_value_list


def compute_distance_row_to_dataset(row, dataset, distance_function):
    row_as_world = row[row.index.str.startswith('X_')].tolist()
    # Perform some computation using the row and dataset
    return distances.distance_from_model_to_dataset(row_as_world, dataset, distance_function)


# given an initial dataset, return a new dataset where a column is added at the end of each row: the value is the
# distance between the data and the ext_dataset where each negative instance (y = 0) is negated (new_x = 1 - x)
def get_dataset_with_distances(dataset, distance_function, ext_dataset=None):
    if ext_dataset is None:
        # then the ext_dataset is the dataset itself
        ext_dataset = dataset
    result_dataset = dataset.copy()
    # Applying the function across each row using apply

    result_dataset[DISTANCE_DATA_COLUMN_NAME] = result_dataset.apply(compute_distance_row_to_dataset, axis=1,
                                                                     dataset=ext_dataset,
                                                                     distance_function=distance_function)
    return result_dataset


def calculate_cell_distance_scaled(r_val, e_val, feature_type):
    """Calculate distance between individual cell values based on feature types."""
    if feature_type in ['Integer', 'Continuous']:
        return abs(r_val - e_val)
    else:
        if r_val == e_val:
            return 0
        else:
            return DISTANCE_BETWEEN_TWO_CATEGORICAL_OR_BINARY_DIFFERENT_VALUES


def calculate_distance_scaled(r, e_row, feature_types):
    """Calculate distance between row r and e_row based on feature types."""
    total_distance = 0

    for feature in r.index:
        if feature == TARGET_DATA_COLUMN_NAME or feature not in feature_types:
            continue  # Skip target column or features not in feature_types

        r_val = r[feature]
        e_val = e_row[feature]
        feature_type = feature_types.get(feature, None)

        cell_distance = calculate_cell_distance_scaled(r_val, e_val, feature_type)
        #print(f'distance between {r_val} and {e_val}: {cell_distance}')

        if e_row[TARGET_DATA_COLUMN_NAME] == 1:
            #print(f'THIS ROW IS +++')
            total_distance += cell_distance
        else:
            #print(f'THIS ROW IS ---')
            total_distance += (MAX_SCALING_VALUE - cell_distance)
        #print(f'distance between {r_val} and {e_val}\n is currently {total_distance}')

    return total_distance


def get_dataset_with_distances_scaled(dataset, df_type, ext_dataset=None):
    if ext_dataset is None:
        ext_dataset = dataset

    # Convert to DataFrame if ext_dataset is a path
    if not isinstance(ext_dataset, pd.DataFrame):
        ext_dataset = pd.read_csv(ext_dataset)

    # Extract feature types
    feature_types = df_type.set_index(FEATURE_NAME_COLUMN_NAME)[FEATURE_TYPE_COLUMN_NAME].to_dict()

    # Initialize the distance column
    dataset[DISTANCE_DATA_COLUMN_NAME] = 0

    # Calculate distances
    for i, r in dataset.iterrows():
        #print(f'TREATING ROW {r}\n')
        total_distance = 0

        for _, e_row in ext_dataset.iterrows():
            #print(f'-> COMPUTING DISTANCE WITH ROW {e_row}\n')
            total_distance += calculate_distance_scaled(r, e_row, feature_types)
            #print(f'THEN total distance IS CURRENTLY {total_distance}')

        dataset.at[i, DISTANCE_DATA_COLUMN_NAME] = total_distance

    return dataset


def convert_from_distances_to_ocf_to_remove(dataset, shift_value=None):
    if shift_value is None:
        shift_value = distances.get_min_distance_to_dataset(dataset)

    result_dataset = dataset.copy()
    result_dataset[OCF_DATA_COLUMN_NAME] = result_dataset[DISTANCE_DATA_COLUMN_NAME].apply(lambda x: x - shift_value)

    return result_dataset, shift_value


def convert_from_distances_to_ocf_scaled(dataset, df_type, shift_value=None):
    if shift_value is None:
        shift_value = distances.get_min_distance_to_dataset_scaled(dataset, df_type)

    result_dataset = dataset.copy()
    #print(f'The shift value is: {shift_value}')
    result_dataset[OCF_DATA_COLUMN_NAME] = result_dataset[DISTANCE_DATA_COLUMN_NAME].apply(lambda x: x - shift_value)

    return result_dataset, shift_value


def get_filtered_data(dataset, sign=-1):
    # sign=-1 by default, meaning that no rows are filtered
    # if sign=1, only positive instances are considered
    # if sign=0, only negative instances are considered
    assert (sign == -1 or sign == 0 or sign == 1)
    if sign == -1:
        return dataset.copy()
    return dataset[dataset['y'] == sign]  # Select rows where the second-to-last column is sign


def get_min_value(dataset, column_name):
    return min(get_column_value_dataset_as_value_list(dataset, column_name))


def get_max_value(dataset, column_name):
    return max(get_column_value_dataset_as_value_list(dataset, column_name))


def get_avg_value(dataset, column_name):
    distance_list = get_column_value_dataset_as_value_list(dataset, column_name)
    return sum(distance_list) / len(distance_list)


def get_median_value(dataset, column_name):
    return statistics.median(get_column_value_dataset_as_value_list(dataset, column_name))


if __name__ == "__main__":
    # dataset_name = "rbm-small-duplicates"
    dataset_name = "rbm-small"
    # dataset_name = "rbm"
    df = dataop.my_from_csv(f'{dataset_name}.csv')

    print('Some small dataset')
    print(df)

    print('\nTarget dataset')
    print(dataop.get_negated_negatives(df))

    print('\nDataset augmented with distances')
    dataset_with_distances = get_dataset_with_distances(df, distance_function=distances.hamming_distance)
    print(dataset_with_distances)

    dataset_with_ocf = convert_from_distances_to_ocf_scaled(dataset_with_distances)
    print(f'\nDataset augmented with OCF values:')
    print(dataset_with_ocf)

    print('\nOverall stats:')
    print(f"Minimum: {get_min_value(dataset_with_ocf, OCF_DATA_COLUMN_NAME)}")
    print(f"Maximum: {get_max_value(dataset_with_ocf, OCF_DATA_COLUMN_NAME)}")
    print(f"Average: {get_avg_value(dataset_with_ocf, OCF_DATA_COLUMN_NAME)}")
    print(f"Median: {get_median_value(dataset_with_ocf, OCF_DATA_COLUMN_NAME)}")
