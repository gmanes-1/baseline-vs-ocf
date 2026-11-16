from constants import *
import dataop


def distance(model1, model2, factor=1, range_dist=None):
    """
    Computes the Hamming distance between model1 and model2, powered by factor and bounded by range_dist^factor.
    By default, the optional parameter factor is set to factor=1, resulting in the Hamming distance.
    range_dist is used as an upper bound for the neighborhood relevant to the use case.
    So, this function returns min((dH(model1, model2))^factor, range_dist^factor)

    Using values of factor>1 is only appropriate when improving worlds (not decrementing them)
    Using values of factor>1 and range_dist<len(model1) are never appropriate

    :param model1: 0-1 vector, the first world.
    :param model2: 0-1 vector, the second world.
    :param factor: int, the power factor applied after computation of the Hamming distance.
    :param range_dist: int, an upper bound for neighborhood used in the resulted distance
    :return: int, the Hamming distance between model1 and model2, powered by factor and bounded by range_dist^factor.
    """
    assert len(model1) == len(model2)
    if range_dist is None:
        range_dist = pow(len(model1), factor)
    else:
        range_dist = pow(range_dist, factor)
    result = 0
    for i in range(len(model1)):
        if model1[i] != model2[i]:
            result = result + 1
    return min(pow(result, factor), range_dist)


def drastic_distance(model1, model2):
    """
    Computes the drastic distance between model1 and model2

    :param model1: 0-1 vector, the first world
    :param model2: 0-1 vector, the second world
    :return: the drastic distance between model1 and model2, i.e., 0 if model1 == model2, 1 otherwise.
    """
    assert len(model1) == len(model2)
    for i in range(len(model1)):
        if model1[i] != model2[i]:
            return 1
    return 0


def hamming_distance(model1, model2, range_dist=None):
    """
    Computes the Hamming distance between model1 and model2, possibly bounded by range_dist.
    range_dist is used as an upper bound for the neighborhood relevant to the use case.
    So, this function returns min((dH(model1, model2)), range_dist)

    :param model1: 0-1 vector, the first world.
    :param model2: 0-1 vector, the second world.
    :param range_dist: int, an upper bound for neighborhood used in the resulted distance
    :return: int, the Hamming distance between model1 and model2, possibly bounded by range_dist^factor.
    """
    return distance(model1, model2, factor=1, range_dist=range_dist)


def quadratic_distance(model1, model2):
    return distance(model1, model2, factor=2)


def cubic_distance(model1, model2):
    return distance(model1, model2, factor=3)


# returns the distance function according to an input string
def get_distance_function(name_distance):
    if name_distance == HAMMING_NAME:
        return hamming_distance
    elif name_distance == DRASTIC_NAME:
        return drastic_distance
    raise ValueError(f'The distance {name_distance} does not exist')


def distance_from_model_to_row(model, row, distance_function):
    """
    Computes the distance between model (a 0-1 vector) and a row of a
    panda dataset, using the distance function distance_function

    :param model: 0-1 vector
    :param row: the row where the second model used in the computation of the distance lies
    :param distance_function: the distance function used to compute the distance. This function will use default
    values for factor and range_dist
    only two parameters (the two worlds whose distance needs to be computed)
    :return: the distance between model and the data of dataset at row row_index, according to distance_function.
    If for the row 'y=0' then return nb_features - dist (same as flip all bits and compute the distance)
    """
    if row[TARGET_DATA_COLUMN_NAME] == 1:
        row_values = [int(value) for column, value in row.items() if column.startswith('X_')]
    else:
        row_values = [1 - int(value) for column, value in row.items() if column.startswith('X_')]
    return distance_function(model, row_values)


def distance_from_model_to_dataset(model, dataset, distance_function):
    """
    Computes the sum of the distances from model to each row of dataset, using the distance function distance_function

    :param model: 0-1 vector
    :param dataset: a panda dataset, where each row is a 0-1 vector
    :param distance_function: the distance function used to compute the distance. This function will use default
    values for factor and range_dist
    :return: the distance between model and the panda dataset dataset, using distance_function
    """
    result = 0
    for i in range(dataop.get_nb_rows(dataset)):
        result = result + distance_from_model_to_row(model, dataset.iloc[i], distance_function)
    return result


def get_min_distance_to_dataset(dataset):
    """
    Computes the minimal possible distance of a world to a dataset, i.e., min({d(\omega, dataset) | \omega is a world})
    The computation (equivalently, the construction of the optimal world \omega) is made column-wise: for each column i,
    set \omega(i)=0 iff #0(i)>=#1(i), i.e., iff the number of 0s in that column i is greater than the number of 1s. Then
    the resulting minimal value is equal to the sum for each column i of min(#0(i), #1(i))

    :param dataset: a panda dataset
    :return: int, the minimal possible distance of a world to a dataset (min({d(\omega, dataset) | \omega is a world}))
    """
    min_dist = 0
    formated_dataset = dataop.get_negated_negatives(dataset)
    for column in [col for col in formated_dataset.columns if col.startswith('X_')]:
        min_dist += min((formated_dataset[column] == 0).sum(), (formated_dataset[column] == 1).sum())
    return min_dist


def get_min_distance_to_dataset_scaled(dataset, df_type):
    """
    Compute the min_possible_value based on feature types and their values in the dataset.

    Parameters:
    - dataset (pd.DataFrame): The dataset containing the features.
    - scaling_param_df (pd.DataFrame): DataFrame containing feature names and types from -scaling-param.csv.

    Returns:
    - min_possible_value (float): The computed minimum possible value.
    """
    # Extract feature types
    feature_types = df_type.set_index('FEATURE_NAME')['FEATURE_TYPE'].to_dict()

    # Initialize min_possible_value
    min_possible_value = 0
    #print(f'current min possible value = {min_possible_value}')
    for feature in dataset.columns:
        if feature.startswith('X_'):  # Skip the target column
            feature_type = feature_types.get(feature, None)

            if feature_type in ['Integer', 'Continuous']:
                # Compute the average value for Integer and Continuous features
                avg_value = dataset[feature].mean()
                min_possible_value += avg_value
            else:
                # For other types, add MAX_SCALING_VALUE
                min_possible_value += MAX_SCALING_VALUE
        #print(f'current min possible value = {min_possible_value}')

    #print(f'final min possible value = {min_possible_value}')
    return round(min_possible_value)


def get_max_distance_to_dataset(dataset):
    """
    Computes the minimal possible distance of a world to a dataset, i.e., max({d(\omega, dataset) | \omega is a world})
    The computation (equivalently, the construction of the optimal world \omega) is made column-wise: for each column i,
    set \omega(i)=0 iff #0(i)<=#1(i), i.e., iff the number of 0s in that column i is lower than the number of 1s. Then
    the resulting minimal value is equal to the sum for each column i of max(#0(i), #1(i))

    :param dataset: a panda dataset
    :return: int, the maximal possible distance of a world to a dataset (max({d(\omega, dataset) | \omega is a world}))
    """
    max_dist = 0
    formated_dataset = dataop.get_negated_negatives(dataset)
    for column in [col for col in formated_dataset.columns if col.startswith('X_')]:
        max_dist = max_dist + max((formated_dataset[column] == 0).sum(), (formated_dataset[column] == 1).sum())
    return max_dist


if __name__ == "__main__":
    modelA = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    modelB = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    up1 = 5
    up2 = 3
    print(f'Model1: {modelA}')
    print(f'Model2: {modelB}')
    print(f'Drastic distance: {drastic_distance(modelA, modelB)}')
    print(f'Hamming distance: {hamming_distance(modelA, modelB)}')
    print(f'Hamming distance bounded by {up1}: {distance(modelA, modelB, range_dist=up1)}')
    print(f'Hamming distance bounded by {up2}: {distance(modelA, modelB, range_dist=up2)}')
    print(f'Quadratic Hamming distance: {distance(modelA, modelB, factor=2)}')
    print(f'Quadratic distance bounded by {up1}: {distance(modelA, modelB, factor=2, range_dist=up1)}')
    print(f'Quadratic distance bounded by {up2}: {distance(modelA, modelB, factor=2, range_dist=up2)}')
    print(f'Cubic Hamming distance: {distance(modelA, modelB, factor=3)}')
    print(f'Cubic distance bounded by {up1}: {distance(modelA, modelB, factor=3, range_dist=up1)}')
    print(f'Cubic distance bounded by {up2}: {distance(modelA, modelB, factor=3, range_dist=up2)}')

    dataset_name = "rbm-small"
    initial_dataset = dataop.my_from_csv(f'{dataset_name}.csv')

    print('Some small dataset')
    print(initial_dataset)
    modelC = [0, 0, 0, 1, 0]
    modelD = [1, 1, 0, 1, 1]
    print(f'Hamming distance between {modelC} and the dataset: {distance_from_model_to_dataset(modelC, initial_dataset, hamming_distance)}')
    print(
        f'Hamming distance between {modelD} and the dataset: {distance_from_model_to_dataset(modelD, initial_dataset, hamming_distance)}')
    print(f'Min distance: {get_min_distance_to_dataset(initial_dataset)}')
    print(f'Max distance: {get_max_distance_to_dataset(initial_dataset)}')

