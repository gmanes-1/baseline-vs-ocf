from constants import *
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix

from constants import OCF_DATA_COLUMN_NAME

from to_file import get_dataset_dir_path, get_dataset_file_path


# consistently write to csv with or without index (condition specified as parameter), whatever df contains or not an index
def my_to_csv(df, file_path, with_index=True):
    # Check if 'ID' column exists
    if INDEX_COLUMN_NAME in df.columns:
        if not with_index:
            # Remove 'ID' column if it exists and with_index is False
            df = df.drop(columns=['ID'])
    else:
        if with_index:
            # Add 'ID' column with row indices if it does not exist and with_index is True
            df = df.reset_index(drop=True)
            df.insert(0, 'ID', range(len(df)))

    # Save DataFrame to CSV
    df.to_csv(file_path, index=False)


def my_from_csv(file_name, remove_first_column=False):
    result = pd.read_csv(file_name)
    if remove_first_column:
        return remove_first_col(result)
    else:
        return result


def get_nb_rows(dataset):
    return dataset.shape[0]


def get_nb_cols(dataset):
    return dataset.shape[1]


def get_nb_features(dataset):
    return sum(dataset.columns.str.startswith('X_'))


def remove_first_col(dataset):
    return dataset.iloc[:, 1:]  # Selecting all rows and columns starting from index 1 onwards


def get_negated_negatives(dataset):
    # This function flips all X_i values for the rows for which y=0
    # It returns a new dataset with only X_i columns (thus, dropping the y column and any other column)
    result_dataset = dataset.copy()

    # Identify rows ending with '0'
    rows_ending_with_zero = result_dataset[result_dataset['y'] == 0]

    # Flip the bits in rows ending with '0' and update result_dataset
    # Get the columns starting with 'X_'
    columns_to_update = result_dataset.columns[result_dataset.columns.str.startswith('X_')]

    # Update columns starting with 'X_' based on the condition
    result_dataset.loc[rows_ending_with_zero.index, columns_to_update] = result_dataset.loc[
        rows_ending_with_zero.index, columns_to_update].map(lambda x: 1 if x == 0 else 0)

    # Get columns to drop that don't start with 'X_'
    columns_to_drop = [col for col in result_dataset.columns if not col.startswith('X_')]
    # Drop columns that don't start with 'X_'
    result_dataset.drop(columns=columns_to_drop, inplace=True, axis=1)

    return result_dataset


# convert a confusion matrix into four values TP, FP, TN, FN (in this order)
def confusion_matrix_into_four_values(y_true, y_pred):
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Extract TP, FP, TN, FN
    if cm.shape == (2, 2):
        TP = cm[1, 1]
        FP = cm[0, 1]
        TN = cm[0, 0]
        FN = cm[1, 0]
    elif cm.size == 1:
        if y_true[0] == y_pred[0] == 1:
            TP = cm[0, 0]
            FP = 0
            TN = 0
            FN = 0
        elif y_true[0] == y_pred[0] == 0:
            TP = 0
            FP = 0
            TN = cm[0, 0]
            FN = 0
    else:
        raise ValueError("Unexpected shape of confusion matrix")

    return [TP, FP, TN, FN]


# Display nicely a confusion matrix (listed as a list of four values [TP, FP, TN, FN] in this order) into a string
def pretty_confusion_matrix(four_values):
    TP = four_values[0]
    FP = four_values[1]
    TN = four_values[2]
    FN = four_values[3]

    # Set the width for each cell
    cell_width = 12

    # Create the formatted table
    table = f"+{'-' * cell_width}+{'-' * cell_width}+{'-' * cell_width}+\n"
    table += f"|{' ' * cell_width}|{' ' * ((cell_width - 4) // 2)}True{' ' * ((cell_width - 4) // 2 + (cell_width - 4) % 2)}|{' ' * ((cell_width - 5) // 2)}False{' ' * ((cell_width - 5) // 2 + (cell_width - 5) % 2)}|\n"
    table += f"+{'-' * cell_width}+{'-' * cell_width}+{'-' * cell_width}+\n"
    table += f"|{' ' * ((cell_width - 8) // 2)}Positive{' ' * ((cell_width - 8) // 2 + (cell_width - 8) % 2)}|{' ' * ((cell_width - len(str(TP))) // 2)}{TP}{' ' * ((cell_width - len(str(TP))) // 2 + (cell_width - len(str(TP))) % 2)}|{' ' * ((cell_width - len(str(FP))) // 2)}{FP}{' ' * ((cell_width - len(str(FP))) // 2 + (cell_width - len(str(FP))) % 2)}|\n"
    table += f"+{'-' * cell_width}+{'-' * cell_width}+{'-' * cell_width}+\n"
    table += f"|{' ' * ((cell_width - 8) // 2)}Negative{' ' * ((cell_width - 8) // 2 + (cell_width - 8) % 2)}|{' ' * ((cell_width - len(str(TN))) // 2)}{TN}{' ' * ((cell_width - len(str(TN))) // 2 + (cell_width - len(str(TN))) % 2)}|{' ' * ((cell_width - len(str(FN))) // 2)}{FN}{' ' * ((cell_width - len(str(FN))) // 2 + (cell_width - len(str(FN))) % 2)}|\n"
    table += f"+{'-' * cell_width}+{'-' * cell_width}+{'-' * cell_width}+\n"

    return table


if __name__ == "__main__":
    dataset_name = "rbm-small"
    df = my_from_csv(f'{dataset_name}.csv')

    print('Some small dataset')
    print(df)

    print('\nTarget dataset')
    print(get_negated_negatives(df))


def get_column_value_dataset_as_value_list(dataset, column_name):
    return dataset[column_name].tolist()


# 'size' is the number of rows of a dataset
# Note: before, 'size' was nb_rows x nb_cols, but it's really nb_rows that matters the most
def get_dataset_size(dataset_path):
    """
    Returns the size of the dataset at the given path.
    Size is defined as the number of rows
    """
    df = my_from_csv(dataset_path)
    #return df.shape[0] * df.shape[1]
    return df.shape[0]


def sorted_by_increasing_size(all_dataset_names):
    # Create a list of tuples (dataset_name, size)
    dataset_sizes = [(dataset_name, get_dataset_size(get_dataset_file_path(dataset_name))) for dataset_name in
                     all_dataset_names]

    # Sort the list of tuples by the size in descending order
    dataset_sizes.sort(key=lambda x: x[1])

    # Extract the sorted dataset names
    sorted_dataset_names = [name for name, size in dataset_sizes]

    #print(dataset_sizes)
    return sorted_dataset_names


# label should be a string equal to 'Training dataset' or 'Test dataset' or 'Dataset'
def get_description_dataset(dataset, label_dataset):
    assert label_dataset in ['Training dataset', 'Test dataset', 'Dataset']

    nb_rows = get_nb_rows(dataset)
    nb_positive = get_nb_rows(dataset[dataset['y'] == 1])
    nb_negative = get_nb_rows(dataset[dataset['y'] == 0])
    nb_features = get_nb_features(dataset)

    result = [f'{label_dataset} size: {nb_rows} ({nb_positive} pos / {nb_negative} neg)',
              f'{label_dataset} nb features: {nb_features}']

    # In the case of an OCF dataset, we want also to write the min and max OCF values
    if OCF_DATA_COLUMN_NAME in dataset.columns:
        min_ocf_value = dataset[OCF_DATA_COLUMN_NAME].min()
        max_ocf_value = dataset[OCF_DATA_COLUMN_NAME].max()

        result.append(f'OCF min/max values in training dataset: {min_ocf_value}/{max_ocf_value}')

    return result


def compute_roc_auc(y_true, y_scores):
    """
    Compute ROC AUC for a predictor that assigns positive integer values.

    Parameters:
    y_true : array-like of shape (n_samples,)
        True binary labels.
    y_scores : array-like of shape (n_samples,)
        Predicted integer scores.

    Returns:
    float
        ROC AUC value.
    list
        False Positive Rates for each threshold.
    list
        True Positive Rates for each threshold.
    """
    # Calculate FPR and TPR for each threshold
    # Invert scores for ROC calculation since lower scores indicate positive class
    fpr, tpr, _ = roc_curve(y_true, -y_scores)
    roc_auc = auc(fpr, tpr)
    # print(f'tpr/fpr/roc auc: {tpr}/{fpr}/{roc_auc}')

    return roc_auc, fpr, tpr


def plot_roc_curve(fpr, tpr, roc_auc, file_name):
    """
    Plot the ROC curve and save it as an image file.

    Parameters:
    fpr : array-like of shape (n_thresholds,)
        False positive rates.
    tpr : array-like of shape (n_thresholds,)
        True positive rates.
    roc_auc : float
        ROC AUC value.
    file_name : str
        File name for saving the plot.
    """
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(file_name)
    plt.close()
    # print(f"ROC curve saved as {file_name}")


# Save test results test_results into a dataset file file_name
# The test_results should be a dictionary [total_tp, total_fp, total_]
def savse_test_results_into_dataset(file_name, method_result):
    # Ensure that the row_data keys match the feature names
    assert set(method_result.keys()) == set(
        ALL_FEATURE_NAME_FOR_TEST_RESULTS), "The input row data keys do not match the feature names."

    # Check if the file exists
    if os.path.exists(file_name):
        # Read the existing dataset
        results_dataset = pd.read_csv(file_name)
    else:
        # Create an empty dataset with specified feature names
        results_dataset = pd.DataFrame(columns=ALL_FEATURE_NAME_FOR_TEST_RESULTS)

    # Check if the MODEL_NAME already exists in the dataset
    if method_result[MODEL_NAME] in results_dataset[MODEL_NAME].values:
        # Replace the existing row
        results_dataset = results_dataset[results_dataset[MODEL_NAME] != method_result[MODEL_NAME]]

    # Create a new DataFrame from the row data
    new_row_df = pd.DataFrame([method_result])
    # Ensure new_row_df has the same columns as df
    new_row_df = new_row_df.reindex(columns=ALL_FEATURE_NAME_FOR_TEST_RESULTS)
    results_dataset = pd.concat([results_dataset, new_row_df], ignore_index=True)

    # Save the updated dataset to the CSV file
    results_dataset.to_csv(file_name, index=False)


def save_test_results_into_dataset(file_name, method_result):
    # Ensure that the method_result keys match the feature names
    assert set(method_result.keys()) == set(
        ALL_FEATURE_NAME_FOR_TEST_RESULTS), "The input row data keys do not match the feature names."

    # Check if the file exists
    if os.path.exists(file_name):
        # Read the existing dataset
        results_dataset = pd.read_csv(file_name)
    else:
        # Create an empty dataset with specified feature names
        results_dataset = pd.DataFrame(columns=ALL_FEATURE_NAME_FOR_TEST_RESULTS)

    # Check if the MODEL_NAME already exists in the dataset
    if method_result[MODEL_NAME] in results_dataset[MODEL_NAME].values:
        # Replace the existing row
        results_dataset = results_dataset[results_dataset[MODEL_NAME] != method_result[MODEL_NAME]]

    # Create a new DataFrame from the method_result
    new_row_df = pd.DataFrame([method_result])

    # Ensure new_row_df has the same columns as results_dataset
    new_row_df = new_row_df.reindex(columns=ALL_FEATURE_NAME_FOR_TEST_RESULTS)

    # Drop all-NA columns from new_row_df
    new_row_df = new_row_df.dropna(axis=1, how='all')

    # Drop all-NA columns from results_dataset if it's empty
    if results_dataset.empty:
        results_dataset = pd.DataFrame(columns=new_row_df.columns)

    # Drop all-NA columns from results_dataset
    results_dataset = results_dataset.dropna(axis=1, how='all')

    # Concatenate the DataFrames
    results_dataset = pd.concat([results_dataset, new_row_df], ignore_index=True)

    # Round numerical values to 2 decimal places
    results_dataset = results_dataset.round(2)

    # Save the updated dataset to the CSV file
    my_to_csv(results_dataset, file_name, with_index=False)


# This procedure separates id, y (the ouput) and the remaining dataset
# return in order: id (one column), the rest of the dataset without y, and y (one column)
# This procedure does the inverse operation of concatenate_id_x_y
def separate_id_x_y(input_df):
    y_df = input_df[TARGET_DATA_COLUMN_NAME]
    id_df = input_df[INDEX_COLUMN_NAME]

    # Separate features from target and ID column
    x_df = input_df.drop(columns=[TARGET_DATA_COLUMN_NAME, INDEX_COLUMN_NAME])

    return id_df, x_df, y_df


# This procedure concatenate id, the remaining dataset without output, and y into a single dataset
# This procedure does the inverse operation of separate_id_x_y
def concatenate_id_x_y(id_df, x_df, y_df):
    return pd.concat([id_df, x_df, y_df], axis=1)
