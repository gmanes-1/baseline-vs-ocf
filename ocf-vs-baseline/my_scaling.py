from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from constants import *
import pandas as pd
import numpy as np

from dataop import separate_id_x_y, concatenate_id_x_y


# This is the standard scaling with imputer mean prior to scaling
def impute_and_scale_sampling(df_train, df_test):
    # separate id, y, x
    id_df_train, x_df_train, y_df_train = separate_id_x_y(df_train)
    id_df_test, x_df_test, y_df_test = separate_id_x_y(df_test)

    # Identify numeric and categorical columns
    numeric_cols = x_df_train.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = x_df_train.select_dtypes(include=['object', 'category']).columns

    #print(f"Numeric columns in the training set: {numeric_cols}")
    #print(f"Categorical columns in the training set: {categorical_cols}")

    # Ensure categorical columns are of type string
    x_df_train[categorical_cols] = x_df_train[categorical_cols].astype(str)
    x_df_test[categorical_cols] = x_df_test[categorical_cols].astype(str)

    # If there are numeric columns, proceed with imputation and scaling
    if numeric_cols.any():
        # Imputation and scaling for numerical data (train set)
        x_df_train_numeric = x_df_train[numeric_cols]
        scaler = StandardScaler()
        imputer_numerical = SimpleImputer(strategy='mean')

        # Impute missing values and then scale on the training set
        x_df_train_numeric_imputed = imputer_numerical.fit_transform(x_df_train_numeric)
        x_df_train_numeric_imputed_scaled = scaler.fit_transform(x_df_train_numeric_imputed)

        # Apply the same transformations to the test set
        x_df_test_numeric = x_df_test[numeric_cols]
        x_df_test_numeric_imputed = imputer_numerical.transform(x_df_test_numeric)
        x_df_scaled_numeric_imputed_scaled = scaler.transform(x_df_test_numeric_imputed)

        # Reconstruct the dataset with imputed and scaled data
        x_df_train[numeric_cols] = x_df_train_numeric_imputed_scaled
        x_df_test[numeric_cols] = x_df_scaled_numeric_imputed_scaled
    else:
        print("No numeric columns to process.")

    # If there are categorical columns, proceed with imputation
    if categorical_cols.any():
        imputer_categorical = SimpleImputer(strategy='most_frequent')

        # Imputation for categorical data (train set)
        x_df_train_categorical = x_df_train[categorical_cols]
        x_df_train_categorical_imputed = imputer_categorical.fit_transform(x_df_train_categorical)

        # Apply the same transformations to the test set
        x_df_test_categorical = x_df_test[categorical_cols]
        x_df_test_categorical_imputed = imputer_categorical.transform(x_df_test_categorical)

        # Reconstruct the dataset with imputed categorical data
        x_df_train[categorical_cols] = x_df_train_categorical_imputed
        x_df_test[categorical_cols] = x_df_test_categorical_imputed

        # Concatenate with non-categorical features
        #x_df_train = pd.concat([x_df_train.drop(columns=categorical_cols), x_df_train_categorical_imputed], axis=1)
        #x_df_test = pd.concat([x_df_test.drop(columns=categorical_cols), x_df_test_categorical_imputed], axis=1)
    else:
        print("No categorical columns to process.")

    df_train_scaled_imputed = concatenate_id_x_y(id_df_train, x_df_train, y_df_train)
    df_test_scaled_imputed = concatenate_id_x_y(id_df_test, x_df_test, y_df_test)

    # Return the imputed and scaled training dataset and test dataset
    return df_train_scaled_imputed, df_test_scaled_imputed


# This is the standard one hot encoding
def one_hot_encode_sampling(df_train, df_test):
    # separate id, y, x
    id_df_train, x_df_train, y_df_train = separate_id_x_y(df_train)
    id_df_test, x_df_test, y_df_test = separate_id_x_y(df_test)

    # Identify categorical columns
    categorical_cols = x_df_train.select_dtypes(include=['object', 'category']).columns

    #print(f"Categorical columns in the training set: {categorical_cols}")

    # Ensure categorical columns are of type string
    x_df_train[categorical_cols] = x_df_train[categorical_cols].astype(str)
    x_df_test[categorical_cols] = x_df_test[categorical_cols].astype(str)

    # If there are categorical columns, proceed with encoding
    if categorical_cols.any():
        # One-hot encoding
        encoder = OneHotEncoder(drop='first', handle_unknown='ignore')
        encoder.fit(x_df_train[categorical_cols])  # Fit only on training data

        # Transform the datasets
        x_df_train_categorical_encoded = encoder.transform(x_df_train[categorical_cols]).toarray()
        x_df_test_categorical_encoded = encoder.transform(x_df_test[categorical_cols]).toarray()

        # Get consistent feature names
        encoded_columns = encoder.get_feature_names_out(categorical_cols)

        # Create DataFrames with aligned columns
        x_df_train_categorical_encoded_aligned = pd.DataFrame(x_df_train_categorical_encoded, columns=encoded_columns, index=x_df_train.index)
        x_df_test_categorical_encoded_aligned = pd.DataFrame(x_df_test_categorical_encoded, columns=encoded_columns, index=x_df_test.index)

        # Concatenate with non-categorical features
        x_df_train = pd.concat([x_df_train.drop(columns=categorical_cols), x_df_train_categorical_encoded_aligned], axis=1)
        x_df_test = pd.concat([x_df_test.drop(columns=categorical_cols), x_df_test_categorical_encoded_aligned], axis=1)
    else:
        print("No categorical columns to process.")

    df_train_encoded = concatenate_id_x_y(id_df_train, x_df_train, y_df_train)
    df_test_encoded = concatenate_id_x_y(id_df_test, x_df_test, y_df_test)

    # Return the encoded training dataset and test dataset
    return df_train_encoded, df_test_encoded


def convert_to_integer_scale(column):
    """Convert a continuous or integer column to an integer scale by multiplying by 10^N.
    Non-numerical or undefined values are mapped to None (an empty cell)."""

    # Function to check if a value is a valid number
    def is_valid_number(x):
        try:
            float(x)
            return True
        except (ValueError, TypeError):
            return False

    def process_value(x):
        if is_valid_number(x):
            value = float(x)
            if value.is_integer():
                return int(value)
            return value
        return None

    # Replace invalid values with NaN for processing
    cleaned_column = column.apply(process_value)
    # print(f'1: {cleaned_column}')

    # Round all valid numeric values to two decimal digits
    rounded_values = cleaned_column.round(3)
    # print(f'2: {rounded_values}')

    # Extract non-null values for calculation
    non_null_values = rounded_values.dropna()
    # print(f'3: {non_null_values}')

    # If there are no valid numbers, return a column of None
    if non_null_values.empty:
        return pd.Series([None] * len(column), dtype=pd.Int64Dtype())

    # Calculate the maximum number of decimal places
    max_decimal_places = max(len(str(val).split('.')[1]) if '.' in str(val) else 0 for val in non_null_values)
    # print(f'MAX DEC PLACES {max_decimal_places}')

    # Determine the scale factor
    scale_factor = 10 ** max_decimal_places

    # Multiply and convert to integer scale, maintaining NaN positions
    integer_scaled = rounded_values * scale_factor

    # Round to nearest integer and convert to Int64 type
    return integer_scaled.round().astype(pd.Int64Dtype())


def linear_scale_train(col):
    # Replace NaN and string values with the ***MEAN*** of the numerical values
    numeric_col = pd.to_numeric(col, errors='coerce')
    mean_value = numeric_col.mean()
    print(f'mean value: {mean_value}')
    #mean_value = numeric_col.mean() if not numeric_col.isna().all() else DEFAULT_NUMERIC_VALUE
    cleaned_col = numeric_col.fillna(mean_value)

    # Compute min and max values of the cleaned column
    min_val = cleaned_col.min()
    max_val = cleaned_col.max()

    # Calculate the scaling parameters
    shift = min_val
    coefficient = 1 if min_val == max_val else MAX_SCALING_VALUE / (max_val - min_val)

    # Apply the scaling transformation and convert to integers
    scaled_column = ((cleaned_col - shift) * coefficient).astype(int)

    return scaled_column, shift, coefficient, round(mean_value)


def linear_scale_test(col, shift, coefficient, min_value, max_value):
    # Replace NaN and string values with the mean of the numerical values
    numeric_col = pd.to_numeric(col, errors='coerce')
    mean_value = numeric_col.mean() if not numeric_col.isna().all() else ((max_value - min_value) / 2)
    #mean_value = numeric_col.mean()
    cleaned_col = numeric_col.fillna(mean_value)

    #print(f'CLEANED COLUMN {cleaned_col}')

    # Apply the scaling transformation and convert to integers
    #print(f'mean value: {mean_value}')
    #print(f'cleaned col:\n {cleaned_col}\n')

    scaled_column = ((cleaned_col - shift) * coefficient).astype(int)

    # Clip values to be within the range [min_value, max_value]
    scaled_column = np.clip(scaled_column, min_value, max_value)

    return scaled_column


# take a dataset and scale it (to be used by scaled OCF Hamming)
# this is typically used when the input dataset df is an original training dataset
# type_df should have two columns. Example:
# FEATURE_NAME,FEATURE_TYPE
# X_1,Integer
# X_2,Continuous
# X_3,Integer
# X_4,Binary
# X_5,Date
# X_6,Binary
# X_7,Categorical
# This function returns two datasets: the scaled dataset, and the dataframe storing the scaling parameters
def my_scaling_dataset_train(train_df, df_type):
    scaling_params = []
    # browse all features
    for _, row in df_type.iterrows():
        feature_name = row[FEATURE_NAME_COLUMN_NAME]
        feature_type = row[FEATURE_TYPE_COLUMN_NAME]
        # Separate features and target
        # X = df.filter(regex='^X_')  # Select columns with names starting with 'X_'
        # y = df['y']  # Select the column with the name 'y'

        # Scale the dataset features when necessary (i.e., when it is Integer or Continuous)
        if feature_type in ['Integer', 'Continuous']:
            scaled_column, shift, coefficient, mean_value = linear_scale_train(train_df[feature_name])
            train_df[feature_name] = scaled_column
            scaling_params.append((feature_name, feature_type, shift, coefficient, MIN_SCALING_VALUE, MAX_SCALING_VALUE, mean_value))
        else:
            # force the type of data for this column to be of 'string' type, and replace the 'nan' values by ''
            train_df[feature_name] = train_df[feature_name].astype(str)
            train_df[feature_name] = train_df[feature_name].replace('nan', pd.NA)
            missing_indices_train = train_df[feature_name].isna()
            # Replace missing values in the train set
            k = 1
            for idx in missing_indices_train[missing_indices_train].index:
                train_df.at[idx, feature_name] = UNKNOWN_CATEGORY_NAME_TRAINING + str(k)
                k += 1
            scaling_params.append((feature_name, feature_type, -1, -1, -1, -1, -1))
    # print(f'DF AFTER FILLING {train_df}')
    # Save the scaling parameters into a dataframe
    scaling_df = pd.DataFrame(scaling_params,
                              columns=[FEATURE_NAME_COLUMN_NAME, FEATURE_TYPE_COLUMN_NAME, SHIFT_COLUMN_NAME,
                                       COEFFICIENT_COLUMN_NAME, MIN_VALUE_COLUMN_NAME, MAX_VALUE_COLUMN_NAME, MEAN_VALUE_COLUMN_NAME])

    return train_df, scaling_df


def my_scaling_dataset_test(test_df, df_scaling_param):
    # Create a dictionary for quick lookup
    scaling_dict = df_scaling_param.set_index('FEATURE_NAME').to_dict(orient='index')

    # Rescale the values in the test set
    for feature in test_df.columns:
        if feature in scaling_dict:
            params = scaling_dict[feature]
            if params[FEATURE_TYPE_COLUMN_NAME] in ['Integer', 'Continuous']:
                shift = params[SHIFT_COLUMN_NAME]
                coefficient = params[COEFFICIENT_COLUMN_NAME]
                min_value = params[MIN_VALUE_COLUMN_NAME]
                max_value = params[MAX_VALUE_COLUMN_NAME]
                # Apply scaling to the feature
                test_df[feature] = linear_scale_test(test_df[feature], shift, coefficient, min_value, max_value)
            else:
                # For non-scaled features
                # force the type of data for this column to be of 'string' type, and replace the 'nan' values by ''
                test_df[feature] = test_df[feature].astype(str)
                test_df[feature] = test_df[feature].replace('nan', pd.NA)
                missing_indices_train = test_df[feature].isna()
                # Replace missing values in the train set
                k = 1
                for idx in missing_indices_train[missing_indices_train].index:
                    test_df.at[idx, feature] = UNKNOWN_CATEGORY_NAME_TEST + str(k)
                    k += 1

    return test_df
