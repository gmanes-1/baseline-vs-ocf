"""Retrieve and format the selected binary UCI datasets (original project script)."""
import os
import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo, list_available_datasets

from constants import *
from dataop import my_to_csv


def map_target_values(value1, value2):
    """Map the two target values to {0, 1} using the original project rules."""
    try:
        first, second = float(value1), float(value2)
        if {first, second} == {0.0, 1.0}:
            return (0, 1) if first < 0.5 else (1, 0)
    except (TypeError, ValueError):
        pass

    first, second = str(value1).lower(), str(value2).lower()
    if {first, second} == {"0", "1"}:
        return value1, value2
    if {first, second} == {"-", "+"}:
        return (0, 1) if second == "+" else (1, 0)
    if {first, second} == {"yes", "no"}:
        return (1, 0) if first == "yes" else (0, 1)
    if first.startswith("no") or second.startswith("no"):
        assert not first.startswith("no") or not second.startswith("no")
        return (0, 1) if first.startswith("no") else (1, 0)
    if {first, second} == {"positive", "negative"}:
        return (1, 0) if first == "positive" else (0, 1)
    if {first, second} == {"t", "f"}:
        return (1, 0) if first == "t" else (0, 1)
    if {first, second} == {"true", "false"}:
        return (1, 0) if first == "true" else (0, 1)
    if {first, second} == {"n", "o"}:
        return (1, 0) if first == "o" else (0, 1)
    return 0, 1


def main():
    list_available_datasets()
    root = os.environ[DATASET_ROOT_PATH_ENV_VAR_NAME] + "/"
    os.makedirs(root, exist_ok=True)

    # Original selected UCI dataset IDs (58 candidates).
    dataset_ids = [
        13, 14, 15, 16, 17, 22, 27, 28, 43, 46, 47, 52, 70, 73, 74, 75,
        88, 94, 95, 96, 101, 105, 143, 144, 145, 151, 161, 172, 174, 176,
        225, 244, 267, 277, 300, 327, 329, 380, 419, 426, 451, 468, 519,
        529, 537, 545, 563, 565, 572, 603, 728, 732, 759, 850, 857, 887,
        890, 915,
    ]
    excluded = {32, 117, 222, 357, 367, 560, 891, 967}
    feature_types_seen, target_values_seen = set(), set()

    for dataset_id in dataset_ids:
        if dataset_id in excluded:
            continue
        try:
            uci = fetch_ucirepo(id=dataset_id)
            features = uci.data.features.copy()
            if not isinstance(features, pd.DataFrame):
                features = pd.DataFrame(features)

            missing_symbol = uci.metadata.missing_values_symbol
            if missing_symbol:
                features.replace(missing_symbol, np.nan, inplace=True)

            target_columns = uci.metadata.target_col
            if not isinstance(target_columns, list) or len(target_columns) != 1:
                print(f"Dataset {dataset_id}: expected exactly one target column; skipped.")
                continue
            target_column = target_columns[0]
            target = pd.DataFrame(uci.data.targets.copy(), columns=[target_column])
            if missing_symbol is not None:
                target.replace(missing_symbol, np.nan, inplace=True)
            else:
                target.replace([None], np.nan, inplace=True)

            combined = pd.concat([features, target], axis=1)
            unique_targets = target[target_column].nunique()
            if unique_targets != 2:
                print(f"Dataset {dataset_id}: target is not binary; skipped.")
                continue

            name = uci.metadata.name
            rows, n_features = combined.shape[0], combined.shape[1] - 1
            dataset_name = f"{dataset_id}-{name}_{rows}r_{n_features}f"
            folder = os.path.join(root, dataset_name)
            os.makedirs(folder, exist_ok=True)
            base = os.path.join(folder, dataset_name)
            my_to_csv(combined, base + DATASET_ORIGINAL + ".csv", with_index=True)

            feature_names, feature_types, mapping = [], [], {}
            for variable_name, role in uci.variables.role.items():
                if role == "Feature":
                    feature_names.append(f"X_{len(feature_names) + 1}")
                    feature_types.append(uci.variables.type.get(variable_name, "Unknown"))
                    mapping[variable_name] = f"X_{len(feature_names)}"
            combined.rename(columns=mapping, inplace=True)

            if len(feature_names) == len(feature_types):
                type_frame = pd.DataFrame({
                    FEATURE_NAME_COLUMN_NAME: feature_names,
                    FEATURE_TYPE_COLUMN_NAME: feature_types,
                })
                my_to_csv(type_frame, base + DATASET_TYPE + ".csv", with_index=False)
            else:
                print(f"Dataset {dataset_id}: feature/type count mismatch.")

            formatted = combined.copy()
            formatted.rename(columns={target_column: "y"}, inplace=True)
            formatted.columns = feature_names + ["y"]
            values = formatted["y"].unique()
            if len(values) != 2:
                print(f"Dataset {dataset_id}: target changed during formatting; skipped.")
                continue
            mapped = map_target_values(values[0], values[1])
            formatted["y"] = formatted["y"].replace({values[0]: mapped[0], values[1]: mapped[1]})
            my_to_csv(formatted, base + ".csv", with_index=True)

            feature_types_seen.update(feature_types)
            target_values_seen.update(values)
            print(
                f"Dataset ID {dataset_id} ({name}) is available for binary "
                f"classification; target={target_column}, rows={rows}, features={n_features}"
            )
        except Exception as error:
            print(f"Dataset ID {dataset_id} is not available: {error}")

    print("Encountered feature types:", feature_types_seen)
    print("Encountered target values:", target_values_seen)


if __name__ == "__main__":
    main()
