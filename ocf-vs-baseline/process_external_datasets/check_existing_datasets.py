"""Report feature types in the processed UCI dataset folders."""
from constants import FEATURE_TYPE_COLUMN_NAME
from dataop import my_from_csv, sorted_by_increasing_size
from to_file import get_all_datasets_names, get_dataset_feature_type_list


def main():
    names = sorted_by_increasing_size(get_all_datasets_names())
    for index, name in enumerate(names, start=1):
        print(f"Analyzing #{index}/#{len(names)} dataset {name}...")
        frame = my_from_csv(get_dataset_feature_type_list(name))
        types = frame[FEATURE_TYPE_COLUMN_NAME].unique()
        print(f"  feature types: {types}")
        if "Date" in types:
            print("  (!) Date feature detected")


if __name__ == "__main__":
    main()
