import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from openml.datasets import get_dataset
from sklearn.preprocessing import StandardScaler


def _extract_columns_from_data_by_name(data, feature_names, columns_to_include):
    """
    Extract certain columns from a numpy array representing a dataset.

    Args:
        data (np.array): [n x p] Two dimensional data with rows as datapoints and columns as features.
        feature_names (list of str): List of all the names of the columns in `data`.
        columns_to_include (list of str): List of the column names to include in the data. All values must be in
            `feature_names`.

    Raises:
        ValueError: If not all values in `columns_to_include` is in `feature_names`.

    Returns:
        np.array: Array of the filtered data
    """
    try:
        indices_to_include = [feature_names.index(column_name) for column_name in columns_to_include]
    except ValueError:
        raise ValueError(f"All values in `columns_to_include` must be in {feature_names}. Got {columns_to_include}. ")

    # Include all rows ([:,), but only the columns corresponding to the `columns_to_use`.
    filtered_data = data[:, indices_to_include]

    return filtered_data


def _split_data_in_train_val_test(x_data, y_data, val_ratio=0.2, test_ratio=0.2, seed=57):
    """
    Split data into train, validation and test data.

    Args:
        x_data (np.array): [n x p] Two dimensional array of the input data. Rows correspond to datapoints and columns
            corresponds to features.
        y_data (np.array): [n] Array of the targets corresponding to `x_data`.
        val_ratio (float, optional): Ratio of the data to include in the validation set. Defaults to 0.2.
        test_ratio (float, optional): Ratio of the data to include in the test set. Defaults to 0.2.
        seed (int, optional): Random seed. Defaults to 57.

    Raises:
        ValueError: If `val_ratio` and `test_ratio` is not between 0 and 1 and does not sum to less than 1.

    Returns:
        dict: Dictionary of the data splits.
    """
    if val_ratio < 0 or test_ratio < 0:
        raise ValueError(
            f"Arguments `val_ratio` and `test_ratio` must be between 0 and 1. Got {val_ratio=} and {test_ratio=}. "
        )
    if val_ratio + test_ratio >= 1:
        raise ValueError(
            f"Arguments `val_ratio` and `test_ratio` must be less than 1 summed. Got {val_ratio=} and {test_ratio=}. "
        )
    # Load random number generator (rng)
    rng = np.random.default_rng(seed=seed)
    n = x_data.shape[0]
    random_indices = rng.permutation(n)  # Get the numbers [0, 1, ..., n-1] in a random order

    train_ratio = 1 - val_ratio - test_ratio
    n_train = int(train_ratio * n)
    n_val = int(val_ratio * n)

    # Index the amount of random indices corresponding to the respective split size
    train_indices = random_indices[:n_train]
    val_indices = random_indices[n_train : n_train + n_val]
    test_indices = random_indices[n_train + n_val :]

    x_train = x_data[train_indices]
    x_val = x_data[val_indices]
    x_test = x_data[test_indices]

    y_train = y_data[train_indices]
    y_val = y_data[val_indices]
    y_test = y_data[test_indices]

    all_data = {
        "x_train": x_train,
        "x_val": x_val,
        "x_test": x_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
    }

    return all_data


def _scale_data_splits(data_dict):
    """
    Applies StandardScaler to x_train, x_val, and x_test in data_dict.
    The scaler is fitted only on x_train.

    Args:
        data_dict (dict): Dictionary with keys "x_train", "x_val" and "x_test".

    Returns:
        dict: Same dictionary with scaled arrays.
    """
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(data_dict["x_train"])
    x_val_scaled = scaler.transform(data_dict["x_val"])
    x_test_scaled = scaler.transform(data_dict["x_test"])

    data_dict["x_train"] = x_train_scaled
    data_dict["x_val"] = x_val_scaled
    data_dict["x_test"] = x_test_scaled

    return data_dict


def get_auto_mpg_data(columns_to_include=None, perform_scaling=True, val_ratio=0.2, test_ratio=0.2, seed=57):
    """
    Get the Auto MPG dataset as a dictionary of data splits.
    See https://archive.ics.uci.edu/dataset/9/auto+mpg for more info.

    The features include:
        ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin']

    Args:
        columns_to_include (list of str, optional): List of features to include in X. Default: all numeric columns.
        perform_scaling (bool): It `True`, scale data with standard-scaler.
        val_ratio (float, optional): Validation split ratio. Default: 0.2.
        test_ratio (float, optional): Test split ratio. Default: 0.2.
        seed (int, optional): Random seed. Default: 57.

    Returns:
        dict: Dictionary with train/val/test splits and metadata.
    """
    # Load and clean the data
    df = sns.load_dataset("mpg").dropna()

    # Define default features (numeric, exclude target)
    all_feature_names = ["cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year", "origin"]
    if columns_to_include is None:
        columns_to_include = all_feature_names

    x_data = df[columns_to_include].to_numpy()
    y_data = df["mpg"].to_numpy()

    filtered_x_data = _extract_columns_from_data_by_name(
        data=x_data, feature_names=columns_to_include, columns_to_include=columns_to_include
    )

    all_data = _split_data_in_train_val_test(
        x_data=filtered_x_data, y_data=y_data, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed
    )
    if perform_scaling:
        all_data = _scale_data_splits(data_dict=all_data)
    all_data["feature_names"] = columns_to_include
    all_data["target_name"] = "mpg"

    return all_data


def get_spambase_data(columns_to_include=None, perform_scaling=True, val_ratio=0.2, test_ratio=0.2, seed=57):
    """
    Get Spambase dataset.
    See https://archive.ics.uci.edu/dataset/94/spambasefor more info.

    Args:
        columns_to_include (list of str, optional): List of features to include in X. Default: all numeric columns.
        perform_scaling (bool): It `True`, scale data with standard-scaler.
        val_ratio (float, optional): Validation split ratio. Default: 0.2.
        test_ratio (float, optional): Test split ratio. Default: 0.2.
        seed (int, optional): Random seed. Default: 57.

    Returns:
        dict: Dictionary with train/val/test splits and metadata.
    """
    spam_metadata = get_dataset(44, version=1)
    x_data, y_data, _, all_feature_names = spam_metadata.get_data(target="class")

    if columns_to_include is None:
        columns_to_include = all_feature_names

    x_data = x_data[columns_to_include].to_numpy()
    y_data = y_data.astype(int).to_numpy()

    all_data = _split_data_in_train_val_test(
        x_data=x_data, y_data=y_data, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed
    )

    if perform_scaling:
        all_data = _scale_data_splits(data_dict=all_data)
    all_data["feature_names"] = columns_to_include
    all_data["target_name"] = "spam"

    return all_data
