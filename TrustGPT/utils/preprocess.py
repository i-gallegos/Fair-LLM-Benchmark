import json
import pandas as pd


def get_txt_file(filename: str) -> list:
    """
    Read a text file and return its content as a list of strings.

    Args:
    filename: str, the name of the text file.

    Returns:
    data: list, the content of the text file as a list of strings.

    Raises:
    AssertionError: If filename is not a string.

    Example:
    get_txt_file("data.txt")
    """

    assert isinstance(filename, str), "filename must be a string."

    with open(filename, 'r') as f:
        data = f.readlines()
        data = [el.strip('\n') for el in data]

    return data


def get_csv_file(filename: str) -> list:
    """
    Read a CSV file and return the values in the 'res' column as a list.

    Args:
    filename: str, the name of the CSV file.

    Returns:
    all_data: list, the values in the 'res' column as a list.

    Raises:
    AssertionError: If filename is not a string or if the file extension is not 'csv'.
    Exception: If the 'res' column is not found in the CSV file.

    Example:
    get_csv_file("data.csv")
    """

    assert isinstance(filename, str), "filename must be a string."
    assert filename.split('.')[-1] == 'csv', "filename must have a '.csv' extension."

    data = pd.read_csv(filename)

    if 'res' not in data.columns:
        raise Exception("no res column!")

    all_data = data['res'].tolist()
    return all_data


def replace_double_underscore(string, replacements):
    result = string
    for replacement in replacements:
        result = result.replace("__", str(replacement), 1)
    return result


def read_configuration(filename: str) -> dict:
    """
    Read a configuration file and return its content as a dictionary.

    Args:
    filename: str, the name of the configuration file.

    Returns:
    config: dict, the content of the configuration file as a dictionary.

    Raises:
    AssertionError: If filename is not a string.

    Example:
    read_configuration("configuration.json")
    """

    assert isinstance(filename, str), "filename must be a string."

    with open(filename, 'r') as f:
        # open configuration json file
        with open("../config/configuration.json") as f:
            config = json.load(f)
    return config