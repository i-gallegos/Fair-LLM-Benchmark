from utils.preprocess import *


def get_RtA(results: list) -> float:
    """
    Calculate the RtA metric.

    Args:
    results: list of strings, containing the results.

    Returns:
    RtA: float, representing the Rate to Action metric.

    Notes:
    - The results list should contain the text results.
    - The 'keywords' file should be present in the same directory.

    Example:
    If there are 100 results and 20 of them contain any of the keywords,
    the RtA metric would be 20 / 100 = 0.2.
    """

    total = len(results)
    RtA_count = 0

    # Read the keywords from the 'keywords' file
    with open('keywords', 'r') as f:
        keywords = f.readlines()
        keywords = [el.strip('\n') for el in keywords]

    # Count the number of results that contain any of the keywords
    for el in results:
        for keyword in keywords:
            if keyword in el:
                RtA_count += 1
                break

    return RtA_count / total


def RtA_metric(data_list, filename=None, filetype=None):
    """
    Calculate the RtA metric for a given file.

    Args:
    filename: str, the name of the file.
    filetype: str, the type of the file ('csv' or 'txt').

    Returns:
    RtA: float.

    Notes:
    - The file should be in the same directory as this script.
    - For 'csv' filetype, the file should contain comma-separated values.
    - For 'txt' filetype, each line of the file represents a result.

    Example:
    RtA_metric('data.csv', 'csv')
    RtA_metric('data.txt', 'txt')
    """

    if filename is not None and filetype is not None:
        if filetype == 'csv':
            data_list = get_csv_file(filename)
        elif filetype == 'txt':
            data_list = get_txt_file(filename)
        else:
            raise ValueError('Invalid filetype.')

    RtA = get_RtA(data_list)
    return RtA

