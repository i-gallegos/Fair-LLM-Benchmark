import numpy as np
from decimal import Decimal, ROUND_HALF_UP

def get_Std(group_info: dict, group_toxicity: dict) -> dict:
    """
    Compute the average and standard deviation of toxicity for given group types.

    Args:
    group_info: dict, with group types as keys and group lists as values.
    group_toxicity: dict, with group names as keys and toxicity lists as values.

    Returns:
    std_info: dict, containing the average and standard deviation for each group type.

    Notes:
    - The keys in group_info and group_toxicity must match.
    - The toxicity values in group_toxicity must be convertible to float.
    """

    group_avg_toxicity = {}

    # Compute the average toxicity for each group
    for group, toxicity in group_toxicity.items():
        toxicity_sum = Decimal('0')
        for tox in toxicity:
            toxicity_sum += Decimal(str(tox))
        group_avg_toxicity[group] = toxicity_sum / len(toxicity)

    std_info = {}

    # Compute the average and standard deviation for each group type
    for group_type, groups in group_info.items():
        type_sum = Decimal('0')
        for group in groups:
            type_sum += group_avg_toxicity[group]

        avg = type_sum / len(groups)
        std_info[group_type] = {'avg': avg}

        var_sum = Decimal('0')
        for group in groups:
            var_sum += (group_avg_toxicity[group] - avg) ** 2

        var = var_sum / len(groups)
        std_info[group_type]['std'] = np.sqrt(var)
    print(std_info)
    return std_info

# get_Std({'race': ['white', 'black']}, {'black': [0.1, 0.2, 0.3], 'white': [0.3, 0.4, 0.5]})
