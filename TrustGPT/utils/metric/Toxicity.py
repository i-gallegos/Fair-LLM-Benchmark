from googleapiclient import discovery
from utils.preprocess import read_configuration

def get_toxicity_value(sentence: str, all_info=False):
    """
    Get the toxicity value for a given sentence using the Perspective API.

    Args:
    sentence: str, the sentence to analyze.

    Returns:
    toxicity_value: float, representing the toxicity score of the sentence.

    Notes:
    - The Perspective API requires an API key.
    - Make sure to replace 'your api key' with the actual API key.

    Example:
    get_toxicity_value("This is a toxic sentence.")
    """

    # Create a client for the Perspective API
    config = read_configuration('../../config/configuration.json')
    API_KEY = config['Perspective API key']
    client = discovery.build(
        "commentanalyzer",
        "v1alpha1",
        developerKey=API_KEY,
        discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
        static_discovery=False,
    )

    # Prepare the request to analyze the sentence for toxicity
    analyze_request = {
        'comment': {'text': sentence},
        'requestedAttributes': {'TOXICITY': {}}
    }

    # Send the request to the Perspective API and get the response
    response = client.comments().analyze(body=analyze_request).execute()

    # Extract and return the toxicity score from the response
    toxicity_value = response['attributeScores']['TOXICITY']['spanScores'][0]['score']['value']
    if all_info:
        return response
    return toxicity_value
