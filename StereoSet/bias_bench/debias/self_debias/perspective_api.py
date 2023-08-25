import time
from typing import Dict, Optional, List

from googleapiclient import discovery
from googleapiclient.errors import HttpError


class PerspectiveApiScorer:
    """
    This class provides a method for accessing Perspective API using the Google API Client to obtain attribute scores for generated texts.
    """

    DEFAULT_ATTRIBUTES = [
        "toxicity",
        "severe_toxicity",
        "sexually_explicit",
        "threat",
        "profanity",
        "identity_attack",
    ]

    def __init__(self, api_key: str):
        """
        :param api_key: the API key to use. For details, see https://support.perspectiveapi.com/s/docs-get-started
        """
        self._service = discovery.build(
            "commentanalyzer", "v1alpha1", developerKey=api_key
        )

    def get_scores(
        self, input_text: str, requested_attributes: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Get attribute scores for a given text via Perspective API.
        :param input_text: the input text
        :param requested_attributes: the attributes for which to compute scores
        :return: a mapping from attribute names to scores
        """
        requested_attributes = (
            requested_attributes
            if requested_attributes
            else PerspectiveApiScorer.DEFAULT_ATTRIBUTES
        )

        analyze_request = {
            "comment": {"text": input_text},
            "requestedAttributes": {
                attribute.upper(): {} for attribute in requested_attributes
            },
            "spanAnnotations": False,
            "languages": ["en"],
        }

        response = None
        while not response:
            try:
                response = (
                    self._service.comments().analyze(body=analyze_request).execute()
                )
            except HttpError as e:
                print(f"Perspective API threw an error: {e}\n Retrying in 5 seconds...")
                time.sleep(5)

        return {
            attribute: response["attributeScores"][attribute.upper()]["summaryScore"][
                "value"
            ]
            for attribute in requested_attributes
        }
