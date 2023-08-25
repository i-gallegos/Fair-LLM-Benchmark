import requests
import json
import re


def get_pushshift_submissions_interval(query, sub, after, before):
    """
    Retrieves Reddit submissions matching query term over a given interval
    Parameters
    ----------
    query : str
    Query term to match in the reddit submissions
    sub : str
    Submission match query
    after : int
    Start of Time interval
    before : int
    End of Time interval

    Returns
    -------
    Dictionary with submissions data
    """
    url = 'https://api.pushshift.io/reddit/search/submission/?q='+str(query)+'&size=1000&after='+str(after)+'&before='+str(before)+'&subreddit='+str(sub)
    print(url)
    r = requests.get(url)
    data = json.loads(r.text)
    return data['data']


def get_pushshift_comments(q1, q2, s, b, a):
    """
    Retrieves Reddit comments matching query terms over a given interval
    Parameters
    ----------
    q1 : str
    Query term to match in the reddit comments
    q2 : str
    Query term to match in the reddit comments
    s : int
    Number of reddit comments to return
    b : int
    End of time interval
    a : int
    Start of time interval

    Returns
    -------
    Dictionary of comments data
    """
    url = 'https://api.pushshift.io/reddit/search/comment/?q='+str(q1)+'+'+str(q2)+'&size='+str(s)+'&after='+str(a)+'&before='+str(b)
    print(url)
    r = requests.get(url)
    data = json.loads(r.text)
    return data['data']


def get_pushshift_submissions(query, sub):
    """
    Retrieves Reddit submissions matching query term over a given interval
    Parameters
    ----------
    query : str
    Query term to match in the reddit submissions
    sub : str
    Submission match query

    Returns
    -------
    Dictionary with submissions data
    """
    url = 'https://api.pushshift.io/reddit/search/submission/?q='+str(query)+'&size=1000'+'&subreddit='+str(sub)
    print(url)
    r = requests.get(url)
    data = json.loads(r.text)
    return data['data']


def process_reddit(comment):
    """
    Pre-processes a given comment by removing some characters
    Parameters
    ----------
    comment : str
    Given sentence

    Returns
    -------
    Processed sentence
    """
    comment = comment.encode("ascii", errors="ignore").decode()
    comment = re.sub('[^A-Za-z,. ]+', '', comment)
    return comment


def process_tweet(sent):
    """
    Pre-processes a given sentence
    Parameters
    ----------
    sent : str
    Given sentence

    Returns
    -------
    Processed sentence
    """

    sent = sent.encode("ascii", errors="ignore").decode()  # check this output
    # print(sent)
    sent = re.sub('@[^\s]+', '', sent)
    sent = re.sub('https: / /t.co /[^\s]+', '', sent)
    sent = re.sub('http: / /t.co /[^\s]+', '', sent)
    sent = re.sub('http[^\s]+', '', sent)

    sent = re.sub('&gt', '', sent)

    # split camel case combined words
    sent = re.sub('([A-Z][a-z]+)', r'\1', re.sub('([A-Z]+)', r' \1', sent))

    sent = sent.lower()

    # remove numbers
    sent = re.sub(' \d+', '', sent)
    # remove words with letter+number
    sent = re.sub('\w+\d+|\d+\w+', '', sent)

    # remove spaces
    sent = re.sub('[\s]+', ' ', sent)
    sent = re.sub('[^\w\s.!\-?]', '', sent)

    # remove 2 or more repeated char
    sent = re.sub(r"(.)\1{2,}", r"\1", sent)
    sent = re.sub(" rt ", "", sent)

    sent = re.sub('- ', '', sent)

    sent = sent.strip()
    # print(sent)
    return sent

