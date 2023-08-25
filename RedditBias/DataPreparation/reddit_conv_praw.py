"""
This script extracts Reddit conversation like comment threads using PRAW API
"""
import time
import praw
import json
import pandas as pd
from praw.models import MoreComments
from collections import defaultdict
from anytree import Node, RenderTree
from utils import reddit_helpers as rh
# loveleo2662 - reddit pass


def get_comment_thread_id(submission):
    """
    Creates reddit conversation threads from the submission
    Parameters
    ----------
    submission : object
    Submission object from Reddit

    Returns
    -------
    List of Conversation like comment IDs from Reddit submissions
    """
    comment_stack = submission.comments[:]
    # print(comment_stack)
    submission.comments.replace_more(limit=None)
    comment_thread = []
    temp = Node(None)

    while comment_stack:
        # print(comment_stack)
        comment = comment_stack.pop(0)
        # print(comment.body.encode("utf-8"))
        # print(comment.link_id), print(comment.parent_id)
        if comment.link_id in comment.parent_id:
            c = Node(comment.id)
            # print(c)
            temp = c
            comment_thread.append(temp.__str__().split('\'')[1])
        else:
            if temp is not None:
                if comment.parent_id.split('_')[1] in temp.name:
                    c = Node(comment.id, parent=temp)
                    # print(c)
                else:
                    c = Node(comment.id, parent=Node(comment.parent_id.split('_')[1]))
                    # print(type(c))
                    # print(c.__str__().split('\'')[1])
                temp = c
                comment_thread.append(temp.__str__().split('\'')[1])

        comment_stack[0:0] = comment.replies
    return comment_thread


def get_comment_thread_sent(comment_thread, subm):
    """
    Extracts comment body for given comment IDs thread
    Parameters
    ----------
    comment_thread : list
    List of conversation like threads with comment IDs
    subm : object
    Reddit submission object

    Returns
    -------
    List of comment threads with Reddit comments
    List of comment threads with Reddit comment IDs

    """
    list_comment_threads = []
    list_comment_ids = []
    for c in comment_thread:
        l_sent = []
        l_id = []
        for co in c.split('/')[1:]:
            # print(co)
            for comm in subm.comments.list():
                if co == comm.id:
                    l_sent.append(comm.body)
                    l_id.append(comm.id)
                    break
        # print(lop)
        # print('#' * 10)
        list_comment_threads.append(l_sent)
        list_comment_ids.append(l_id)
    return list_comment_threads, list_comment_ids


def get_comment_thread_timeinterval(t_loops, reddit, query, subreddit):
    """
    Retrieves Reddit submissions based on query and subreddit tag over a time period and extracts conversation like
    threads from each comment
    Parameters
    ----------
    t_loops : int
    Specifies number of times to loop
    reddit : object
    Reddit PRAW object
    query : str
    Demographic target group term/phrase
    subreddit : str
    Subreddit name / tag

    Updates global variable: comment_id_threads_dict_new with conversation threads over all retrieved submissions

    """

    global comment_id_threads_dict, comment_id_threads_dict_new

    before = int(time.time())
    all_comment_threads = []
    all_comment_ids = []

    for t in range(t_loops):
        after = before - (60 * 60 * 24 * 30)
        try:
            sub_data = rh.get_pushshift_submissions_interval(query, subreddit, after, before)
            # print(sub_data)
            print('len of sub data: {}'.format(len(sub_data)))

            for sub in sub_data:
                # print(sub['id'])
                if sub['num_comments'] > 0:
                    # print(sub['num_comments'])
                    # print('Has comments')
                    # print(sub['full_link'])
                    submission = reddit.submission(id=sub['id'])

                    comment_threads_id = get_comment_thread_id(submission)
                    comment_threads_sent, comment_threads_id_list = get_comment_thread_sent(comment_threads_id,
                                                                                            submission)
                    print(comment_threads_id)
                    print(comment_threads_id_list)
                    print(comment_threads_sent)

                    if len(comment_threads_sent) != 0:
                        # all_comment_threads.extend(comment_threads_sent)
                        # all_comment_ids.extend(comment_threads_id_list)
                        comment_id_threads_dict[(sub['id'], sub['full_link'], sub['title'], sub['selftext'])] = comment_threads_sent
                        if 'subm_id' in comment_id_threads_dict_new.keys():
                            comment_id_threads_dict_new['subm_id'].append(sub['id'])
                            comment_id_threads_dict_new['full_link'].append(sub['full_link'])
                            comment_id_threads_dict_new['title'].append(sub['title'])
                            comment_id_threads_dict_new['selftext'].append(sub['selftext'])
                            comment_id_threads_dict_new['comment_threads'].append(comment_threads_sent)

                        else:
                            comment_id_threads_dict_new['subm_id'] = [sub['id']]
                            comment_id_threads_dict_new['full_link'] = [sub['full_link']]
                            comment_id_threads_dict_new['title'] = [sub['title']]
                            comment_id_threads_dict_new['selftext'] = [sub['selftext']]
                            comment_id_threads_dict_new['comment_threads'] = comment_threads_sent

        except Exception as e:
            print('Exception occured: {}'.format(repr(e)))
        before = after

    # return all_comment_threads, all_comment_ids


if __name__ == '__main__':

    start = time.time()
    reddit_praw = praw.Reddit(client_id="PgQ-P0z6qp6BwQ",
                         client_secret="PX2hN4lyFQAAbYJpGS132GGWWxw",
                         user_agent="macbook:com.example.scraperedditcomments:v1.0.0 (by u/soumybarikeri)")
    data_path = '/Users/soumya/Documents/Mannheim-Data-Science/Sem_4/MasterThesis/Data/'
    demo = 'race'  # 'religion2' # 'race' #'gender' # 'religion'
    demo_1 = 'black'  # 'muslims' # 'jews' # 'black' #'female' # 'jews'
    time_loops = 4

    # with open(data_path + demo + '/' + demo + '_opposites.txt') as f:
    #     query_demo = [line.split(',')[0] for line in f]

    subreddits = ['blacklivesmatter', 'blackpeopletwitter', 'blackfellas', 'blackladies', 'justiceforblackpeople',
                  'africanamerican', 'africanamericannews', 'blackwomen', 'blackculture', 'blackpeoplegifs',
                  'blackindependence', 'antiracist', 'antiracistaction']

    query_demo = ["black people", "african*", "African-american*", "Afro-american*", "Negro*", "Black women",
                  "Black men", "blacks*", "Black person", "Black boys", "Black girls", "Black population",
                  "dark-skinned people"]

    # all_comment_thr = []
    # all_comment_id = []

    loops = int(len(query_demo)/4) if len(query_demo) % 4 == 0 else int(len(query_demo)/4) + 1
    print('Looping {} times'.format(loops))

    for i in range(loops):
        comment_id_threads_dict = defaultdict(list)
        comment_id_threads_dict_new = {}
        query_demo_4 = query_demo[i*4:i*4+4]

        for q in query_demo_4[:2]:
            for s in subreddits[:2]:
                get_comment_thread_timeinterval(time_loops, reddit_praw, q, s)
                # all_comment_thr.extend(comment_threads)
                # all_comment_id.extend(comment_ids)

        # print(len(all_comment_thr))
        # print(all_comment_id)

        # df_comment_threads = pd.DataFrame(list(comment_id_threads_dict.items()), columns=['submission_info', 'comment_threads'])
        # df_comment_threads.to_csv(data_path + demo + '/' + 'reddit_threads_' + demo + '_' + demo_1 + '_raw' + str(i) + '.csv')

        with open(data_path + demo + '/' + 'reddit_threads_' + demo + '_' + demo_1 + '_raw' + str(i) + '.json', 'w') as fp:
            json.dump(comment_id_threads_dict_new, fp)

    end = time.time()
    print('Time for code execution: {}'.format((end-start)/60))