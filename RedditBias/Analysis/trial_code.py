import pandas as pd
import re
import requests
import json
from collections import defaultdict


def process_reddit(comment):
    comment = comment.encode("ascii", errors="ignore").decode()
    comment = re.sub('[^A-Za-z,. ]+', '', comment)
    return comment


def get_pushshift_data(query, after=None, before=None, sub=None):
    url = 'https://api.pushshift.io/reddit/search/submission/?title='+str(query)+'&size=1000'+'&subreddit=food'
    print(url)
    r = requests.get(url)
    data = json.loads(r.text)
    return data

# data_path = '/Users/soumya/Documents/Mannheim-Data-Science/Sem_4/MasterThesis/Data/'
# data_df = pd.read_csv(data_path + 'reddit_comments_religion_muslims.csv')
#
# data_df['comments_processed'] = data_df['comments'].apply(lambda x: process_reddit(x))
#
# data_df.to_csv(data_path + 'reddit_comments_religion_muslims_processed.csv')


# dat = get_pushshift_data('africans')
# print(dat)


def get_pushshift_submissions(query, sub, after=None, before=None):
    url = 'https://api.pushshift.io/reddit/search/submission/?q='+str(query)+'&size=1000'+'&subreddit='+str(sub)
    print(url)
    r = requests.get(url)
    print(r)
    data = json.loads(r.text)
    return data['data']


sub_data = get_pushshift_submissions("black people", 'blacklivesmatter')
print(sub_data)
print('len of sub data: {}'.format(len(sub_data)))
# &after='+str(after)+'&before='+str(before)
first_level = defaultdict(list)
thread = defaultdict(list)

for sub in sub_data:
    print(sub['id'])
    url = 'https://api.pushshift.io/reddit/comment/search/?link_id=' + sub['id'] + '&limit=1000'
    r = requests.get(url)
    subm_comment = json.loads(r.text)
    comments = subm_comment['data']
    # print(comments['id'])
    for c in comments:
        if c['parent_id'] == c['link_id']:
            first_level[c['parent_id']].append(c['body'])
        elif c['parent_id'].split('_')[1] in comments['id']:
            print('child comment')
            print(c['id'])
            thread[['parent_id']].append(c['body'])


print(thread)
print(len(thread))

'''
comment_stack = submission.comments[:]
print(comment_stack)
submission.comments.replace_more(limit=None)
comment_tree_dict = defaultdict(list)
while comment_stack:
    print(comment_stack)
    comment = comment_stack.pop(0)
    # print(comment.body.encode("utf-8"))
    print(comment.link_id), print(comment.parent_id)
    if comment.link_id in comment.parent_id:
        comment_tree_dict[comment.id] = [(comment.id, comment.permalink)]
    for k, val in comment_tree_dict.items():
        if comment.parent_id.split('_')[1] in val:
            comment_tree_dict[k].append(comment.id)

    comment_stack[0:0] = comment.replies

print(comment_stack)
print(comment_tree_dict)
'''