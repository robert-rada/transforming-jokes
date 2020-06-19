import praw
from datetime import datetime
import json
import os
import time
from tqdm import tqdm


IDS_DIR = 'submission_ids'
DATA_DIR = 'data'
MAX_COMMENTS_PER_SUBMISSION = 10
MAX_RETRIES = 20


def gather_submissions_by_id(subreddit, save_every=1000, limit=None, start_id=None):
    reddit = praw.Reddit(client_id='pXJ-sYyjZ1iRCA',
                         client_secret='YRGI2E8KjGD-wNK2kq5cA96IaJM',
                         user_agent='python:praw (by /u/Kalydos)')

    # Load the ids of all submissions
    ids = []
    with open(os.path.join(IDS_DIR, subreddit + '.json')) as fout:
        for line in fout:
            id_json = json.loads(line)
            ids.append(id_json['id'])

    final_id = ids[-1]
    print(len(ids), 'total submissions')

    dataset_json = []

    # If a previous run did not finish processing all submissions we have to continue from
    # whe first unprocessed id.
    if start_id:
        ids = ids[ids.index(start_id) + 1:]

        dataset_file = open(os.path.join(DATA_DIR, subreddit + '.json'), 'r')
        dataset_json = json.load(dataset_file)
        dataset_file.close()

    if limit and len(ids) > limit:
        ids = ids[:limit]

    last_id = None
    for id in tqdm(ids):
        last_id = id

        # Save all data processed so far
        if len(dataset_json) % save_every == 0:
            with open(os.path.join(DATA_DIR, subreddit + '.json'), 'w') as f:
                json.dump(dataset_json, f, ensure_ascii=True, indent=4)

        # Fetch the submission based on the current id.
        # In case of error wait and retry for up to MAX_RETRIES times.
        submission = None
        for _ in range(MAX_RETRIES):
            try:
                submission = reddit.submission(id)
                submission._fetch()
                break
            except Exception as e:
                print('Could not fetch submission ' + str(id) + '.Retrying in 5 seconds...')
                print(e)
                time.sleep(5)

        if submission is None:
            print('Could not fetch submission ' + str(id))
            break

        # Make sure that submissions are self posts (ignore links) and are not stickied
        if not submission.is_self or submission.stickied:
            continue

        # Skip deleted submissions
        if submission.selftext in {'[removed]', '[deleted]'}:
            continue

        # Skip submissions with negative score
        if submission.score < 0:
            continue

        submission_json = {
            'title': submission.title,
            'body': submission.selftext,
            'author': submission.author.name if submission.author else 'None',
            'score': submission.score,
            'nsfw?': submission.over_18,
            'time': datetime.utcfromtimestamp(submission.created_utc).strftime('%Y-%m-%d %H:%M:%S'),
            'edited': submission.edited,
            'id': id,
            'comments': []
        }

        # Get some top level comments ordered by score
        submission.comment_sort = 'best'
        for i, comment in enumerate(submission.comments):
            if i >= MAX_COMMENTS_PER_SUBMISSION:
                break
            submission_json['comments'].append({'body': comment.body, 'score': comment.score})

        dataset_json.append(submission_json)

    with open(os.path.join(DATA_DIR, subreddit + '.json'), 'w') as f:
        json.dump(dataset_json, f, ensure_ascii=True, indent=4)

    print('Stopping... Dataset contains', len(dataset_json), 'submissions')
    if last_id == final_id:
        print('Finished gathering data')
    if last_id:
        print('Stopped at id', last_id)
        print('Rerun with start_id=\'', last_id, '\' to continue', sep='')


def main():
    gather_submissions_by_id('dadjokes', start_id='cyw82b')


if __name__ == '__main__':
    main()
