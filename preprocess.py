import json
import os

DATA_DIR = 'data'
PROCESSED_DATA_DIR = 'processed_data'
MAX_COMMENT_LEN = 40


def replace_unicode(s):
    s = s.replace(u'\u2018', '\'')
    s = s.replace(u'\u2019', '\'')
    s = s.replace(u'\u2013', '-')
    s = s.replace(u'\u201c', '"')
    s = s.replace(u'\u201d', '"')
    s = s.replace(u'\u00a0', ' ')

    return s


def valid_text(s):
    # Remove submissions or comments containing links
    if s.find('http://') != -1 or s.find('https://') != -1:
        return False

    # Remove deleted submissions or comments
    if s in {'[removed]', '[deleted]'}:
        return False

    return True


def valid_submission(submission):
    if submission['score'] <= 0:
        return False

    if submission['edited']:
        return False

    if not valid_text(submission['title']):
        return False

    if not valid_text(submission['body']):
        return False

    return True


def valid_comment(comment):
    if comment['score'] <= 0:
        return False

    if not valid_text(comment['body']):
        return False

    if len(comment['body']) > MAX_COMMENT_LEN:
        return False

    return True


def process_dataset(subreddit):
    with open(os.path.join(DATA_DIR, subreddit + '.json'), 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    print('dataset contains', len(dataset), 'submissions before processing')

    # Remove invalid submissions
    dataset = [submission for submission in dataset if valid_submission(submission)]

    for submission in dataset:
        submission['comments'] = [comment for comment in submission['comments'] if valid_comment(comment)]

        if submission['comments']:
            score_threshold = max([comment['score'] for comment in submission['comments']]) / 4
            submission['comments'] = [comment for comment in submission['comments'] if comment['score'] >= score_threshold]

    print('dataset contains', len(dataset), 'submissions after processing')


def main():
    process_dataset('antijokes')
    return

    json_file_name = 'reddit_jokes.json'
    txt_file_name = 'reddit_jokes.txt'

    start_token = '<|startoftext|> '
    end_token = ' <|endoftext|> '
    title_end_token = ' <|endoftitle|> '

    with open(json_file_name, 'r') as f:
        dataset = json.load(f)

    dataset_file = open(txt_file_name, 'w', encoding='utf-8')

    for example in dataset:
        dataset_file.write(start_token + example['title'] + title_end_token + example['body'] + end_token + '\n')


if __name__ == '__main__':
    main()
