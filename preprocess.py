import json
import os
import random

DATA_DIR = 'data'
PROCESSED_DATA_DIR = 'processed_data'
MAX_COMMENT_LEN = 50
MAX_SUBMISSION_LEN = 300

START_TOKEN = '<|startoftext|>'
END_TOKEN = '<|endoftext|>'
TITLE_END_TOKEN = '<|endoftitle|>'
COMMENT_TOKEN = '<|startofcomment|>'

DEV_PCT = 0.1
TEST_PCT = 0.1


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

    if not all(ord(c) < 128 for c in s):
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

    if len(submission['body']) > MAX_SUBMISSION_LEN:
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


def gen_text_corpus(dataset, subreddit):
    file_all = open(os.path.join(PROCESSED_DATA_DIR, subreddit + '.txt'), 'w')
    file_title = open(os.path.join(PROCESSED_DATA_DIR, subreddit + '_title.txt'), 'w')
    file_comment = open(os.path.join(PROCESSED_DATA_DIR, subreddit + '_comment.txt'), 'w')

    counter = 0
    comm_counter = 0
    for example in dataset:
        try:
            file_title.write(START_TOKEN + ' ' + example['title'] + ' ' + TITLE_END_TOKEN + '\n')
            file_all.write(START_TOKEN + ' ' + example['title'] + ' ' + TITLE_END_TOKEN + ' '
                           + example['body'] + ' ' + END_TOKEN + '\n')
            counter += 1

            for comment in example['comments']:
                file_comment.write(START_TOKEN + ' ' + example['title'] + ' ' + TITLE_END_TOKEN + ' '
                                   + example['body'] + ' ' + COMMENT_TOKEN + ' '
                                   + comment['body'] + ' ' + END_TOKEN + '\n')
                comm_counter += 1
        except Exception as e:
            # print(e)
            pass

    print('dataset contains', counter, 'submissions after processing')
    print('dataset contains', comm_counter, 'comments after processing')

    file_all.close()
    file_title.close()
    file_comment.close()


def gen_humor_detection_data(dataset, subreddit):
    data = []

    for submission in dataset:
        good = submission['score'] > 50
        text = submission['title'] + ' ' + submission['body']
        text = text.replace('\n', ' ')
        text = text.replace('\r', ' ')

        try:
            data.append(text + '<endoftext>' + str(good))
        except:
            pass

    random.shuffle(data)

    test_idx = int(TEST_PCT * len(data))
    dev_idx = test_idx + int(DEV_PCT * len(data))

    test_data = data[:test_idx]
    dev_data = data[test_idx:dev_idx]
    train_data = data[dev_idx:]

    with open(os.path.join(PROCESSED_DATA_DIR, subreddit + '_test.csv'), 'w') as f:
        f.write('text<endoftext>humor\n')
        for example in test_data:
            try:
                f.write(example + '\n')
            except:
                pass

    with open(os.path.join(PROCESSED_DATA_DIR, subreddit + '_dev.csv'), 'w') as f:
        f.write('text<endoftext>humor\n')
        for example in dev_data:
            try:
                f.write(example + '\n')
            except:
                pass

    with open(os.path.join(PROCESSED_DATA_DIR, subreddit + '_train.csv'), 'w') as f:
        f.write('text<endoftext>humor\n')
        for example in train_data:
            try:
                f.write(example + '\n')
            except:
                pass

    with open(os.path.join(PROCESSED_DATA_DIR, subreddit + '.csv'), 'w') as f:
        f.write('text<endoftext>humor\n')
        for example in data:
            try:
                f.write(example + '\n')
            except:
                pass


def process_dataset(subreddit):
    with open(os.path.join(DATA_DIR, subreddit + '.json'), 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    for submission in dataset:
        submission['title'] = replace_unicode(submission['title'])
        submission['body'] = replace_unicode(submission['body'])

        for comment in submission['comments']:
            comment['body'] = replace_unicode(comment['body'])

    print('dataset contains', len(dataset), 'submissions before processing')

    # Remove invalid submissions
    dataset = [submission for submission in dataset if valid_submission(submission)]

    for submission in dataset:
        submission['comments'] = [comment for comment in submission['comments'] if valid_comment(comment)]

        if submission['comments']:
            score_threshold = max([comment['score'] for comment in submission['comments']]) / 4
            submission['comments'] = [comment for comment in submission['comments'] if comment['score'] >= score_threshold]

    gen_text_corpus(dataset, subreddit)
    gen_humor_detection_data(dataset, subreddit)


def main():
    random.seed(42)
    process_dataset('jokes')


if __name__ == '__main__':
    main()
