import json
import operator
import os
import random
import re
import matplotlib.pyplot as plt
import sys

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

    # Remove non-ASCII characters
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

    fix_file(os.path.join(PROCESSED_DATA_DIR, subreddit + '.txt'))
    fix_file(os.path.join(PROCESSED_DATA_DIR, subreddit + '_title.txt'))
    fix_file(os.path.join(PROCESSED_DATA_DIR, subreddit + '_comment.txt'))


def gen_humor_detection_data(dataset, subreddit):
    data = []

    good_nr = 0
    bad_nr = 0
    removed_nr = 0

    for submission in dataset:
        if 15 < submission['score'] < 75:
            removed_nr += 1
            continue

        good = submission['score'] > 50

        if good:
            good_nr += 1
        else:
            bad_nr += 1

        text = submission['title'] + ' ' + TITLE_END_TOKEN + ' ' + submission['body']
        text = text.replace('\n', ' ')
        text = text.replace('\r', ' ')

        try:
            data.append(text + '<endoftext>' + str(int(good)) + '<endoftext>' + str(submission['score']))
        except:
            pass

    print('Positive exaxmples:', good_nr)
    print('Negative exaxmples:', bad_nr)
    print('Removed examples:', removed_nr)

    random.shuffle(data)

    test_idx = int(TEST_PCT * len(data))
    dev_idx = test_idx + int(DEV_PCT * len(data))

    test_data = data[:test_idx]
    dev_data = data[test_idx:dev_idx]
    train_data = data[dev_idx:]
    print('test_data:', len(test_data))
    print('dev_data:', len(dev_data))
    print('train_data:', len(train_data))

    with open(os.path.join(PROCESSED_DATA_DIR, subreddit + '_test.csv'), 'w') as f:
        f.write('text<endoftext>humor<endoftext>score\n')
        for example in test_data:
            try:
                f.write(example + '\n')
            except:
                pass
    fix_file(os.path.join(PROCESSED_DATA_DIR, subreddit + '_test.csv'))

    with open(os.path.join(PROCESSED_DATA_DIR, subreddit + '_dev.csv'), 'w') as f:
        f.write('text<endoftext>humor<endoftext>score\n')
        for example in dev_data:
            try:
                f.write(example + '\n')
            except:
                pass
    fix_file(os.path.join(PROCESSED_DATA_DIR, subreddit + '_dev.csv'))

    with open(os.path.join(PROCESSED_DATA_DIR, subreddit + '_train.csv'), 'w') as f:
        f.write('text<endoftext>humor<endoftext>score\n')
        for example in train_data:
            try:
                f.write(example + '\n')
            except:
                pass
    fix_file(os.path.join(PROCESSED_DATA_DIR, subreddit + '_train.csv'))

    with open(os.path.join(PROCESSED_DATA_DIR, subreddit + '.csv'), 'w') as f:
        f.write('text<endoftext>humor<endoftext>score\n')
        for example in data:
            try:
                f.write(example + '\n')
            except:
                pass
    fix_file(os.path.join(PROCESSED_DATA_DIR, subreddit + '.csv'))


def remove_reposts(dataset):
    submissions = []
    for submission in dataset:
        text = submission['title'] + submission['body']

        # Convert text to lowercase, remove non-letter characters
        text = re.sub(r'[^a-z]', '', text.lower())
        submissions.append((text, submission['score'], submission['id']))

    # Sort submissions by text and score
    submissions.sort(key=operator.itemgetter(0, 1))
    removed_ids = {}

    for i in range(1, len(submissions)):
        # If 2 submissions are identical remove the one with the lower score
        if submissions[i-1][0] == submissions[i][0] or submissions[i-1][0] == '':
            removed_ids[submissions[i-1][2]] = True

    print('Submissions:', len(submissions))
    print('Reposts removed:', len(removed_ids))

    return [example for example in dataset if example['id'] not in removed_ids]


def fix_file(path):
    with open(path, 'r') as f:
        text = f.read()
    text = text.replace('&#x200B;', '')
    with open(path, 'w') as f:
        f.write(text)


def plot_score_distribution(dataset):
    scores = [submission['score'] for submission in dataset]
    scores.sort()

    x = [0]
    y = [0]
    for score in scores:
        if score > 200:
            break
        if score != x[-1]:
            x.append(score)
            y.append(y[-1] + 1)
        else:
            y[-1] += 1

    y = [py / len(scores) for py in y]

    plt.plot(x, y)
    plt.plot(15, y[15], marker='o', color='red', markersize=10)
    plt.plot(75, y[75], marker='o', color='green', markersize=10)
    plt.xlabel('Scor')
    plt.ylabel('Procent postări')
    plt.show()


def process_dataset(subreddit):
    with open(os.path.join(DATA_DIR, subreddit + '.json'), 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    # Replace common unicode characters with their ascii counterpart
    for submission in dataset:
        submission['title'] = replace_unicode(submission['title'])
        submission['body'] = replace_unicode(submission['body'])

        for comment in submission['comments']:
            comment['body'] = replace_unicode(comment['body'])

    print('dataset contains', len(dataset), 'submissions before processing')

    # Remove invalid submissions
    dataset = [submission for submission in dataset if valid_submission(submission)]
    dataset = remove_reposts(dataset)

    # Uncomment to plot score distribution
    # plot_score_distribution(dataset)
    # return

    for submission in dataset:
        submission['comments'] = [comment for comment in submission['comments'] if valid_comment(comment)]

        if submission['comments']:
            score_threshold = max([comment['score'] for comment in submission['comments']]) / 4
            submission['comments'] = [comment for comment in submission['comments'] if comment['score'] >= score_threshold]

    # Generate input files for the generator
    gen_text_corpus(dataset, subreddit)
    # Generate input files for the classifier
    gen_humor_detection_data(dataset, subreddit)


def main():
    for i in range(1, len(sys.argv)):
        if sys.argv[i] in {'antijokes', 'jokes', 'dadjokes'}:
            subreddit = sys.argv[i]
            print('Processing', subreddit, 'data...')

            random.seed(42)
            process_dataset(subreddit)
        else:
            print('Unknown argument:', sys.argv[i])
            print('Expected one of:', 'antijokes', 'jokes', 'dadjokes')


if __name__ == '__main__':
    main()
