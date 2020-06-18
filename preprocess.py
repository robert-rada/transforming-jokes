import json


def replace_unicode(s):
    s = s.replace(u'\u2018', '\'')
    s = s.replace(u'\u2019', '\'')
    s = s.replace(u'\u2013', '-')
    s = s.replace(u'\u201c', '"')
    s = s.replace(u'\u201d', '"')

    return s


def main():
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
