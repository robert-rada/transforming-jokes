import math
import re
from collections import Counter
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool

WORD = re.compile(r"\w+")
NPROC = 6


def get_cosine(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x] ** 2 for x in list(vec1.keys())])
    sum2 = sum([vec2[x] ** 2 for x in list(vec2.keys())])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator


def text_to_vector(text):
    words = WORD.findall(text)
    return Counter(words)


def similarity(vector1, vector2):
    return get_cosine(vector1, vector2)


def is_repost(args):
    row, corpus_vector, tolerance = args
    query_text = row
    query_vec = text_to_vector(query_text)

    for vec in corpus_vector:
        if similarity(query_vec, vec) > tolerance:
            return True
    return False


def remove_reposts(df, subreddit, tolerance=0.9):
    df_corpus = pd.read_csv('processed_data/' + subreddit + '.csv', sep='<endoftext>', engine='python')
    corpus = [data[0] for data in df_corpus.values.tolist()]
    corpus_vector = [text_to_vector(text) for text in corpus]

    with Pool(6) as p:
        args = [(row[0], corpus_vector, tolerance) for row in df.values.tolist()]
        r = list(tqdm(p.imap(is_repost, args), total=len(df)))
        r = pd.Series(r, dtype='bool')

    return df[~r]


if __name__ == '__main__':
    df = pd.DataFrame(["You know it's Monday when... Sunday was the day before.",
                       'Why was Helen Keller a bad driver? She was deaf and blind...',
                       'Why did the boy cross the road? Because I was following him.',
                       'This is not a joke. Just a test!'])

    df = remove_reposts(df, 'antijokes')
    print(df.head(5))
