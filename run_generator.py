import gpt_2_simple as gpt2
from datetime import datetime
import os

RUN_NAME = 'jokes1'
OUTPUT_DIR = 'outputs'
SUBREDDIT = 'jokes'


def generate_title_body(sess):
    for _ in range(20):
        title = gpt2.generate(sess,
                              length=50,
                              temperature=0.7,
                              prefix="<|startoftext|>",
                              truncate="<|endoftitle|>",
                              include_prefix=False,
                              nsamples=1,
                              batch_size=1,
                              return_as_list=True,
                              run_name=RUN_NAME,
                              checkpoint_dir='generator_models/'
                              )[0]
        body = gpt2.generate(sess,
                             length=50,
                             temperature=0.7,
                             prefix=title + ' <|endoftitle|>',
                             truncate="<|endoftext|>",
                             include_prefix=False,
                             nsamples=1,
                             batch_size=1,
                             return_as_list=True,
                             run_name=RUN_NAME,
                             checkpoint_dir='generator_models/'
                             )[0]

        print('Title:', title)
        print('Body:', body)


def generate_to_file(sess, n=1000):
    file = SUBREDDIT + '_gentext_{:%Y%m%d_%H%M%S}.txt'.format(datetime.utcnow())
    path = os.path.join(OUTPUT_DIR, file)

    print('Generating', n, 'submission to file', file)

    gpt2.generate_to_file(sess,
                          destination_path=path,
                          length=100,
                          temperature=0.8,
                          nsamples=10000,
                          batch_size=20,
                          checkpoint_dir='generator_models/',
                          prefix="<|startoftext|>",
                          truncate="<|endoftext|>",
                          include_prefix=True,
                          run_name=RUN_NAME)


def main():
    sess = gpt2.start_tf_sess()
    gpt2.load_gpt2(sess, run_name=RUN_NAME, checkpoint_dir='generator_models')

    generate_to_file(sess, 1000)


if __name__ == '__main__':
    main()
