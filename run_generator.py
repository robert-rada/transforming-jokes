import os
import gpt_2_simple as gpt2
from datetime import datetime
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

OUTPUT_DIR = 'outputs'


# def generate_title_body(sess, run_name):
#     for _ in range(20):
#         title = gpt2.generate(sess,
#                               length=50,
#                               temperature=0.7,
#                               prefix="<|startoftext|>",
#                               truncate="<|endoftitle|>",
#                               include_prefix=False,
#                               nsamples=1,
#                               batch_size=1,
#                               return_as_list=True,
#                               run_name=RUN_NAME,
#                               checkpoint_dir='generator_models/'
#                               )[0]
#         body = gpt2.generate(sess,
#                              length=50,
#                              temperature=0.7,
#                              prefix=title + ' <|endoftitle|>',
#                              truncate="<|endoftext|>",
#                              include_prefix=False,
#                              nsamples=1,
#                              batch_size=1,
#                              return_as_list=True,
#                              run_name=RUN_NAME,
#                              checkpoint_dir='generator_models/'
#                              )[0]
#
#         print('Title:', title)
#         print('Body:', body)


def generate_to_file(sess, run_name, subreddit, n=1000, temp=0.7):
    file = subreddit + '_gentext_{:%Y%m%d_%H%M%S}.txt'.format(datetime.utcnow())
    path = os.path.join(OUTPUT_DIR, file)

    print('Generating', n, 'submissions to file', file)

    try:
        gpt2.generate_to_file(sess,
                              destination_path=path,
                              length=50,
                              temperature=temp,
                              nsamples=n,
                              batch_size=20,
                              top_k=50, # test
                              top_p=0.95, # test
                              checkpoint_dir='generator_models/',
                              prefix="<|startoftext|>",
                              truncate="<|endoftext|>",
                              include_prefix=True,
                              run_name=run_name)
    except UnicodeEncodeError:
        print('Stopping early due to an encoding error.')


# The model is loaded from a path written in the file 'checkpoint' in the model folder.
# If the model was trained in colab that file will contain the path to the google drive folder.
# This function replaces that path with the correct one.
def update_checkpoint(run_name):
    file_name = 'generator_models/' + run_name + '/checkpoint'
    new_text = ''
    with open(file_name, 'r') as f:
        for line in f:
            key, value, _ = line.split('"')
            new_text += key + '"'
            new_text += value.split('/')[-1]
            new_text += '"\n'

    with open(file_name, 'w') as f:
        f.write(new_text)


def main():
    if len(sys.argv) < 4:
        print('Usage: python run_generator.py RUN_NAME SUBREDDIT NO_SAMPLES (TEMPERATURE)')
        return

    run_name = sys.argv[1]
    subreddit = sys.argv[2]
    try:
        no_samples = int(sys.argv[3])
    except Exception as e:
        print(e)
        print('Third argument should be an integer')
        return

    temperature = 1
    if len(sys.argv) >= 5:
        try:
            temperature = float(sys.argv[4])
        except Exception as e:
            print(e)
            print('Fourth argument should be a float')
            return

    update_checkpoint(run_name)

    sess = gpt2.start_tf_sess()
    gpt2.load_gpt2(sess, run_name=run_name, checkpoint_dir='generator_models')

    generate_to_file(sess, run_name, subreddit, n=no_samples, temp=temperature)

    print('Done.')


if __name__ == '__main__':
    main()
