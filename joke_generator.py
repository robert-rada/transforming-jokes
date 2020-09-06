import gpt_2_simple as gpt2
import sys

gpt2.download_gpt2(model_name="355M")

SUBREDDIT = sys.argv[1]

file_name = "processed_data/" + SUBREDDIT + ".txt"
RUN_NAME = SUBREDDIT + "1"
CHKPOINT_DIR = 'generator_models/'

sess = gpt2.start_tf_sess()

gpt2.finetune(sess,
              dataset=file_name,
              model_name='355M',
              steps=-1,
              restore_from='latest',
              run_name=RUN_NAME,
              print_every=50,
              sample_every=1000,
              save_every=500,
              learning_rate=0.00001,
              checkpoint_dir=CHKPOINT_DIR,
              overwrite=True,
              )
