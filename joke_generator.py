# %tensorflow_version 1.x
# !pip install -q gpt-2-simple
import gpt_2_simple as gpt2
from datetime import datetime


gpt2.download_gpt2(model_name="355M")

file_name = "processed_data/jokes.txt"
RUN_NAME = "jokes1"
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
