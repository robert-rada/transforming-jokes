import gpt_2_simple as gpt2
from datetime import datetime

RUN_NAME="jokes1"

# gpt2.download_gpt2(model_name="355M")

sess = gpt2.start_tf_sess()

gpt2.finetune(sess,
              dataset='jokes.txt',
              model_name='355M',
              steps=5000,
              restore_from='fresh',
              run_name=RUN_NAME,
              print_every=50,
              sample_every=1000,
              save_every=500,
              learning_rate=0.00001,
              checkpoint_dir='generator_models',
              batch_size=1,
              use_memory_saving_gradients=True,
              )


# sess = gpt2.start_tf_sess()
# gpt2.load_gpt2(sess, run_name=RUN_NAME)



# gpt2.generate(sess,
#               length=250,
#               temperature=0.7,
#               prefix="<|startoftext|>",
#               truncate="<|endoftext|>",
#               include_prefix=False,
#               nsamples=5,
#               batch_size=5,
#               )[0]

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
              )[0]
body = gpt2.generate(sess,
              length=50,
              temperature=0.7,
              prefix=title,
              truncate="<|endoftext|>",
              include_prefix=False,
              nsamples=1,
              batch_size=1,
              return_as_list=True,
              run_name=RUN_NAME,
              )[0]

print('Title:', title)
print('Body:', body)


gen_file = 'gpt2_gentext_{:%Y%m%d_%H%M%S}.txt'.format(datetime.utcnow())

gpt2.generate_to_file(sess,
                      destination_path=gen_file,
                      length=500,
                      temperature=0.7,
                      nsamples=10000,
                      batch_size=20
                      )

