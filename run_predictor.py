import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import os
import glob
import sys
import getopt
from tqdm import tqdm
from datetime import datetime
from transformers import *

from similarity_test import remove_reposts

MODEL_TYPE = 'bert-base-uncased'
MAX_SEQUENCE_LENGTH = 200
GEN_FILE_FOLDER = 'outputs'
SEPARATOR = '===================='


def _convert_to_transformer_inputs(title, question, answer, tokenizer, max_sequence_length):
    def return_id(str1, str2, truncation_strategy, length):
        inputs = tokenizer.encode_plus(str1, str2,
                                       add_special_tokens=True,
                                       max_length=length,
                                       truncation_strategy=truncation_strategy,
                                       truncation=True)

        input_ids = inputs["input_ids"]
        input_masks = [1] * len(input_ids)
        input_segments = inputs["token_type_ids"]
        padding_length = length - len(input_ids)
        padding_id = tokenizer.pad_token_id
        input_ids = input_ids + ([padding_id] * padding_length)
        input_masks = input_masks + ([0] * padding_length)
        input_segments = input_segments + ([0] * padding_length)

        return [input_ids, input_masks, input_segments]

    input_ids_q, input_masks_q, input_segments_q = return_id(
        title, None, 'longest_first', max_sequence_length)

    input_ids_a, input_masks_a, input_segments_a = return_id(
        '', None, 'longest_first', max_sequence_length)

    return [input_ids_q, input_masks_q, input_segments_q,
            input_ids_a, input_masks_a, input_segments_a]


def compute_input_arrays(df, columns, tokenizer, max_sequence_length):
    input_ids_q, input_masks_q, input_segments_q = [], [], []
    input_ids_a, input_masks_a, input_segments_a = [], [], []
    for _, instance in tqdm(df[columns].iterrows()):
        t, q, a = instance.text, instance.text, instance.text

        try:
            ids_q, masks_q, segments_q, ids_a, masks_a, segments_a = \
                _convert_to_transformer_inputs(t, q, a, tokenizer, max_sequence_length)
        except Exception as e:
            print(e)
            print(t)

        input_ids_q.append(ids_q)
        input_masks_q.append(masks_q)
        input_segments_q.append(segments_q)
        input_ids_a.append(ids_a)
        input_masks_a.append(masks_a)
        input_segments_a.append(segments_a)

    return [np.asarray(input_ids_q, dtype=np.int32),
            np.asarray(input_masks_q, dtype=np.int32),
            np.asarray(input_segments_q, dtype=np.int32),
            np.asarray(input_ids_a, dtype=np.int32),
            np.asarray(input_masks_a, dtype=np.int32),
            np.asarray(input_segments_a, dtype=np.int32)]


def compute_output_arrays(df, columns):
    return np.asarray(df[columns])


def create_model():
    q_id = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
    a_id = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)

    q_mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
    a_mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)

    q_atn = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
    a_atn = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)

    config = BertConfig()  # print(config) to see settings
    config.output_hidden_states = False  # Set to True to obtain hidden states
    # caution: when using e.g. XLNet, XLNetConfig() will automatically use xlnet-large config

    bert_model = TFBertModel.from_pretrained('bert-base-uncased', config=config)

    # if config.output_hidden_states = True, obtain hidden states via bert_model(...)[-1]
    q_embedding = bert_model(q_id, attention_mask=q_mask, token_type_ids=q_atn)[0]
    a_embedding = bert_model(a_id, attention_mask=a_mask, token_type_ids=a_atn)[0]

    q = tf.keras.layers.GlobalAveragePooling1D()(q_embedding)
    a = tf.keras.layers.GlobalAveragePooling1D()(a_embedding)

    x = tf.keras.layers.Dropout(0.2)(q)

    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.models.Model(inputs=[q_id, q_mask, q_atn, ], outputs=x)

    return model


def valid_submission(text):
    if text.find('<|startoftext|>') == -1:
        return False
    if text.find('<|endoftitle|>') == -1:
        return False
    return True


def read_generated_text(subreddit):
    print('Reading generated text...')

    text_list = []
    text_list_with_tokens = []
    gen_file_prefix = subreddit + '_gentext'

    for file_name in glob.glob(os.path.join(GEN_FILE_FOLDER, gen_file_prefix + '*')):
        with open(file_name, 'r') as file:
            for submission in file.read().split(SEPARATOR):
                if not valid_submission(submission):
                    continue

                submission = submission.lstrip()
                submission = submission.replace('&#x200B;', '')

                text_list_with_tokens.append(submission[:])

                submission = submission.replace('<|startoftext|>', '')
                submission = submission.replace('<|endoftitle|>', '')

                text_list.append(submission)

    d = {'text': text_list, 'text_token': text_list_with_tokens}

    df = pd.DataFrame(d)
    print(df)

    return df


def write_result_to_file(df, subreddit):
    file_name = subreddit + '_pred_{:%Y%m%d_%H%M%S}.txt'.format(datetime.utcnow())
    path = os.path.join(GEN_FILE_FOLDER, file_name)

    d = df.to_dict('list')

    with open(path, 'w') as file:
        for i in range(len(df)):
            try:
                file.write(SEPARATOR + '\n')
                file.write(str(d['pred'][i]) + '\n')
                file.write(d['text'][i] + '\n')
            except:
                pass
        file.write(SEPARATOR)


def main2():
    if len(sys.argv) < 2:
        print('Usage: python run_predictor.py SUBREDDIT')
        return

    subreddit = sys.argv[1]

    df_test = pd.read_csv('processed_data/' + subreddit + '_test.csv', sep='<endoftext>')
    df_dev = pd.read_csv('processed_data/' + subreddit + '_dev.csv', sep='<endoftext>')

    df = pd.concat([df_test, df_dev])

    df = df[df.score < 2]

    print('before removing reposts:', len(df))
    df = remove_reposts(df, subreddit, tolerance=0.75)
    print('after removing reposts:', len(df))

    tokenizer = BertTokenizer.from_pretrained(MODEL_TYPE)

    inputs = compute_input_arrays(df, ['text'], tokenizer, MAX_SEQUENCE_LENGTH)

    print('Loading model...')

    K.clear_session()
    model = create_model()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)
    model.compile(loss='binary_crossentropy', optimizer=optimizer)

    t_inputs = [(inputs[i][:])[:1] for i in range(len(inputs))]
    t_outputs = np.array([[False]])
    model.fit(t_inputs, t_outputs, epochs=1, batch_size=1)
    model.load_weights('predictor_models/' + subreddit + '/weights.h5')

    prediction = model.predict(inputs, verbose=1)
    df['pred'] = prediction
    df = df.sort_values('pred', ascending=False)
    print(df.head())
    write_result_to_file(df, subreddit)


def main():
    if len(sys.argv) < 2:
        print('Usage: python run_predictor.py SUBREDDIT')
        return

    subreddit = sys.argv[1]

    df = read_generated_text(subreddit)

    print('before removing reposts:', len(df))
    df = remove_reposts(df, subreddit, tolerance=0.75)
    print('after removing reposts:', len(df))

    tokenizer = BertTokenizer.from_pretrained(MODEL_TYPE)

    inputs = compute_input_arrays(df, ['text'], tokenizer, MAX_SEQUENCE_LENGTH)

    print('Loading model...')

    K.clear_session()
    model = create_model()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)
    model.compile(loss='binary_crossentropy', optimizer=optimizer)

    t_inputs = [(inputs[i][:])[:1] for i in range(len(inputs))]
    t_outputs = np.array([[False]])
    model.fit(t_inputs, t_outputs, epochs=1, batch_size=1)
    model.load_weights('predictor_models/' + subreddit + '/weights.h5')

    prediction = model.predict(inputs, verbose=1)
    df['pred'] = prediction
    df = df.sort_values('pred', ascending=False)
    print(df.head())
    write_result_to_file(df, subreddit)


if __name__ == '__main__':
    main()

