'''
This file is a modified version of https://github.com/Moradnejad/ColBERT-Using-BERT-Sentence-Embedding-for-Humor-Detection
'''

import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import random
import sklearn
import pickle
from transformers import *
from tqdm import tqdm
import getopt
import sys

random.seed(42)

print(tf.test.gpu_device_name())
np.set_printoptions(suppress=True)
print(tf.__version__)


MODEL_TYPE = 'bert-base-uncased'
MAX_SIZE = 200
BATCH_SIZE = 200

MAX_SEQUENCE_LENGTH = 200
SUBREDDIT = 'antijokes'
training_epochs = 0
load_model = False
rebalance_classes = False
LR = 0.000001

argument_list = sys.argv[1:]
options = "s:n:lra:"
long_options = ["subreddit=", "no_epochs=", "load_model", "rebalance", "learning_rate="]

try:
    arguments, values = getopt.getopt(argument_list, options, long_options)

    for current_argument, current_value in arguments:
        if current_argument in ("-s", "--subreddit"):
            SUBREDDIT = current_value

        elif current_argument in ("-l", "--no_epochs"):
            training_epochs = int(current_value)

        elif current_argument in ("-r", "--load_model"):
            load_model = True

        elif current_argument in ("-r", "--rebalance"):
            rebalance_classes = True

        elif current_argument in ("-r", "--learning_rate"):
            LR = float(current_value)
except getopt.error as err:
    print(str(err))

print(SUBREDDIT, training_epochs, load_model, rebalance_classes, LR)

"""
#### 2. Read data
"""

tokenizer = BertTokenizer.from_pretrained(MODEL_TYPE)

df = pd.read_csv('processed_data/' + SUBREDDIT + '.csv', sep='<endoftext>', engine='python')
df_train = pd.read_csv('processed_data/' + SUBREDDIT + '_train.csv', sep='<endoftext>', engine='python')
df_test = pd.read_csv('processed_data/' + SUBREDDIT + '_test.csv', sep='<endoftext>', engine='python')
df_dev = pd.read_csv('processed_data/' + SUBREDDIT + '_dev.csv', sep='<endoftext>', engine='python')

print(df_train.head(3))
print(df_test.head(3))

if rebalance_classes:
    print(df_train.humor.value_counts())
    df_majority = df_train[df_train.humor == 0]
    df_minority = df_train[df_train.humor == 1]

    df_minority_upsampled = df_minority.sample(n=len(df_majority), replace=True, random_state=42)

    df_train = pd.concat([df_majority, df_minority_upsampled])

    print(df_train.humor.value_counts())


test_df_y = df_test.copy()
del df_test['humor']

df_sub = test_df_y.copy()

print(len(df),len(df_train),len(df_test))
print(df_train.head())
print(df_test.head())

print(list(df_train.columns))

output_categories = list(df_train.columns[[1]])
input_categories = list(df_train.columns[[0]])

TARGET_COUNT = len(output_categories)

print('\ninput categories:\n\t', input_categories)
print('\noutput TARGET_COUNT:\n\t', TARGET_COUNT)
print('\noutput categories:\n\t', output_categories)

"""#### 2. Preprocessing functions

These are some functions that will be used to preprocess the raw text data into useable Bert inputs.

credits to [Minh](https://www.kaggle.com/dathudeptrai) for this implementation.
"""


def _convert_to_transformer_inputs(title, question, answer, tokenizer, max_sequence_length):
    """Converts tokenized input to ids, masks and segments for transformer (including bert)"""
    
    def return_id(str1, str2, truncation_strategy, length):

        inputs = tokenizer.encode_plus(str1, str2,
            add_special_tokens=True,
            max_length=length,
            truncation_strategy=truncation_strategy,
            truncation=True)
        
        input_ids =  inputs["input_ids"]
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


outputs = compute_output_arrays(df_train, output_categories)
inputs = compute_input_arrays(df_train, input_categories, tokenizer, MAX_SEQUENCE_LENGTH)
test_inputs = compute_input_arrays(df_test, input_categories, tokenizer, MAX_SEQUENCE_LENGTH)
dev_inputs = compute_input_arrays(df_dev, input_categories, tokenizer, MAX_SEQUENCE_LENGTH)
dev_outputs = compute_output_arrays(df_dev, output_categories)

def create_model():
    q_id = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
    a_id = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
    
    q_mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
    a_mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
    
    q_atn = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
    a_atn = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
    
    config = BertConfig() # print(config) to see settings
    config.output_attentions = True # Set to True to obtain hidden states
    # caution: when using e.g. XLNet, XLNetConfig() will automatically use xlnet-large config
    
    bert_model = TFBertModel.from_pretrained('bert-base-uncased', config=config)
    
    # if config.output_hidden_states = True, obtain hidden states via bert_model(...)[-1]
    q_embedding = bert_model(q_id, attention_mask=q_mask, token_type_ids=q_atn)[0]
    a_embedding = bert_model(a_id, attention_mask=a_mask, token_type_ids=a_atn)[0]

    q = tf.keras.layers.GlobalAveragePooling1D()(q_embedding)
    a = tf.keras.layers.GlobalAveragePooling1D()(a_embedding)
    
#     x = tf.keras.layers.Concatenate()([q, q])
    
    x = tf.keras.layers.Dropout(0.2)(q)
    
    x = tf.keras.layers.Dense(TARGET_COUNT, activation='sigmoid')(x)

    model = tf.keras.models.Model(inputs=[q_id, q_mask, q_atn, ], outputs=x)
    
    return model, bert_model

# Evaluation Metrics
def print_evaluation_metrics(y_true, y_pred, label='', is_regression=True, label2=''):
    print('==================', label2)
    ### For regression
    if is_regression:
        print('mean_absolute_error',label,':', sklearn.metrics.mean_absolute_error(y_true, y_pred))
        print('mean_squared_error',label,':', sklearn.metrics.mean_squared_error(y_true, y_pred))
        print('r2 score',label,':', sklearn.metrics.r2_score(y_true, y_pred))
        #     print('max_error',label,':', sklearn.metrics.max_error(y_true, y_pred))
        return sklearn.metrics.mean_squared_error(y_true, y_pred)
    else:
        ### FOR Classification
        print('balanced_accuracy_score',label,':', sklearn.metrics.balanced_accuracy_score(y_true, y_pred))
        print('average_precision_score',label,':', sklearn.metrics.average_precision_score(y_true, y_pred))
        print('balanced_accuracy_score',label,':', sklearn.metrics.balanced_accuracy_score(y_true, y_pred))
        print('accuracy_score',label,':', sklearn.metrics.accuracy_score(y_true, y_pred))
        print('f1_score',label,':', sklearn.metrics.f1_score(y_true, y_pred))
        
        matrix = sklearn.metrics.confusion_matrix(y_true, y_pred)
        print(matrix)
        TP,TN,FP,FN = matrix[1][1],matrix[0][0],matrix[0][1],matrix[1][0]
        Accuracy = (TP+TN)/(TP+FP+FN+TN)
        Precision = TP/(TP+FP)
        Recall = TP/(TP+FN)
        F1 = 2*(Recall * Precision) / (Recall + Precision)
        print('Acc', Accuracy, 'Prec', Precision, 'Rec', Recall, 'F1',F1)
        return sklearn.metrics.accuracy_score(y_true, y_pred)


def max_acc(y_true, y_pred):
    ret = 0
    best_split = 0
    for split_val in np.arange(0.05, 0.9, 0.001):
        acc = sklearn.metrics.balanced_accuracy_score(y_true, y_pred >= split_val)
        if acc > ret:
            ret = acc
            best_split = split_val

    return ret, best_split


print(len(dev_inputs), len(dev_inputs[0]))

# TRAINING
min_test = []
test_preds = []
valid_preds = []
best_model = False
best_split = 0
max_score = 0

print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

train_inputs = inputs
train_outputs = outputs

valid_inputs = dev_inputs
valid_outputs = dev_outputs

print(np.array(train_inputs).shape, np.array(train_outputs).shape)

K.clear_session()
model, bert_model = create_model()
optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
model.compile(loss='binary_crossentropy', optimizer=optimizer)

if load_model:
    print('Loading best model...')
    t_inputs = [(inputs[i][:])[:1] for i in range(len(inputs))]
    t_outputs = (outputs[:])[:1]
    model.fit(t_inputs, t_outputs, epochs=1, batch_size=6)

    model.load_weights('predictor_models/' + SUBREDDIT + '/weights.h5')
    with open('predictor_models/' + SUBREDDIT + '/optimizer.pkl', 'rb') as f:
        weight_values = pickle.load(f)
    model.optimizer.set_weights(weight_values)

    valid_preds.append(model.predict(valid_inputs, verbose=1))
    max_score, best_split = max_acc(np.array(valid_outputs), np.array(valid_preds[-1]))
    print('Current max score:', max_score)

    best_model = model

for i in range(training_epochs):
    print('Starting epoch', i)

    model.fit(train_inputs, train_outputs, epochs=1, batch_size=6)

    valid_preds.append(model.predict(valid_inputs, verbose=1))
    score, split_val = max_acc(np.array(valid_outputs), np.array(valid_preds[-1]))

    print('score', score)
    if score >= max_score:
        print('new score >> ', score)
        print('split value: ', split_val)
        print('Saving model...')
        max_score = score
        best_model = model
        best_split = split_val

        # SAVE MODEL
        best_model.save_weights('predictor_models/' + SUBREDDIT + '/weights.h5')
        symbolic_weights = getattr(best_model.optimizer, 'weights')
        weight_values = K.batch_get_value(symbolic_weights)
        with open('predictor_models/' + SUBREDDIT + '/optimizer.pkl', 'wb') as f:
            pickle.dump(weight_values, f)

        print(' ')
    else:
        print('score did not improve')

len(valid_inputs[0])

print(valid_outputs.shape, valid_preds[-1].shape)
print_evaluation_metrics(np.array(valid_outputs), np.array(valid_preds[-1]), '')

min_test = best_model.predict(test_inputs, verbose=1)

df_sub = test_df_y.copy()
df_sub['pred'] = min_test

print_evaluation_metrics(df_sub['humor'], df_sub['pred'], '', True)

y_test = df_sub['humor']
y_pred = (df_sub['pred'] > best_split)

print('Best split:', best_split)
print('Balanced accuracy:', sklearn.metrics.balanced_accuracy_score(y_test, y_pred))
print('F1 Score:', sklearn.metrics.f1_score(y_test, y_pred))
print('Accuracy:', sklearn.metrics.accuracy_score(y_test, y_pred))
print('Precision:', sklearn.metrics.average_precision_score(y_test, y_pred))
print('Recall:', sklearn.metrics.recall_score(y_test, y_pred))

with open('predictions/' + SUBREDDIT + '_predictions.csv', 'w') as f:
    f.write('test,pred\n')
    for i in range(len(y_pred)):
        f.write(str(y_test[i]) + ',' + str(y_pred[i]) + ',' + df_sub['text'][i] + '\n')

