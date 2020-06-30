import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import GroupKFold
import matplotlib.pyplot as plt
from tqdm import tqdm
# import tensorflow_hub as hub
import tensorflow as tf
# import bert_tokenization as tokenization
import tensorflow.keras.backend as K
import os
from scipy.stats import spearmanr
from math import floor, ceil
from transformers import *

import string
import re    #for regex
import pickle

print(tf.test.gpu_device_name())
np.set_printoptions(suppress=True)
print(tf.__version__)

"""# Choose model"""

from transformers import BertTokenizer, TFBertModel


MODEL_TYPE = 'bert-base-uncased'
MAX_SIZE = 200
BATCH_SIZE = 200

tokenizer = BertTokenizer.from_pretrained(MODEL_TYPE)

"""#### 1. Read data and tokenizer

Read tokenizer and data, as well as defining the maximum sequence length that will be used for the input to Bert (maximum is usually 512 tokens)
"""

HAS_ANS = False
training_sample_count = 1000 # 4000
training_epochs = 1 # 3
test_count = 1000
running_folds = 1 # 2
MAX_SEQUENCE_LENGTH = 200

"""### original dataset"""

import io


df = pd.read_csv('processed_data/jokes.csv', sep='<endoftext>')

df_train = pd.read_csv('processed_data/jokes_train.csv', sep='<endoftext>')
print(df_train.head(3))

df_test = pd.read_csv('processed_data/jokes_test.csv', sep='<endoftext>')
print(df_test.head(3))

df_dev = pd.read_csv('processed_data/jokes_dev.csv', sep='<endoftext>')


test_df_y = df_test.copy()
del df_test['humor']

df_sub = test_df_y.copy()

print(len(df),len(df_train),len(df_test))
print(df_train.head())
print(df_test.head())

print(list(df_train.columns))

output_categories = list(df_train.columns[[1]])
input_categories = list(df_train.columns[[0]])

if HAS_ANS:
    output_categories = list(df_train.columns[11:])
    input_categories = list(df_train.columns[[1,2,5]])
    

TARGET_COUNT = len(output_categories)

print('\ninput categories:\n\t', input_categories)
print('\noutput TARGET_COUNT:\n\t', TARGET_COUNT)
print('\noutput categories:\n\t', output_categories)

"""#### 2. Preprocessing functions

These are some functions that will be used to preprocess the raw text data into useable Bert inputs.<br>

*update 4:* credits to [Minh](https://www.kaggle.com/dathudeptrai) for this implementation. If I'm not mistaken, it could be used directly with other Huggingface transformers too! Note that due to the 2 x 512 input, it will require significantly more memory when finetuning BERT.
"""

def _convert_to_transformer_inputs(title, question, answer, tokenizer, max_sequence_length):
    """Converts tokenized input to ids, masks and segments for transformer (including bert)"""
    
    def return_id(str1, str2, truncation_strategy, length):

        inputs = tokenizer.encode_plus(str1, str2,
            add_special_tokens=True,
            max_length=length,
            truncation_strategy=truncation_strategy)
        
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


"""## 3. Create model

~~`compute_spearmanr()`~~ `mean_squared_error` is used to compute the competition metric for the validation set
<br><br>
`create_model()` contains the actual architecture that will be used to finetune BERT to our dataset.
"""

def create_model():
    q_id = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
    a_id = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
    
    q_mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
    a_mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
    
    q_atn = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
    a_atn = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
    
    config = BertConfig() # print(config) to see settings
    config.output_hidden_states = False # Set to True to obtain hidden states
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
    
    return model

"""## 5. Training, validation and testing

Loops over the folds in gkf and trains each fold for 3 epochs --- with a learning rate of 3e-5 and batch_size of 6. A simple binary crossentropy is used as the objective-/loss-function.
"""

# Evaluation Metrics
import sklearn
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

print_evaluation_metrics([1,0], [0.9,0.1], '', True)
print_evaluation_metrics([1,0], [1,1], '', False)

print(len(dev_inputs), len(dev_inputs[0]))

"""### Training
Regression problem between 0 and 1, so binary_crossentropy and mean_absolute_error seem good.

Here are the explanations: https://www.dlology.com/blog/how-to-choose-last-layer-activation-and-loss-function/
"""

min_err = 1000000
min_test = []
valid_preds = []
test_preds = []
best_model = False
# for LR in np.arange(1e-5, 2e-5, 3e-5).tolist():
for LR in [1e-5]:
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    print('LR=', LR)

    # train_inputs = [(inputs[i][:])[:training_sample_count] for i in range(len(inputs))]
    # train_outputs = (outputs[:])[:training_sample_count]
    train_inputs = inputs
    train_outputs = outputs

    valid_inputs = dev_inputs
    valid_outputs = dev_outputs

    print(np.array(train_inputs).shape, np.array(train_outputs).shape)

    K.clear_session()
    model = create_model()
    optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
    model.compile(loss='binary_crossentropy', optimizer=optimizer)

    # LOAD MODEL
    t_inputs = [(inputs[i][:])[:1] for i in range(len(inputs))]
    t_outputs = (outputs[:])[:1]
    model.fit(t_inputs, t_outputs, epochs=1, batch_size=6)

    model.load_weights('jokes_model/weights.h5')
    with open('jokes_model/optimizer.pkl', 'rb') as f:
        weight_values = pickle.load(f)
    model.optimizer.set_weights(weight_values)

    model.fit(train_inputs, train_outputs, epochs=training_epochs, batch_size=6)
    print('finished fit')
            # model.save_weights(f'bert-{fold}.h5')
    valid_preds.append(model.predict(valid_inputs, verbose=1))
    print('finished predict')
            #rho_val = compute_spearmanr_ignore_nan(valid_outputs, valid_preds[-1])
            #print('validation score = ', rho_val)
    acc = print_evaluation_metrics(np.array(valid_outputs), np.array(valid_preds[-1]), 'on #'+str(1))
    print('acc', acc)
    if acc < min_err:
        print('new acc >> ', acc)
        min_err = acc
        best_model = model
#                 min_test = model.predict(test_inputs)
#                 test_preds.append(min_test)

        # SAVE MODEL
        best_model.save_weights('/content/gdrive/My Drive/jokes_model/weights2.h5')
        symbolic_weights = getattr(best_model.optimizer, 'weights')
        weight_values = K.batch_get_value(symbolic_weights)
        with open('/content/gdrive/My Drive/jokes_model/optimizer2.pkl', 'wb') as f:
            pickle.dump(weight_values, f)

        print(' ')

print('best acc >> ', acc)

len(valid_inputs[0])

print(valid_outputs.shape, valid_preds[-1].shape)
print_evaluation_metrics(np.array(valid_outputs), np.array(valid_preds[-1]), '')

# %%time
min_test = best_model.predict(test_inputs, verbose=1)

## use min_test
# min_test

df_test.to_csv('df_test.csv')
test_df_y.to_csv('test_df_y.csv')

"""## Regression submission"""

df_sub = test_df_y.copy()
# df_sub['pred'] = np.average(test_preds, axis=0) # for weighted average set weights=[...]
df_sub['pred'] = min_test


print_evaluation_metrics(df_sub['humor'], df_sub['pred'], '', True)


df_sub.to_csv('sub1.csv', index=False)
df_sub.head()

"""## Binary submission"""

for split in np.arange(0.1, 0.99, 0.1).tolist():
    df_sub['pred_bi'] = (df_sub['pred'] > split)

    print_evaluation_metrics(df_sub['humor'], df_sub['pred_bi'], '', False, 'SPLIT on '+str(split))

    df_sub.to_csv('sub3.csv', index=False)
    df_sub.head()
