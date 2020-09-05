import pandas as pd

df_cnn = pd.read_csv('predictions/' + 'CNN_predictions.csv', sep=',')
df_bert = pd.read_csv('predictions/' + 'BERT_predictions.csv', sep=',')
df_test = pd.read_csv('processed_data/' + 'jokes' + '_test.csv', sep='<endoftext>', engine='python')

print(len(df_cnn), len(df_bert), len(df_test))
assert(len(df_bert) == len(df_test))

length = len(df_cnn)
df_bert = df_bert[:length]
df_test = df_test[:length]

print(len(df_cnn), len(df_bert), len(df_test))

test = [not bool(x) for x in df_cnn['test']]
cnn_pred = [not bool(x) for x in df_cnn['pred']]
bert_pred = [not bool(x) for x in df_bert['pred']]
text = [x for x in df_test['text']]

both_nr = 0
cnn_nr = 0
bert_nr = 0

f = open('predictions/bert_exclusive.txt', 'w')

for i in range(length):
    if test[i]:
        if cnn_pred[i] and bert_pred[i]:
            both_nr += 1
        elif cnn_pred[i]:
            cnn_nr += 1
        elif bert_pred[i]:
            bert_nr += 1
            f.write(text[i] + '\n')

print('both:', both_nr)
print('cnn:', cnn_nr)
print('bert:', bert_nr)
