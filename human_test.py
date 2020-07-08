import pandas as pd
import sklearn


def main():
    subreddit = 'antijokes'
    no_samples = 100

    df = pd.read_csv('processed_data/' + subreddit + '.csv', sep='<endoftext>', engine='python')
    samples = df.sample(no_samples).values.tolist()

    print(len(df))
    return

    tp = 0
    fp = 0
    tn = 0
    fn = 0
    guesses = []
    for sample in samples:
        print(sample[0])

        ans = ''
        while ans not in {'y', 'n'}:
            ans = input()

        guesses.append(ans == 'y')

        if ans == 'y' and sample[1]:
            tp += 1
        if ans == 'y' and not sample[1]:
            fp += 1
        if ans == 'n' and sample[1]:
            fn += 1
        if ans == 'n' and not sample[1]:
            tn += 1

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    balanced_acc = (tp / (tp + fn) + tn / (tn + fp)) / 2
    f1_score = 2 * precision * recall / (precision + recall)

    print('TP:', tp)
    print('FP:', fp)
    print('TN:', tn)
    print('FN:', fn)

    print('Accuracy:', accuracy)
    print('Precision:', precision)
    print('Recall:', recall)
    print('Balanced accuracy:', balanced_acc)
    print('F1 score:', f1_score)

    print('====================')
    y_true = [sample[1] for sample in samples]
    y_pred = guesses
    print('balanced_accuracy_score', ':', sklearn.metrics.balanced_accuracy_score(y_true, y_pred))
    print('average_precision_score', ':', sklearn.metrics.average_precision_score(y_true, y_pred))
    print('balanced_accuracy_score', ':', sklearn.metrics.balanced_accuracy_score(y_true, y_pred))
    print('accuracy_score', ':', sklearn.metrics.accuracy_score(y_true, y_pred))
    print('f1_score', ':', sklearn.metrics.f1_score(y_true, y_pred))


if __name__ == '__main__':
    main()
