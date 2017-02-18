import pandas as pd
from collections import Counter
import nltk

# use this method to evaluate your model
def evaluate_model(predicted, gold):
    if len(predicted) != len(gold):
        raise Exception('predictions and gold set should be in the same size!')
    correct = 0.0
    correct_positives = 0.0
    positives = 0
    for i, pred in enumerate(predicted):
        if pred == gold[i]:
            correct += 1

        if gold[i] == 1:
            positives += 1
            if pred == gold[i]:
                correct_positives += 1

    accuracy = correct / len(predicted)
    recall = correct_positives / positives
    print 'recall: {}'.format(recall)
    print 'precision: {}'.format(accuracy)
    return accuracy


# load the switchboard dataset, divided to train/dev/test
def load_data():
    print 'loading data...'

    # load the data
    raw_data = pd.read_csv('switchboard_complete.csv')

    # tokenize the sentences
    sents = raw_data.clean_text.tolist()
    tokenized_sents = []
    for s in sents:
        if isinstance(s, basestring):
            tokenized_sent = nltk.word_tokenize(s)
            tokenized_sents.append(tokenized_sent)
        else:
            # print 'not string'
            tokenized_sents.append([])

    # get the binary labels
    is_question = raw_data.act_label_1.str.contains("Info-request:Yes-No-Question|Info-request:Wh-Question")
    binary_labels = is_question == True

    # divide to train-dev-test
    data_size = len(tokenized_sents)
    train_sents = tokenized_sents[:int(0.8 * data_size)]
    dev_sents = tokenized_sents[int(0.8 * data_size):int(0.9 * data_size)]
    test_sents = tokenized_sents[int(0.9 * data_size):data_size]
    train_labels = binary_labels[:int(0.8 * data_size)]
    dev_labels = binary_labels[int(0.8 * data_size):int(0.9 * data_size)]
    test_labels = binary_labels[int(0.9 * data_size):data_size]

    # build vocabulary
    words = []
    for s in tokenized_sents:
        words += s
    vocab = list(set(words))
    vocab.append(UNK)
    vocab.append(EOS)

    word2int = dict(zip(vocab, range(0, len(vocab))))

    print 'finished loading data.'
    print 'vocab size: ' + str(len(vocab))
    cnt1 = Counter(train_labels).most_common()
    cnt2 = Counter(dev_labels).most_common()
    cnt3 = Counter(test_labels).most_common()
    print 'train labels counts: {}'.format(cnt1)
    print 'dev labels counts: {}'.format(cnt2)
    print 'test labels counts: {}'.format(cnt3)

    return train_sents, train_labels, dev_sents, dev_labels, test_sents, test_labels, word2int
