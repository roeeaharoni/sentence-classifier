import random
import dynet as dy
import pandas as pd
import numpy as np
from collections import Counter
import nltk

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_recall_curve
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression


UNK = 'UNK'
EOS = 'EOS'
INPUT_DIM = 200
HIDDEN_DIM = 200
OUTPUT_DIM = 2
EPOCHS = 100


def build_model(vocab):
    model = dy.Model()
    embeddings_lookup = model.add_lookup_parameters((len(vocab), INPUT_DIM))

    hidden_W = model.add_parameters((HIDDEN_DIM, HIDDEN_DIM))
    hidden_bias = model.add_parameters(HIDDEN_DIM)

    MLP_W = model.add_parameters((OUTPUT_DIM, HIDDEN_DIM))
    MLP_bias = model.add_parameters(OUTPUT_DIM)

    encoder_lstm = dy.LSTMBuilder(layers=1, hidden_dim=HIDDEN_DIM, input_dim=INPUT_DIM, model=model)

    return model, embeddings_lookup, hidden_W, hidden_bias, MLP_W, MLP_bias, encoder_lstm


def one_sent_loss(model, embeddings_lookup, hidden_W, hidden_bias, MLP_W, MLP_bias, encoder_lstm, sent, label,
                  word2int):
    mlp_output = compute_network_output(embeddings_lookup, encoder_lstm, hidden_W, hidden_bias, MLP_W, MLP_bias, sent,
                                        word2int)
    loss = dy.pickneglogsoftmax(mlp_output, label)
    return loss


def compute_network_output(embeddings_lookup, encoder_lstm, hidden_W, hidden_bias, MLP_W, MLP_bias, sent, word2int):
    dy.renew_cg()
    hidden_W = dy.parameter(hidden_W)
    hidden_bias = dy.parameter(hidden_bias)
    mlp_W = dy.parameter(MLP_W)
    bias = dy.parameter(MLP_bias)

    # encode
    embeddings = []
    for w in sent + [EOS]:
        if w in word2int:
            embeddings.append(embeddings_lookup[word2int[w]])
        else:
            embeddings.append(embeddings_lookup[word2int[UNK]])

    # transduce
    s = encoder_lstm.initial_state()
    outputs = s.transduce(embeddings)
    last_lstm_output = outputs[-1]

    # hidden layer
    hidden_output = dy.tanh(hidden_W * last_lstm_output + hidden_bias)

    # MLP
    mlp_output = mlp_W * hidden_output + bias

    return mlp_output


def train_model(model, embeddings_lookup, hidden_W, hidden_bias, MLP_W, MLP_bias, encoder_lstm, train_sents,
                train_labels, dev_sents, dev_labels, word2int):
    print 'training...'
    aggregated_loss = 0
    trainer = dy.AdadeltaTrainer(model)
    train_len = len(train_sents)
    patience = 10
    best_dev = 0
    avg_loss = 0
    for e in xrange(EPOCHS):
        print 'starting epoch {}'.format(e)

        # randomize the training set
        indices = range(train_len)
        random.shuffle(indices)
        train_set = zip(train_sents, train_labels)
        shuffled_train_set = [train_set[i] for i in indices]

        # compute loss for each example and update
        for i, example in enumerate(shuffled_train_set):
            sent, label = example
            loss = one_sent_loss(model, embeddings_lookup, hidden_W, hidden_bias, MLP_W, MLP_bias, encoder_lstm, sent,
                                 label, word2int)

            loss_value = loss.value()
            aggregated_loss += loss_value
            loss.backward()
            trainer.update()
            if i > 0:
                avg_loss = aggregated_loss / float(i + e * train_len)
            else:
                avg_loss = aggregated_loss

            if i % 10000 == 0:
                print 'epoch: {} avg. loss: {} went through {} examples'.format(e, avg_loss, i)

        # evaluate on dev after each epoch:
        dev_score = evaluate_model(model, embeddings_lookup, hidden_W, hidden_bias, MLP_W, MLP_bias, encoder_lstm,
                                   dev_sents, dev_labels,
                                   word2int)

        if dev_score < best_dev:
            patience += 1
        else:
            patience = 0
            best_dev = dev_score
            model.save('best_model.txt')

        print 'epoch: {} avg. loss: {} dev acc.: {} best dev acc.:{}'.format(e, avg_loss, dev_score, best_dev)

        if patience > 10:
            return


def evaluate_model(model, embeddings_lookup, hidden_W, hidden_bias, MLP_W, MLP_bias, encoder_lstm, sents, labels,
                   word2int):
    correct = 0.0
    correct_positives = 0.0
    positives = 0
    for i, example in enumerate(zip(sents, labels)):
        sent, label = example
        output = compute_network_output(embeddings_lookup, encoder_lstm, hidden_W, hidden_bias, MLP_W, MLP_bias, sent,
                                        word2int)
        probs = dy.softmax(output).npvalue()
        pred = np.argmax(probs)
        if pred == label:
            correct += 1

        if label == 1:
            positives += 1
            if pred == label:
                correct_positives += 1

    accuracy = correct / len(sents)
    recall = correct_positives / positives
    print 'recall: {}'.format(recall)
    print 'precision: {}'.format(accuracy)
    return accuracy


def load_data():
    print 'loading data...'

    # load the data
    raw_data = pd.read_csv('/Users/roeeaharoni/git/sentence_classifier/switchboard_complete.csv')

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
    # print len(train_sents)
    # print len(dev_sents)
    # print len(test_sents)

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


def log_reg_baseline(train_sents, train_labels, dev_sents, dev_labels, test_sents, test_labels, word2int):
    raw_data = pd.read_csv('/Users/roeeaharoni/git/sentence_classifier/switchboard_complete.csv')
    sents = raw_data.clean_text.fillna("").tolist()

    MAX_FEATURES = 1000
    CLASS_WEIGHT = None  # "balanced"
    model = LogisticRegression(verbose=True, class_weight=CLASS_WEIGHT)
    count_vectorizer = CountVectorizer(max_features=MAX_FEATURES)
    bag_of_words = count_vectorizer.fit_transform(sents)
    train_x = bag_of_words[:len(train_sents)]
    dev_x = bag_of_words[len(train_sents):len(train_sents) + len(dev_sents)]
    test_x = bag_of_words[len(train_sents) + len(dev_sents):]

    print 'train {} {}\n'.format(train_x.shape[0], len(train_sents))
    print 'dev {} {}\n'.format(dev_x.shape[0], len(dev_sents))
    print 'test {} {}\n'.format(test_x.shape[0], len(test_sents))

    model.fit(train_x, train_labels)
    dev_pred = model.predict_proba(dev_x)
    test_pred = model.predict_proba(test_x)

    correct = 0.0
    correct_positives = 0.0
    positives = 0
    for i, example in enumerate(zip(dev_sents, dev_labels)):
        sent, label = example
        probs = dev_pred[i]
        pred = np.argmax(probs)
        if pred == label:
            correct += 1

        if label == 1:
            positives += 1
            if pred == label:
                correct_positives += 1

    accuracy = correct / len(dev_sents)
    recall = correct_positives / positives
    print 'log reg dev recall: {}'.format(recall)
    print 'log reg dev precision: {}'.format(accuracy)

    correct = 0.0
    correct_positives = 0.0
    positives = 0
    for i, example in enumerate(zip(test_sents, test_labels)):
        sent, label = example
        probs = test_pred[i]
        pred = np.argmax(probs)
        if pred == label:
            correct += 1

        if label == 1:
            positives += 1
            if pred == label:
                correct_positives += 1

    accuracy = correct / len(test_sents)
    recall = correct_positives / positives
    print 'log reg test recall: {}'.format(recall)
    print 'log reg test precision: {}'.format(accuracy)



def main():
    train = False

    train_sents, train_labels, dev_sents, dev_labels, test_sents, test_labels, word2int = load_data()

    log_reg_baseline(train_sents, train_labels, dev_sents, dev_labels, test_sents, test_labels, word2int)

    # create the network (single layer lstm with 1-hidden mlp, binary softmax)
    model, embeddings_lookup, hidden_W, hidden_bias, MLP_W, MLP_bias, encoder_lstm = build_model(word2int)
    print 'created the model'

    if train:
        # train the network
        train_model(model, embeddings_lookup, hidden_W, hidden_bias, MLP_W, MLP_bias, encoder_lstm, train_sents,
                train_labels, dev_sents, dev_labels, word2int)

    model.load('best_model.txt')

    print 'rnn dev:'
    evaluate_model(model, embeddings_lookup, hidden_W, hidden_bias, MLP_W, MLP_bias, encoder_lstm,
                   dev_sents, dev_labels, word2int)

    print 'rnn test:'
    evaluate_model(model, embeddings_lookup, hidden_W, hidden_bias, MLP_W, MLP_bias, encoder_lstm,
                   test_sents, test_labels, word2int)

    return


if __name__ == '__main__':
    main()
