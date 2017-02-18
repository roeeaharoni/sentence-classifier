from utils import *
import dynet as dy


UNK = 'UNK'
EOS = 'EOS'


def main():

    train_sents, train_labels, dev_sents, dev_labels, test_sents, test_labels, word2int = load_data()

    # TODO: implement
    model = build_model(word2int)

    # TODO: implement
    train_model(model, train_sents, train_labels, dev_sents, dev_labels)

    model.load('best_model.txt')

    # TODO: implement
    predictions = predict(model, test_sents)

    evaluate_model(predictions, test_labels)

    return


def build_model():
    # TODO: build the model. you should add the appropriate input and output parameters to this method
    return


def compute_network_output(model, sent):
    dy.renew_cg()
    # TODO: compute the network output for a sentence
    network_output = None
    return network_output


def one_sent_loss(model, sent, label):
    # TODO: compute the loss for a sentence. you may use compute_network_output() to do this.
    loss = None
    return loss


def train_model(model, train_sents, train_labels, dev_sents, dev_labels):
    # TODO: train the model

    # trainer = dy.AdadeltaTrainer(model)

    # for every example, you should compute the loss, backpropagate and update by using the following commands:
    # loss_value = loss.value()
    # loss.backward()
    # trainer.update()

    # save the best model
    # model.save('best_model.txt')

    return


def predict(model, test_sents):
    # TODO: predict using a trained model
    # predict using a trained model
    predictions = None
    return predictions


if __name__ == '__main__':
    main()
