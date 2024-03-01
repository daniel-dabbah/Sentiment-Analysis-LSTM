#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 07:42:49 2024

@author: danieldabbah
"""
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset, Dataset
import operator
import data_loader
import pickle
import tqdm

# ------------------------------------------- Constants ----------------------------------------

SEQ_LEN = 52
W2V_EMBEDDING_DIM = 300

ONEHOT_AVERAGE = "onehot_average"
W2V_AVERAGE = "w2v_average"
W2V_SEQUENCE = "w2v_sequence"

TRAIN = "train"
VAL = "val"
TEST = "test"


# ------------------------------------------ Helper methods and classes --------------------------

def get_available_device():
    """
    Allows training on GPU if available. Can help with running things faster when a GPU with cuda is
    available but not a most...
    Given a device, one can use module.to(device)
    and criterion.to(device) so that all the computations will be done on the GPU.
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_model(model, path, epoch, optimizer):
    """
    Utility function for saving checkpoint of a model, so training or evaluation can be executed later on.
    :param model: torch module representing the model
    :param optimizer: torch optimizer used for training the module
    :param path: path to save the checkpoint into
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()}, path)


def load(model, path, optimizer):
    """
    Loads the state (weights, paramters...) of a model which was saved with save_model
    :param model: should be the same model as the one which was saved in the path
    :param path: path to the saved checkpoint
    :param optimizer: should be the same optimizer as the one which was saved in the path
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return model, optimizer, epoch


class PerformenceAnalysis:
    def __init__(self, title):
        self.title = title
        self.train_loss, self.train_acc, self.val_loss, self.val_acc = [], [], [], []

    def append_train(self, train_loss, train_acc):
        self.train_loss.append(train_loss)
        self.train_acc.append(train_acc)

    def append_validation(self, val_loss, val_acc):
        self.val_loss.append(val_loss)
        self.val_acc.append(val_acc)

    def get_train_loss(self):
        return self.train_loss

    def get_train_acc(self):
        return self.train_acc

    def get_val_loss(self):
        return self.val_loss

    def get_val_acc(self):
        return self.val_acc

    def plot_accuracy(self):
        plt.title("%s Accuracy" % self.title)
        plt.plot(self.train_acc, label='Training')
        plt.plot(self.val_acc, label='Validation')
        plt.legend()
        plt.show()

    def plot_loss(self):
        plt.title("%s Loss" % self.title)
        plt.plot(self.train_loss, label='Training')
        plt.plot(self.val_loss, label='Validation')
        plt.legend()
        plt.show()

# ------------------------------------------ Data utilities ----------------------------------------


def load_word2vec():
    """ Load Word2Vec Vectors
        Return:
            wv_from_bin: All 3 million embeddings, each lengh 300
    """
    import gensim.downloader as api
    wv_from_bin = api.load("word2vec-google-news-300")
    vocab = list(wv_from_bin.key_to_index.keys())
    print(wv_from_bin.key_to_index[vocab[0]])
    print("Loaded vocab size %i" % len(vocab))
    return wv_from_bin


def create_or_load_slim_w2v(words_list, cache_w2v=True):
    """
    returns word2vec dict only for words which appear in the dataset.
    :param words_list: list of words to use for the w2v dict
    :param cache_w2v: whether to save locally the small w2v dictionary
    :return: dictionary which maps the known words to their vectors
    """
    w2v_path = "w2v_dict.pkl"
    if not os.path.exists(w2v_path):
        full_w2v = load_word2vec()
        w2v_emb_dict = {k: full_w2v[k] for k in words_list if k in full_w2v}
        if cache_w2v:
            save_pickle(w2v_emb_dict, w2v_path)
    else:
        w2v_emb_dict = load_pickle(w2v_path)
    return w2v_emb_dict


def get_w2v_average(sent, word_to_vec, embedding_dim):
    """
    This method gets a sentence and returns the average word embedding of the words consisting
    the sentence.
    :param sent: the sentence object
    :param word_to_vec: a dictionary mapping words to their vector embeddings
    :param embedding_dim: the dimension of the word embedding vectors
    :return The average embedding vector as numpy ndarray.
    """
    avg = np.zeros(embedding_dim)
    text = sent.text
    for word in text:
        try:
            avg += word_to_vec[word]
        except KeyError:
            continue
    return avg / len(text)


def get_one_hot(size, ind):
    """
    this method returns a one-hot vector of the given size, where the 1 is placed in the ind entry.
    :param size: the size of the vector
    :param ind: the entry index to turn to 1
    :return: numpy ndarray which represents the one-hot vector
    """
    vec = np.zeros(size)
    vec[ind] = 1
    return vec


def average_one_hots(sent, word_to_ind):
    """
    this method gets a sentence, and a mapping between words to indices, and returns the average
    one-hot embedding of the tokens in the sentence.
    :param sent: a sentence object.
    :param word_to_ind: a mapping between words to indices
    :return:
    """
    size = len(word_to_ind)
    avg = np.zeros(size)
    text = sent.text
    for word in text:
        avg[word_to_ind[word]] += 1
    return avg / len(text)


def get_word_to_ind(words_list):
    """
    this function gets a list of words, and returns a mapping between
    words to their index.
    :param words_list: a list of words
    :return: the dictionary mapping words to the index
    """
    indice = [i for i in range(len(words_list))]
    return dict(zip(words_list, indice))


def sentence_to_embedding(sent, word_to_vec, seq_len, embedding_dim=300):
    """
    this method gets a sentence and a word to vector mapping, and returns a list containing the
    words embeddings of the tokens in the sentence.
    :param sent: a sentence object
    :param word_to_vec: a word to vector mapping.
    :param seq_len: the fixed length for which the sentence will be mapped to.
    :param embedding_dim: the dimension of the w2v embedding
    :return: numpy ndarray of shape (seq_len, embedding_dim) with the representation of the sentence
    """
    mat = np.zeros(shape=(seq_len, embedding_dim))
    i = 0
    for word in sent.text:
        if i >= seq_len:
            break
        try:
            mat[i] = word_to_vec[word]
        except:
            pass
        finally:
            i += 1
    return mat


class OnlineDataset(Dataset):
    """
    A pytorch dataset which generates model inputs on the fly from sentences of SentimentTreeBank
    """

    def __init__(self, sent_data, sent_func, sent_func_kwargs):
        """
        :param sent_data: list of sentences from SentimentTreeBank
        :param sent_func: Function which converts a sentence to an input datapoint
        :param sent_func_kwargs: fixed keyword arguments for the state_func
        """
        self.data = sent_data
        self.sent_func = sent_func
        self.sent_func_kwargs = sent_func_kwargs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sent = self.data[idx]
        sent_emb = self.sent_func(sent, **self.sent_func_kwargs)
        sent_label = sent.sentiment_class
        return sent_emb, sent_label


class DataManager():
    """
    Utility class for handling all data management task. Can be used to get iterators for training and
    evaluation.
    """

    def __init__(self, data_type=ONEHOT_AVERAGE, use_sub_phrases=True, dataset_path="stanfordSentimentTreebank", batch_size=50,
                 embedding_dim=None):
        """
        builds the data manager used for training and evaluation.
        :param data_type: one of ONEHOT_AVERAGE, W2V_AVERAGE and W2V_SEQUENCE
        :param use_sub_phrases: if true, training data will include all sub-phrases plus the full sentences
        :param dataset_path: path to the dataset directory
        :param batch_size: number of examples per batch
        :param embedding_dim: relevant only for the W2V data types.
        """

        # load the dataset
        self.sentiment_dataset = data_loader.SentimentTreeBank(
            dataset_path, split_words=True)
        # map data splits to sentences lists
        self.sentences = {}
        if use_sub_phrases:
            self.sentences[TRAIN] = self.sentiment_dataset.get_train_set_phrases()
        else:
            self.sentences[TRAIN] = self.sentiment_dataset.get_train_set()

        self.sentences[VAL] = self.sentiment_dataset.get_validation_set()
        self.sentences[TEST] = self.sentiment_dataset.get_test_set()

        # map data splits to sentence input preperation functions
        words_list = list(self.sentiment_dataset.get_word_counts().keys())
        if data_type == ONEHOT_AVERAGE:
            self.sent_func = average_one_hots
            self.sent_func_kwargs = {
                "word_to_ind": get_word_to_ind(words_list)}
        elif data_type == W2V_SEQUENCE:
            self.sent_func = sentence_to_embedding

            self.sent_func_kwargs = {"seq_len": SEQ_LEN,
                                     "word_to_vec": create_or_load_slim_w2v(words_list),
                                     "embedding_dim": embedding_dim
                                     }
        elif data_type == W2V_AVERAGE:
            self.sent_func = get_w2v_average
            words_list = list(self.sentiment_dataset.get_word_counts().keys())
            self.sent_func_kwargs = {"word_to_vec": create_or_load_slim_w2v(words_list),
                                     "embedding_dim": embedding_dim
                                     }
        else:
            raise ValueError("invalid data_type: {}".format(data_type))
        # map data splits to torch datasets and iterators
        self.torch_datasets = {k: OnlineDataset(sentences, self.sent_func, self.sent_func_kwargs) for
                               k, sentences in self.sentences.items()}
        self.torch_iterators = {k: DataLoader(dataset, batch_size=batch_size, shuffle=k == TRAIN)
                                for k, dataset in self.torch_datasets.items()}

    def get_torch_iterator(self, data_subset=TRAIN):
        """
        :param data_subset: one of TRAIN VAL and TEST
        :return: torch batches iterator for this part of the datset
        """
        return self.torch_iterators[data_subset]

    def get_labels(self, data_subset=TRAIN):
        """
        :param data_subset: one of TRAIN VAL and TEST
        :return: numpy array with the labels of the requested part of the datset in the same order of the
        examples.
        """
        return np.array([sent.sentiment_class for sent in self.sentences[data_subset]])

    def get_input_shape(self):
        """
        :return: the shape of a single example from this dataset (only of x, ignoring y the label).
        """
        return self.torch_datasets[TRAIN][0][0].shape


# ------------------------------------ Models ----------------------------------------------------

class LSTM(nn.Module):
    """
    An LSTM for sentiment analysis with architecture as described in the exercise description.
    """

    def __init__(self, embedding_dim, hidden_dim, n_layers, dropout):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_dim
        self.n_layers = n_layers
        self.rnn = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=n_layers,
                           batch_first=True, dtype=torch.float64)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(2 * hidden_dim, 1, dtype=torch.float64)

    def forward(self, text):
        # h0 = torch.zeros(self.n_layers * 2, text.shape[0], self.hidden_size, dtype=torch.float64)
        # c0 = torch.zeros(self.n_layers * 2, text.shape[0], self.hidden_size, dtype=torch.float64)
        # out, _ = self.rnn(text, (h0, c0))
        # out = self.dropout(out)
        # out = self.linear(out[:, -1, :])
        # return out

        h0 = torch.zeros(
            self.n_layers, text.shape[0], self.hidden_size, dtype=torch.float64)
        c0 = torch.zeros(
            self.n_layers, text.shape[0], self.hidden_size, dtype=torch.float64)
        h0_r = torch.zeros(
            self.n_layers, text.shape[0], self.hidden_size, dtype=torch.float64)
        c0_r = torch.zeros(
            self.n_layers, text.shape[0], self.hidden_size, dtype=torch.float64)

        out, _ = self.rnn(text, (h0, c0))
        out_r, _ = self.rnn(torch.flip(text, [1]), (h0_r, c0_r))
        out = out[:, -1, :]
        out_r = out_r[:, -1, :]
        out = torch.cat([out, out_r], dim=1)
        out = self.dropout(out)
        out = self.linear(out)
        return out

    def predict(self, text):
        return torch.sigmoid(self.forward(text))

    def get_name(self):
        return "LSTM"


class LogLinear(nn.Module):
    """
    general class for the log-linear models for sentiment analysis.
    """

    def __init__(self, embedding_dim):
        super(LogLinear, self).__init__()
        self.layer = nn.Linear(embedding_dim, 1, dtype=torch.float64)

    def forward(self, x):
        return self.layer(x)

    def predict(self, x):
        return torch.sigmoid(self.forward(x))

    def get_name(self):
        return "LogLinear"


# ------------------------- training functions -------------


def binary_accuracy(preds, y):
    """
    This method returns tha accuracy of the predictions, relative to the labels.
    You can choose whether to use numpy arrays or tensors here.
    :param preds: a vector of predictions
    :param y: a vector of true labels
    :return: scalar value - (<number of accurate predictions> / <number of examples>)
    """
    return torch.sum(torch.round(preds) == y) / len(y)


def train_epoch(model, data_iterator, optimizer, criterion):
    """
    This method operates one epoch (pass over the whole train set) of training of the given model,
    and returns the accuracy and loss for this epoch
    :param model: the model we're currently training
    :param data_iterator: an iterator, iterating over the training data for the model.
    :param optimizer: the optimizer object for the training process.
    :param criterion: the criterion object for the training process.
    """
    epoch_loss, epoch_acc, epoch_size = 0, [], 0
    size = len(data_iterator)
    for batch, (batch_X, batch_y) in enumerate(data_iterator):
        pred = model(batch_X)
        y_shaped = batch_y.reshape(pred.shape)
        loss, acc = criterion(pred, batch_y.reshape(
            pred.shape)), binary_accuracy(torch.sigmoid(pred), y_shaped)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc.append(acc.item())
        epoch_size += len(batch_X)

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(batch_X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size*len(batch_X):>5d}]")

    return epoch_loss / epoch_size, np.average(epoch_acc)


def evaluate(model, data_iterator, criterion):
    """
    evaluate the model performance on the given data
    :param model: one of our models..
    :param data_iterator: torch data iterator for the relevant subset
    :param criterion: the loss criterion used for evaluation
    :return: tuple of (average loss over all examples, average accuracy over all examples)
    """
    epoch_loss, epoch_acc, epoch_size = 0, [], 0
    for batch_X, batch_y in data_iterator:
        pred_loss, pred_acc = model(batch_X), model.predict(batch_X)
        y_shaped = batch_y.reshape(pred_loss.shape)
        loss, acc = criterion(pred_loss, y_shaped), binary_accuracy(
            pred_acc, y_shaped)

        epoch_loss += loss.item()
        epoch_acc.append(acc.item())
        epoch_size += len(batch_X)

    return epoch_loss / epoch_size, np.average(epoch_acc)


def get_predictions_for_data(model, data_iter):
    """

    This function should iterate over all batches of examples from data_iter and return all of the models
    predictions as a numpy ndarray or torch tensor (or list if you prefer). the prediction should be in the
    same order of the examples returned by data_iter.
    :param model: one of the models you implemented in the exercise
    :param data_iter: torch iterator as given by the DataManager
    :return:
    """
    tensors = []
    for batch_X, batch_y in data_iter:
        tensors.append(model.predict(batch_X))
    return torch.cat(tensors)


def test_model(model, data_manager, title):
    dataset = data_loader.SentimentTreeBank()
    test_iter, test_labels = data_manager.get_torch_iterator(data_subset=TEST), \
        data_manager.get_labels(data_subset=TEST)
    test_sent = dataset.get_test_set()
    negated_polarity_iter = data_loader.get_negated_polarity_examples(
        test_sent)
    rare_words_iter = data_loader.get_rare_words_examples(test_sent, dataset)
    test_pred = get_predictions_for_data(model=model, data_iter=test_iter)
    test_labels = torch.Tensor(test_labels).reshape(test_pred.shape)
    print("TEST module %s:" % title)
    print("All-Test accuracy: ", binary_accuracy(test_pred, test_labels))
    print("All-Test loss: ", evaluate(model=model,
          data_iterator=test_iter, criterion=nn.BCEWithLogitsLoss())[0])
    print("Negated polarity test accuracy: ", binary_accuracy(
        test_pred[negated_polarity_iter], test_labels[negated_polarity_iter]))
    print("Rare words test accuracy: ", binary_accuracy(
        test_pred[rare_words_iter], test_labels[rare_words_iter]))


def train_model(model, data_manager, n_epochs, lr, weight_decay=0., analysis=PerformenceAnalysis(None)):
    """
    Runs the full training procedure for the given model. The optimization should be done using the Adam
    optimizer with all parameters but learning rate and weight decay set to default.
    :param model: module of one of the models implemented in the exercise
    :param data_manager: the DataManager object
    :param n_epochs: number of times to go over the whole training set
    :param lr: learning rate to be used for optimization
    :param weight_decay: parameter for l2 regularization
    """
    optimizer = optim.Adam(model.parameters(), lr=lr,
                           weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss().to(device=get_available_device())
    for i in range(n_epochs):
        start = time.time()
        print(f"Epoch {i + 1}\n-------------------------------")
        train_loss, train_acc = train_epoch(model=model, data_iterator=data_manager.get_torch_iterator(),
                                            optimizer=optimizer, criterion=criterion)
        analysis.append_train(train_loss, train_acc)
        val_loss, val_acc = evaluate(model=model, data_iterator=data_manager.get_torch_iterator(data_subset=VAL),
                                     criterion=criterion)
        analysis.append_validation(val_loss, val_acc)
        print("Total time: %.3f\n" % (time.time() - start))

    return analysis


def train_log_linear_with_one_hot():
    """
    Here comes your code for training and evaluation of the log linear model with one hot representation.
    """
    dm = DataManager(batch_size=64)
    model = LogLinear(dm.get_input_shape()[0]).to(
        device=get_available_device())
    title = "Log Linear with one hot vector"
    analysis = PerformenceAnalysis(title)

    print("=========================== START %s =============================" % title)
    train_model(model=model, data_manager=dm, n_epochs=20,
                lr=0.01, weight_decay=0.001, analysis=analysis)
    save_pickle(model, "%s.pkl" % title.replace(" ", "_"))
    analysis.plot_loss()
    analysis.plot_accuracy()
    test_model(model=model, data_manager=dm, title=title)
    print("=========================== END %s =============================" % title)


def train_log_linear_with_w2v():
    """
    Here comes your code for training and evaluation of the log linear model with word embeddings
    representation.
    """
    dm = DataManager(data_type=W2V_AVERAGE, batch_size=64,
                     embedding_dim=W2V_EMBEDDING_DIM)
    model = LogLinear(W2V_EMBEDDING_DIM).to(device=get_available_device())
    title = "Log Linear with W2V"
    analysis = PerformenceAnalysis(title)

    print("=========================== START %s =============================" % title)
    train_model(model=model, data_manager=dm, n_epochs=20,
                lr=0.01, weight_decay=0.001, analysis=analysis)
    save_pickle(model, "%s.pkl" % title.replace(" ", "_"))
    analysis.plot_loss()
    analysis.plot_accuracy()
    test_model(model=model, data_manager=dm, title=title)
    print("=========================== END %s =============================" % title)


def train_lstm_with_w2v():
    """
    Here comes your code for training and evaluation of the LSTM model.
    """
    dm = DataManager(data_type=W2V_SEQUENCE, batch_size=64,
                     embedding_dim=W2V_EMBEDDING_DIM)
    model = LSTM(embedding_dim=W2V_EMBEDDING_DIM, hidden_dim=100,
                 n_layers=1, dropout=0.5).to(device=get_available_device())
    title = "LSTM with W2V"
    analysis = PerformenceAnalysis(title)

    print("=========================== START %s =============================" % title)
    train_model(model=model, data_manager=dm, n_epochs=4,
                lr=0.001, weight_decay=0.0001, analysis=analysis)
    save_pickle(model, "%s.pkl" % title.replace(" ", "_"))
    analysis.plot_loss()
    analysis.plot_accuracy()
    test_model(model=model, data_manager=dm, title=title)
    print("=========================== END %s =============================" % title)


if __name__ == '__main__':
    # train_log_linear_with_one_hot()
    # train_log_linear_with_w2v()
    train_lstm_with_w2v()
