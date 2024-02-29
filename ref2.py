#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 08:29:35 2024

@author: danieldabbah
"""
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
import matplotlib.pyplot as plt
from data_loader import *

EPOCH_STR = 'Epoch'

ACCURACY = "accuracy"

# ------------------------------------------- Constants ----------------------------------------

SEQ_LEN = 52
W2V_EMBEDDING_DIM = 300

ONEHOT_AVERAGE = "onehot_average"
W2V_AVERAGE = "w2v_average"
W2V_SEQUENCE = "w2v_sequence"

TRAIN = "train"
VAL = "val"
TEST = "test"
EPOCH = 20
LR = 0.01
BATCH_SIZE = 64
WEIGHT_DECAY = 0.001
LOSS = "loss"

LR_LSTM = 0.001
WEIGHT_DECAY_LSTM = 0.0001
DROPOUT_LSTM = 0.5
EPOCH_LSTM = 4
POLAR = "polar"
RARE = "rare"
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


# ------------------------------------------ Data utilities ----------------------------------------

def load_word2vec():
    """ Load Word2Vec Vectors
        Return:
            wv_from_bin: All 3 million embeddings, each lengh 300
    """
    import gensim.downloader as api
    wv_from_bin = api.load("word2vec-google-news-300")
    vocab = list(wv_from_bin.vocab.keys())
    print(wv_from_bin.vocab[vocab[0]])
    print("Loaded vocab size %i" % len(vocab))
    return wv_from_bin


def create_or_load_slim_w2v(words_list, cache_w2v=False):
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

    sentence = sent.text
    vec_sum = np.zeros(embedding_dim)

    for word in sentence:
        if word in word_to_vec.keys():
            vec_sum += word_to_vec[word]
    return vec_sum / len(sentence)


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
    sentences = sent.text
    size = len(word_to_ind.keys())
    average_vec = np.zeros(size, dtype='float32')
    for word in sentences:
        average_vec += get_one_hot(size, word_to_ind[word])
    return average_vec / len(sentences)


def get_word_to_ind(words_list):
    """
    this function gets a list of words, and returns a mapping between
    words to their index.
    :param words_list: a list of words
    :return: the dictionary mapping words to the index
    """
    word_to_ind = dict()
    for i, word in enumerate(words_list):
        word_to_ind[word] = i
    return word_to_ind


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
    sentence_embedding = []
    sentence = sent.text
    # deals with 2 cases >,= at the same time
    # We will map each of its seq_len=52 words to its embedding
    # We will map each of their first 52 words to their embeddings and ignore all other words.
    # If word not in dict we add the zero embedding to list

    if len(sentence) >= seq_len:
        for i in range(seq_len):
            word = sentence[i]
            embedding = np.zeros(embedding_dim)
            if word in word_to_vec.keys():
                embedding = word_to_vec[word]
            sentence_embedding.append(embedding)

    # These embeddings should be followed by 52 − 𝑥 more 0-vectors, to make sure the sentence is of length 52
    elif len(sentence) < seq_len:
        for word in sentence:
            embedding = np.zeros(embedding_dim)
            if word in word_to_vec.keys():
                embedding = word_to_vec[word]
            sentence_embedding.append(embedding)
        for i in range(seq_len - len(sentence)):
            sentence_embedding.append(np.zeros(embedding_dim))

    return np.array(sentence_embedding)


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

    def __init__(self, data_type=ONEHOT_AVERAGE, use_sub_phrases=True, dataset_path="stanfordSentimentTreebank",
                 batch_size=50,
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
        ########################### USING THE RARE WORDS Q.9 #######################################
        # self.sentences["rare"] = get_rare_or_polar(self, "rare")
        # self.sentences["polar"] = get_rare_or_polar(self, "polar")
        # map data splits to sentence input preperation functions
        words_list = list(self.sentiment_dataset.get_word_counts().keys())
        if data_type == ONEHOT_AVERAGE:
            self.sent_func = average_one_hots
            self.sent_func_kwargs = {
                "word_to_ind": get_word_to_ind(words_list)}
        elif data_type == W2V_SEQUENCE:
            self.sent_func = sentence_to_embedding

            self.sent_func_kwargs = {"seq_len": SEQ_LEN,
                                     "word_to_vec": create_or_load_slim_w2v(words_list, True),
                                     "embedding_dim": embedding_dim
                                     }
        elif data_type == W2V_AVERAGE:
            self.sent_func = get_w2v_average
            words_list = list(self.sentiment_dataset.get_word_counts().keys())
            self.sent_func_kwargs = {"word_to_vec": create_or_load_slim_w2v(words_list, True),
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
        :return: torch batches iterator for this part of the dataset
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
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=n_layers, bidirectional=True,
                            batch_first=True, dropout=dropout)
        self.linear = nn.Linear(in_features=2 * hidden_dim, out_features=1)
        self.name = "LSTM"

    def forward(self, text):
        # l2 = self.lstm(text)
        out, (h_n, c_n) = self.lstm(text)
        lin = self.linear(out)
        # prob for sentence by taking the product of the words prob (output form softmax)
        ans = torch.prod(lin, 1)
        return ans

    def predict(self, text):
        # in our case forward = predict
        return torch.sigmoid(self.forward(text))


class LogLinear(nn.Module):
    """
    general class for the log-linear models for sentiment analysis.
    """

    def __init__(self, embedding_dim, name):
        super().__init__()
        self.linear = nn.Linear(embedding_dim, 1)
        self.name = name

    def forward(self, x):
        # uses forward of linear #TODO maybe remove forward
        return self.linear.forward(x)

    def predict(self, x):
        return torch.sigmoid(self.linear(x))  # in our case forward = predict


# ------------------------- training functions -------------


def binary_accuracy(preds, y):
    """
    INPUTE BINARY
    This method returns tha accuracy of the predictions, relative to the labels.
    You can choose whether to use numpy arrays or tensors here.
    :param preds: a vector of predictions
    :param y: a vector of true labels
    :return: scalar value - (<number of accurate predictions> / <number of examples>)
    """
    # binary_pred_labels = torch.round(preds)
    # return (np.array(binary_pred_labels == y)) / len(y)
    accuracy = 0
    for ind, y_pred in enumerate(preds):
        y_pred = torch.round(torch.sigmoid(y_pred))
        if (y[ind] == y_pred):
            accuracy += 1
    return accuracy / len(y)


def train_epoch(model, data_iterator, optimizer, criterion):
    """
    This method operates one epoch (pass over the whole train set) of training of the given model,
    and returns the accuracy and loss for this epoch
    :param model: the model we're currently training
    :param data_iterator: an iterator, iterating over the training data for the model.
    :param optimizer: the optimizer object for the training process.
    :param criterion: the criterion object for the training process.
    """
    # TODO: we train using forward and inference using eval
    # TODO: THEY USE forward only for train and calculate accu and error using eval
    model.train()
    loss = 0
    accuracy = 0

    for batch in data_iterator:
        # TODO: used sentiment val here
        data, true_labels = batch[0].to(torch.float32), batch[1]
        # Forward pass

        pred_labels = model.forward(data).squeeze()

        # Calculating loss
        # TODO: round up predic_vals
        current_loss = criterion(pred_labels, true_labels)

        # Backward pass
        optimizer.zero_grad()  # zero gradients
        current_loss.backward()  # calculate new grad
        optimizer.step()  # step in direction of grad - update weights

        loss += current_loss.item()  # extracts the loss’s value as a Python float
        accuracy += binary_accuracy(pred_labels, true_labels)

    loss = loss / len(data_iterator)  # number of batches
    accuracy = accuracy / len(data_iterator)

    return loss, accuracy


def evaluate(model, data_iterator, criterion):  # validation/test
    """
    evaluate the model performance on the given data
    :param model: one of our models..
    :param data_iterator: torch data iterator for the relevant subset
    :param criterion: the loss criterion used for evaluation
    :return: tuple of (average loss over all examples, average accuracy over all examples)
    """
    model.eval()
    loss = 0
    accuracy = 0
    # we don't want our model parameters to change while we're evaluating it.
    with torch.no_grad():
        for batch in (data_iterator):
            data, true_labels = batch[0].to(torch.float32), batch[1]
            pred_labels = model.forward(data).squeeze()

            # Calculating loss
            current_loss = criterion(pred_labels, true_labels)
            loss += current_loss.item()

            accuracy += binary_accuracy(pred_labels, true_labels)
    loss = loss / len(data_iterator)  # number of batches
    accuracy = accuracy / len(data_iterator)
    return loss, accuracy


def get_predictions_for_data(model, data_iter):
    """

    This function should iterate over all batches of examples from data_iter and return all of the models
    predictions as a numpy ndarray or torch tensor (or list if you prefer). the prediction should be in the
    same order of the examples returned by data_iter.
    :param model: one of the models you implemented in the exercise
    :param data_iter: torch iterator as given by the DataManager
    :return:
    """
    preds = []
    for batch in (data_iter):
        data = batch[0]
        pred_labels = model.predict(data)
        preds.append(pred_labels)
    return torch.Tensor(preds).to(torch.float32)


def train_model(model, data_manager, n_epochs, lr, weight_decay=0.):
    """
    Runs the full training procedure for the given model. The optimization should be done using the Adam
    optimizer with all parameters but learning rate and weight decay set to default.
    :param model: module of one of the models implemented in the exercise
    :param data_manager: the DataManager object
    :param n_epochs: number of times to go over the whole training set
    :param lr: learning rate to be used for optimization
    :param weight_decay: parameter for l2 regularization
    """
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay)
    train_losses = []
    train_accuracies = []
    validation_losses = []
    validation_accuracies = []

    for epoch in range(n_epochs):
        # printing to follow training
        print('epoch number :' + str(epoch))

        train_sentences = data_manager.get_torch_iterator(TRAIN)
        validation_sentences = data_manager.get_torch_iterator(VAL)

        train_epoch(model, data_manager.get_torch_iterator(),
                    optimizer, criterion)  # train model

        train_loss, train_accuracy = evaluate(model, train_sentences,
                                              criterion)
        validation_loss, validation_accuracy = evaluate(model, validation_sentences,
                                                        criterion)
        # save losses and accuracy
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        validation_losses.append(validation_loss)
        validation_accuracies.append(validation_accuracy)

    save_model(model, model.name, n_epochs, optimizer)
    return train_losses, train_accuracies, validation_losses, validation_accuracies


def train_log_linear_with_one_hot():
    """
    Here comes your code for training and evaluation of the log linear model with one hot representation.
    """

    data_manager = DataManager(ONEHOT_AVERAGE, batch_size=BATCH_SIZE)
    embedding_dim = len(
        data_manager.sentiment_dataset.get_word_counts().keys())  # word length
    model = LogLinear(embedding_dim, "oneHot")
    train_losses, train_accuracies, validation_losses, validation_accuracies = train_model(model, data_manager, EPOCH,
                                                                                           LR, WEIGHT_DECAY)
    plot_loss_or_accuracy(train_losses, validation_losses, LOSS)
    plot_loss_or_accuracy(train_accuracies, validation_accuracies, ACCURACY)


def train_log_linear_with_w2v():
    """
    Here comes your code for training and evaluation of the log linear model with word embeddings
    representation.
    """
    data_manager = DataManager(
        W2V_AVERAGE, batch_size=BATCH_SIZE, embedding_dim=W2V_EMBEDDING_DIM)
    model = LogLinear(W2V_EMBEDDING_DIM, "w2v")
    train_losses, train_accuracies, validation_losses, validation_accuracies = train_model(model, data_manager, EPOCH,
                                                                                           LR, WEIGHT_DECAY)
    plot_loss_or_accuracy(train_losses, validation_losses, LOSS)
    plot_loss_or_accuracy(train_accuracies, validation_accuracies, ACCURACY)


def train_lstm_with_w2v():
    """
    Here comes your code for training and evaluation of the LSTM model.
    """

    data_manager = DataManager(
        W2V_SEQUENCE, batch_size=BATCH_SIZE, embedding_dim=W2V_EMBEDDING_DIM)
    model = LSTM(embedding_dim=W2V_EMBEDDING_DIM, hidden_dim=100,
                 n_layers=2, dropout=DROPOUT_LSTM)
    train_losses, train_accuracies, validation_losses, validation_accuracies = train_model(model, data_manager, EPOCH_LSTM,
                                                                                           LR_LSTM, WEIGHT_DECAY_LSTM)
    plot_loss_or_accuracy(train_losses, validation_losses, LOSS)
    plot_loss_or_accuracy(train_accuracies, validation_accuracies, ACCURACY)


# ------------------------------------------- Helpers ----------------------------------------


def plot_loss_or_accuracy(train, test, name):
    # with lstm we run with different color
    x = np.arange(1, len(train) + 1)
    y_train = np.array(train)
    y_test = np.array(test)
    plt.plot(x, y_train, 'b', label='Train ' + name)
    plt.plot(x, y_test, 'y', label='Validation ' + name)
    plt.xlabel(EPOCH_STR)
    plt.ylabel(name)
    plt.legend()
    plt.title("Train and Validation " + str(name) +
              " with " + str(EPOCH) + " epochs")
    plt.show()


if __name__ == '__main__':
    train_log_linear_with_one_hot()
    # train_log_linear_with_w2v()
    # train_lstm_with_w2v()
