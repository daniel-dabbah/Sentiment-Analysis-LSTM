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
    res = torch.zeros(embedding_dim)
    # TODO: check how to make this function faster. maybe use 2d array
    for word in sent.text:

        if word in word_to_vec.keys():

            res += word_to_vec[word]

    return res/len(sent.text)


def get_one_hot(size, ind):
    """
    this method returns a one-hot vector of the given size, where the 1 is placed in the ind entry.
    :param size: the size of the vector
    :param ind: the entry index to turn to 1
    :return: numpy ndarray which represents the one-hot vector
    """
    result = torch.zeros(size)
    result[ind] = 1

    return result


def average_one_hots(sent, word_to_ind):
    """
    this method gets a sentence, and a mapping between words to indices, and returns the average
    one-hot embedding of the tokens in the sentence.
    :param sent: a sentence object.
    :param word_to_ind: a mapping between words to indices
    :return:
    """
    size = len(word_to_ind)

    result = torch.zeros(size)
    for word in sent.text:
        result[word_to_ind[word]] += 1

    return result / len(sent.text)


def get_word_to_ind(words_list):
    """
    this function gets a list of words, and returns a mapping between
    words to their index.
    :param words_list: a list of words
    :return: the dictionary mapping words to the index
    """

    return {word: i for i, word in enumerate(words_list)}


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
    sent_len = len(sent.text)

    def get_embedding(word):
        return word_to_vec[word] if word in word_to_vec.keys() else np.zeros(embedding_dim)

    return torch.tensor(np.stack([get_embedding(sent.text[i]) if i < sent_len
                                  else np.zeros(embedding_dim)
                                  for i in range(seq_len)]))


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

    def get_sents(self, data_subset=TRAIN):
        return self.sentences[data_subset]


# ------------------------------------ Models ----------------------------------------------------

class LSTM(nn.Module):
    """
    An LSTM for sentiment analysis with architecture as described in the exercise description.
    """

    def __init__(self, embedding_dim, hidden_dim, n_layers, dropout):

        super().__init__()

        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        self.rnn = nn.LSTM(embedding_dim, hidden_dim, n_layers,
                           batch_first=True, dtype=torch.float64)
        self.rnn2 = nn.LSTM(embedding_dim, hidden_dim, n_layers, bidirectional=True,
                            batch_first=True, dtype=torch.float64)
        self.linear = nn.Linear(hidden_dim * 2, 1, dtype=torch.float64)
        self.sigmoid = torch.nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):

        h0 = torch.zeros(
            self.n_layers, text.shape[0], self.hidden_dim, dtype=torch.float64)
        c0 = torch.zeros(
            self.n_layers, text.shape[0], self.hidden_dim, dtype=torch.float64)
        h0_r = torch.zeros(
            self.n_layers, text.shape[0], self.hidden_dim, dtype=torch.float64)
        c0_r = torch.zeros(
            self.n_layers, text.shape[0], self.hidden_dim, dtype=torch.float64)

        out, _ = self.rnn(text, (h0, c0))
        out_r, _ = self.rnn(torch.flip(text, [1]), (h0_r, c0_r))

        out = torch.cat([out[:, -1, :], out_r[:, -1, :]], dim=1)

        # _, (h, _) = self.rnn2(text)
        # # print(h[1].size())
        # out = torch.cat((h[0], h[1]), dim=-1)

        return self.linear(self.dropout(out))

    def predict(self, text):
        with torch.no_grad():
            return self.sigmoid(self(text))


class LogLinear(nn.Module):
    """
    general class for the log-linear models for sentiment analysis.
    """

    def __init__(self, embedding_dim):
        super().__init__()

        self.linear = nn.Linear(embedding_dim, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        return self.linear(x)

    def predict(self, x):
        with torch.no_grad():
            return self.sigmoid(self(x))


# ------------------------- training functions -------------


def binary_accuracy(preds, y):
    """
    This method returns tha accuracy of the predictions, relative to the labels.
    You can choose whether to use numpy arrays or tensors here.
    the function assumes the preds are after the sigmoid function and
    therefore between 0-1.
    :param preds: a vector of predictions
    :param y: a vector of true labels
    :return: scalar value - (<number of accurate predictions> / <number of examples>)
    """

    return (torch.sum(torch.round(preds) == y) / len(y)) * 100


def train_epoch(model, data_iterator, optimizer, criterion):
    """
    This method operates one epoch (pass over the whole train set) of training of the given model,
    and returns the accuracy and loss for this epoch
    :param model: the model we're currently training
    :param data_iterator: an iterator, iterating over the training data for the model.
    :param optimizer: the optimizer object for the training process.
    :param criterion: the criterion object for the training process.
    """

    for xb, yb in data_iterator:

        # calculate the gradients:
        preds = model(xb).squeeze()
        loss = criterion(preds, yb)
        optimizer.zero_grad()
        loss.backward()

        # update the weights:

        optimizer.step()
        optimizer.zero_grad()


def evaluate(model, data_iterator, criterion):
    """
    evaluate the model performance on the given data
    :param model: one of our models..
    :param data_iterator: torch data iterator for the relevant subset
    :param criterion: the loss criterion used for evaluation
    :return: tuple of (average loss over all examples, average accuracy over all examples)
    """

    with torch.no_grad():

        losses, accs = zip(*[(criterion(model(xb).squeeze(), yb).item(),
                              binary_accuracy(model.predict(xb).squeeze(), yb))
                             for xb, yb in data_iterator])

        return round(torch.tensor(losses).mean().item(), 10), \
            round(torch.tensor(accs).mean().item(), 10)


def get_predictions_for_data(model, data_iter):
    """

    This function should iterate over all batches of examples from data_iter and return all of the models
    predictions as a numpy ndarray or torch tensor (or list if you prefer). the prediction should be in the
    same order of the examples returned by data_iter.
    :param model: one of the models you implemented in the exercise
    :param data_iter: torch iterator as given by the DataManager
    :return:
    """

    with torch.no_grad():
        return torch.cat([model.predict(xb).squeeze() for xb, _ in data_iter])


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

    optimizer = optim.Adam(model.parameters(), lr=lr,
                           weight_decay=weight_decay)

    criterion = nn.BCEWithLogitsLoss()

    train_losses, train_accs = torch.Tensor(n_epochs), torch.Tensor(n_epochs)
    val_losses, val_accs = torch.Tensor(n_epochs), torch.Tensor(n_epochs)

    for i in range(n_epochs):

        train_epoch(model, data_manager.get_torch_iterator(
            data_subset=TRAIN), optimizer, criterion)

        train_losses[i], train_accs[i] = evaluate(model, data_manager.get_torch_iterator(
            data_subset=TRAIN), criterion)

        val_losses[i], val_accs[i] = evaluate(model, data_manager.get_torch_iterator(
            data_subset=VAL), criterion)

    plot_results(train_losses, train_accs, val_losses, val_accs)
    return


def train_log_linear_with_one_hot(batch_size, n_epochs, lr, weight_decay=0.):
    """
    Here comes your code for training and evaluation of the log linear model with one hot representation.
    """
    # Train the model:

    model_path = "log_linear_one_hot_model.pkl"
    model_name = "Simple Log-Linear"
    dm = DataManager(data_type=ONEHOT_AVERAGE, batch_size=batch_size)
    model = LogLinear(dm.get_input_shape()[0])
    train_model(model=model, data_manager=dm, n_epochs=n_epochs,
                lr=lr, weight_decay=weight_decay)

    # Save the model:
    save_pickle(model, model_path)

    # print results:
    print_results(model, dm, model_name)

    return model


def train_log_linear_with_w2v(batch_size, n_epochs, lr, weight_decay=0.):
    """
    Here comes your code for training and evaluation of the log linear model with word embeddings
    representation.
    """

    model_path = "log_linear_with_w2v_model.pkl"
    model_name = "Word2Vec Log-Linear"
    dm = DataManager(data_type=W2V_AVERAGE, batch_size=batch_size,
                     embedding_dim=W2V_EMBEDDING_DIM)

    model = LogLinear(dm.get_input_shape()[0])
    train_model(model=model, data_manager=dm, n_epochs=n_epochs,
                lr=lr, weight_decay=weight_decay)

    # Save the model:
    save_pickle(model, model_path)

    print_results(model, dm, model_name)

    return model


def train_lstm_with_w2v(batch_size, n_epochs, lr, weight_decay=0.):
    """
    Here comes your code for training and evaluation of the LSTM model.
    """

    model_path = "LSTM_model.pkl"
    model_name = "LSTM"

    dm = DataManager(data_type=W2V_SEQUENCE,
                     batch_size=batch_size, embedding_dim=W2V_EMBEDDING_DIM)

    model = LSTM(embedding_dim=W2V_EMBEDDING_DIM,
                 hidden_dim=100, n_layers=1, dropout=0.5)

    train_model(model=model, data_manager=dm, n_epochs=n_epochs,
                lr=lr, weight_decay=weight_decay)

    save_pickle(model, model_path)

    print_results(model, dm, model_name)

    return model


def plot_loss_acc(train_res, val_res, res_unit, ax=None):

    if ax is None:
        ax = plt.subplots()[1]
    x_values = list(range(1, len(train_res)+1))

    ax.plot(x_values, train_res,
            label=f" Train {res_unit}")
    ax.plot(x_values, val_res,
            label=f" Validation {res_unit}")

    ax.set_xlabel('Epochs')
    ax.set_ylabel('Values')
    ax.set_title(f"{res_unit} Results")

    ax.legend()


def plot_results(train_losses, train_accs, val_losses, val_accs):
    # Plot Train and Validation results:
    res_units = ("Losses", "Accuracies")
    train_results = (train_losses, train_accs)
    val_results = (val_losses, val_accs)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    for train_res, val_res, res_unit, ax in zip(train_results, val_results, res_units, axs):
        plot_loss_acc(train_res, val_res, res_unit, ax)


def print_results(model, dm, model_name):
    dataset = data_loader.SentimentTreeBank()

    # compute test predictions:
    logits = torch.cat([model(xb).squeeze()
                       for xb, _ in dm.get_torch_iterator(data_subset=TEST)])
    test_predicts = get_predictions_for_data(
        model, dm.get_torch_iterator(data_subset=TEST))
    # TODO: check if use get_labels function
    y_true = torch.cat(
        [yb for _, yb in dm.get_torch_iterator(data_subset=TEST)])

    # compute overall test results:
    test_loss, test_acc = round(nn.BCEWithLogitsLoss()(
        logits, y_true).item(), 6), round(binary_accuracy(test_predicts, y_true).item(), 6)

    print(f"Results for the {model_name} Model: ")
    print(f"Test Set loss is: {test_loss}")
    print(f"Test Set accuracy is: {test_acc}%")

    # compute special subsets accuracy:
    test_sents_objects = dm.get_sents(data_subset=TEST)

    polar_indexes = data_loader.get_negated_polarity_examples(
        test_sents_objects)
    rare_words_indexes = data_loader.get_rare_words_examples(
        test_sents_objects, dataset)

    polar_test_acc = round(binary_accuracy(
        test_predicts[polar_indexes], y_true[[polar_indexes]]).item(), 6)
    rare_word_acc = round(binary_accuracy(
        test_predicts[rare_words_indexes], y_true[[rare_words_indexes]]).item(), 6)

    print("Accuracies for the special subsets:")
    print(f"Negated polarity examples accuracy is: {polar_test_acc}%")
    print(f"Rare words examples accuracy is: {rare_word_acc}%\n")


if __name__ == '__main__':

    # train_log_linear_with_one_hot(
    #     batch_size=64, n_epochs=20, lr=0.01, weight_decay=0.001)

    # train_log_linear_with_w2v(
    #     batch_size=64, n_epochs=20, lr=0.01, weight_decay=0.001)

    train_lstm_with_w2v(batch_size=64, n_epochs=1,
                        lr=.001, weight_decay=0.0001)

    """
    TODO: 
        1. add docstring
        2. deal with magic numbers.
        3. improve lstm class
        4. check other to do
    """
