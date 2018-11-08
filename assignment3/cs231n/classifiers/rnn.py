#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 @Time    : 2018/11/6 16:26
@Author  : LI Zhe
"""
import numpy as np
# from layer_basic import *
# from layer_rnn import *
from cs231n.layer_basic import *
from cs231n.layer_rnn import *

class CaptioningRNN(object):
    def __init__(self, word_to_idx, input_dim=512,
                 wordvec_dim=128, hidden_dim=128,
                 cell_type='rnn', dtype=np.float32):
        """

        :param word_to_idx: A dictionary giving the vocabulary. It contains V entries,
                            and maps each string to a unique integer in the range{0, V)
        :param input_dim:  D dimension
        :param wordvec_dim: W dimension
        :param hidden_dim: H dimension
        :param cell_type:
        :param dtype:
        """
        if cell_type not in {'rnn', 'lstm'}:
            raise ValueError('invalid cell_type "%s"' % cell_type)

        self.cell_type = cell_type
        self.dtype = dtype
        self.word_to_idx = word_to_idx
        self.idx_to_word = {i: w for w,i in word_to_idx.items()}
        self.params ={}

        vocab_size = len(word_to_idx)

        self._null = word_to_idx['<NULL>']
        self._start = word_to_idx.get('<START>', None)
        self._end = word_to_idx.get('<END>', None)

        # Initialize word vectors
        self.params['W_embed'] = np.random.randn(vocab_size, wordvec_dim)
        self.params['W_embed'] /= 100

        # Initialize CNN -> hidden state projection parameters
        self.params['W_proj'] = np.random.randn(input_dim, hidden_dim)
        self.params['W_proj'] /= np.sqrt(input_dim)
        self.params['b_proj'] = np.zeros(hidden_dim)

        # Initialize parameters for RNN
        dim_mul = {'rnn':1, 'lstm':4}[cell_type]
        self.params['Wx'] = np.random.randn(wordvec_dim, dim_mul * hidden_dim)
        self.params['Wx'] /= np.sqrt(wordvec_dim)
        self.params['Wh'] = np.random.randn(hidden_dim, dim_mul * hidden_dim)
        self.params['Wh'] /= np.sqrt(hidden_dim)
        self.params['b'] = np.random.randn(dim_mul * hidden_dim)

        # Initialize output to vocab weights
        self.params['W_vocab'] = np.random.randn(hidden_dim, vocab_size)
        self.params['W_vocab'] /= np.sqrt(hidden_dim)
        self.params['b_vocab'] = np.zeros(vocab_size)

        # Cast parameters to correct dtype
        for k, v in self.params.items():
            self.params[k] = v.astype(self.dtype)

    def loss(self, features, captions):
        """

        :param self:
        :param features: (N, D) input image features
        :param captions: (N, T) (0 <= y[i, t] <V )
        :return:
            - loss:
            - grads:
        """
        # the first element of captions_in is <START>
        # the first element of captions_out is the first word
        captions_in = captions[:, :-1]
        captions_out = captions[:, 1:]

        mask = (captions_out != self._null)

        # the affine transform from image features to initial hidden state
        W_proj, b_proj = self.params['W_proj'], self.params['b_proj']

        # word embedding matrix
        W_embed = self.params['W_embed']

        # input-to-hidden, hidden-to-hidden, biases for RNN
        Wx, Wh, b = self.params['Wx'], self.params['Wh'], self.params['b']

        # hidden-to-vocab
        W_vocab, b_vocab = self.params['W_vocab'], self.params['b_vocab']

        loss, grads = 0.0, {}

        # (1) Use an affine transformation to compute the initial hidden state     #
        #     from the image features. This should produce an array of shape (N, H)#
        affine_out, affine_cache = affine_forward(features, W_proj, b_proj)
        # (2) Use a word embedding layer to transform the words in captions_in     #
        #     from indices to vectors, giving an array of shape (N, T, W).         #
        word_embedding_out, word_embedding_cache = word_embedding_forward(captions_in, W_embed)
        # (3) Use either a vanilla RNN or LSTM (depending on self.cell_type) to    #
        #     process the sequence of input word vectors and produce hidden state  #
        #     vectors for all timesteps, producing an array of shape (N, T, H).    #
        if self.cell_type == 'rnn':
            rnn_or_lstm_out, rnn_cache = rnn_forward(word_embedding_out, affine_out, Wx, Wh, b)
        elif self.cell_type == 'lstm':
            rnn_or_lstm_out, lstm_cache = lstm_forward(word_embedding_out, affine_out, Wx, Wh, b)
        else:
            raise ValueError('Invalid cell_type "%s" ' % self.cell_type)
        # (4) Use a (temporal) affine transformation to compute scores over the    #
        #     vocabulary at every timestep using the hidden states, giving an      #
        #     array of shape (N, T, V).                                            #
        temporal_affine_out, temporal_affine_cache = temporal_affine_forward(rnn_or_lstm_out, W_vocab, b_vocab)
        # (5) Use (temporal) softmax to compute loss using captions_out, ignoring  #
        #     the points where the output word is <NULL> using the mask above.     #
        loss, dtemporal_affine_out = temporal_softmax_loss(temporal_affine_out, captions_out, mask)

        # (4) backward #
        drnn_or_lstm_out, grads['W_vocab'], grads['b_vocab'] = temporal_affine_backward(dtemporal_affine_out, temporal_affine_cache)

        # (3) backward #
        if self.cell_type == 'rnn':
            dword_embedding_out, daffine_out, grads['Wx'], grads['Wh'], grads['b'] = rnn_backward(drnn_or_lstm_out, rnn_cache)
        elif self.cell_type == 'lstm':
            dword_embedding_out, daffine_out, grads['Wx'], grads['Wh'], grads['b'] = lstm_backward(drnn_or_lstm_out, lstm_cache)
        else:
            raise ValueError('Invalid cell_type "%s" ' % self.cell_type)

        # (2) backward #
        grads['W_embed'] = word_embedding_backward(dword_embedding_out, word_embedding_cache)

        # (1) backward #
        dfeatures, grads['W_proj'], grads['b_proj'] = affine_backward(daffine_out, affine_cache)

        return loss, grads

    def sample(self, features, max_length=30):
        """
        run a test_time forward pass for the model, sampling captions for input feature vectors
        :param self:
        :param features: (N, D)
        :param max_length: Maximum length of T of generates captions
        :return:
            - captions: (N, max_length)
        """
        N = features.shape[0]
        captions = self._null * np.ones((N, max_length), dtype=np.int32)

        W_proj, b_proj = self.params['W_proj'], self.params['b_proj']
        W_embed = self.params['W_embed']
        Wx, Wh, b = self.params['Wx'], self.params['Wh'], self.params['b']
        W_vocab, b_vocab = self.params['W_vocab'], self.params['b_vocab']

        N, D = features.shape
        affine_out, affine_cache = affine_forward(features, W_proj, b_proj)
        prev_word_idx = [self._start] * N
        prev_h = affine_out
        prev_c = np.zeros(prev_h.shape)
        captions[:, 0] = self._start

        for i in range(1, max_length):
            # (1) Embed the previous word using the learned word embeddings           #
            prev_word_embed = W_embed[prev_word_idx]
            # (2) Make an RNN step using the previous hidden state and the embedded   #
            #     current word to get the next hidden state.                          #
            if self.cell_type == 'rnn':
                next_h, rnn_step_cache = rnn_step_forward(prev_word_embed, prev_h, Wx, Wh, b)
            elif self.cell_type == 'lstm':
                next_h, next_c, lstm_step_cache = lstm_step_forward(prev_word_embed, prev_h, prev_c, Wx, Wh, b)
                prev_c = next_c
            else:
                raise ValueError('Invalid cell_type "%s"' % self.cell_type)
            # (3) Apply the learned affine transformation to the next hidden state to #
            #     get scores for all words in the vocabulary                          #
            vocab_affine_out, vocab_affine_out_cache = affine_forward(next_h, W_vocab, b_vocab)
            # (4) Select the word with the highest score as the next word, writing it #
            #     to the appropriate slot in the captions variable                    #
            captions[:, i] = list(np.argmax(vocab_affine_out, axis=1))
            prev_word_idx = captions[:, i]
            prev_h = next_h

        return captions
