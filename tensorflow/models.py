from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf


class LSTM(keras.Model):
    
    def __init__(self,
                 vocab_size,
                 output_dim, 
                 embedding_dim=100,
                 mask_zero=True, 
                 n_layers=1,
                 bidirectional=False,
                 dropout_rate=0.0,
                 activation=layers.ReLU(), 
                 activation_out=tf.math.softmax):
        
        super().__init__()
        
        self.embedding = layers.Embedding(
            vocab_size, embedding_dim, mask_zero=mask_zero)

        assert isinstance(n_layers, int) and n_layers >= 1, "" \
        + "n_layers must be an integer greater than 0."
        
        self.n_layers = n_layers
        
        def get_lstm(bi_direct, return_seq):
            if bi_direct:
                return layers.Bidirectional(
                    layers.LSTM(embedding_dim, 
                                     return_sequences=return_seq))
            
            return layers.LSTM(embedding_dim, 
                                    return_sequences=return_seq)
  
        if self.n_layers == 1:
            self.lstm = get_lstm(bidirectional, False)
        else:
            self.lstm = []
            for _ in range(self.n_layers-1):
                self.lstm.append(get_lstm(bidirectional, True))
            self.lstm.append(get_lstm(bidirectional, False))
        
        self.dense = layers.Dense(embedding_dim // 2)
        self.activation = activation
        self.dense_out = layers.Dense(output_dim)
        self.activation_out = activation_out
    
    def encoder(self, embd):
        if self.n_layers == 1:
            return self.lstm(embd)

        for lstm in self.lstm:
            embd = lstm(embd)
        return embd

    def call(self, text_ids):
        text_embd = self.embedding(text_ids)        
        encoded = self.encoder(text_embd)
        hidden_out = self.activation(self.dense(encoded))
        out_logits = self.dense_out(hidden_out)
        return self.activation_out(out_logits)


class SimpleRNN(keras.Model):
    
    def __init__(self,
                 vocab_size,
                 output_dim, 
                 embedding_dim=100,
                 mask_zero=True, 
                 n_layers=1,
                 bidirectional=False,
                 dropout_rate=0.0,
                 activation=layers.ReLU(), 
                 activation_out=tf.math.softmax):
        
        super().__init__()
        
        self.embedding = layers.Embedding(
            vocab_size, embedding_dim, mask_zero=mask_zero)
        
        assert isinstance(n_layers, int) and n_layers >= 1, "" \
        + "n_layers must be an integer greater than 0."
        
        self.n_layers = n_layers
        
        def get_rnn(bi_direct, return_seq):
            if bi_direct:
                return layers.Bidirectional(
                    layers.SimpleRNN(embedding_dim, 
                                     return_sequences=return_seq))
            
            return layers.SimpleRNN(embedding_dim, 
                                    return_sequences=return_seq)
  
        if self.n_layers == 1:
            self.rnn = get_rnn(bidirectional, False)
        else:
            self.rnn = []
            for _ in range(self.n_layers-1):
                self.rnn.append(get_rnn(bidirectional, True))
            self.rnn.append(get_rnn(bidirectional, False))
        
        self.dense = layers.Dense(embedding_dim // 2)
        self.activation = activation
        self.dense_out = layers.Dense(output_dim)
        self.activation_out = activation_out
    
    def encoder(self, embd):
        if self.n_layers == 1:
            return self.rnn(embd)

        for rnn in self.rnn:
            embd = rnn(embd)
        return embd

    def call(self, text_ids):
        text_embd = self.embedding(text_ids)       
        encoded = self.encoder(text_embd)
        hidden_out = self.activation(self.dense(encoded))
        out_logits = self.dense_out(hidden_out)
        return self.activation_out(out_logits)
