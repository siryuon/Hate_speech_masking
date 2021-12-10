from tensorflow.keras import initializers, regularizers, constraints, optimizers, layers
from tensorflow.keras import Input, Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import concatenate, Dense, LSTM, Dropout, Bidirectional, Embedding, Conv1D, MaxPooling1D, BatchNormalization, Concatenate, Layer, GlobalMaxPool1D, GlobalMaxPooling1D, GlobalAveragePooling1D, GRU, SpatialDropout1D
from keras import backend as K
from tensorflow.keras.preprocessing.sequence import pad_sequences
from gensim.models import FastText
import re

import konlpy
from eunjeon import Mecab

ft_model = FastText.load('fasttext_310k_1109.model')
max_len = 35

class Attention(Layer):
    def __init__(self, step_dim, W_regularizer=None, b_regularizer=None, W_constraint=None, b_constraint=None, bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight(shape=(input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim  

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim

def create_model(embedding_matrix, maxlen):
  sequence_input = Input(shape=(maxlen,), dtype='int32')
  embedded_sequences = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],  input_length=maxlen, weights=[embedding_matrix], trainable=False)(sequence_input)
  dropout_1 = SpatialDropout1D(0.1)(embedded_sequences)
  bilstm = Bidirectional(LSTM(32, return_sequences=True))(dropout_1)
  bigru = Bidirectional(GRU(32, return_sequences=True))(bilstm)

  atten_1 = Attention(maxlen)(bilstm)
  atten_2 = Attention(maxlen)(bigru)
  avg_pool = GlobalAveragePooling1D()(bigru)
  max_pool = GlobalMaxPooling1D()(bigru)

  conc = concatenate([atten_1, atten_2, avg_pool, max_pool])
  dense = Dense(16, activation='relu')(conc)
  dropout_2 = Dropout(0.1)(dense)
  output = Dense(1, activation='sigmoid')(dropout_2)

  model = Model(inputs=sequence_input, outputs=output)

  return model

#Remove morphemes from specific parts-of-speech
def pos_delete(tag, pos):
  for p in pos:
    if (p not in tag) == False:
      if 'VV+' not in tag:
        return False
      else:
        return True
    else:
      continue
  
  return True

#Parts-of-speech list
POS = ['CP', 'JKS', 'JKC', 'JKG', 'JKO', 'JKB', 'JKV', 'JKQ', 'JX', 'JC', 'JO', 'EP', 'EF', 'EC', 'ETN', 'ETM', 'VCP', 'VCN', 'SF', 'SY', 'XSN', 'XSV', 'XSA', 'XR']

#Vectorized using a trained FastText model
def vectorize_data(data, vocab: dict) -> list:
  keys = list(vocab.keys())
  filter_unknown = lambda word: vocab.get(word, None) is not None
  result = []
  for d in data:
    filtered = list(filter(filter_unknown,d))
    res = []
    for f in filtered:
      res.append(vocab[f].index)
    result.append(res)
  
  return result

#Text tokenize function (To use Predictor function)
def text_tokenize(text):
  mecab = Mecab()
  train_tokenized = []

  ps = mecab.pos(text)
  tagged = []
  for p in ps:
    if pos_delete(p[1], POS):
      tagged.append(p[0])
  temp_X = [word for word in tagged]
  train_tokenized.append(temp_X)

  return train_tokenized

#Convert to probability corresponding to each label
def predict_proba(pred):
  proba = []
  for i in range(len(pred)):
    proba.append([float(1-pred[i]), pred[i][0]])
  
  return proba

#Class prediction based on the transformed probability
def class_predict(proba):
  return np.argmax(proba, axis=1)

def model_predict(model, X_test):
  test_pred = model.predict(X_test)
  test_pred_proba = predict_proba(test_pred)
  test_preds = class_predict(test_pred_proba)

  return test_preds

def predictor(text, model, vocab=ft_model.wv.vocab, maxlen=max_len):
  pattern = re.compile('[^ㄱ-ㅎㅏ-ㅣ가-힣]')
  text = re.sub(pattern, ' ', text)
  tokenized = text_tokenize(text)
  tokenized_padded = pad_sequences(sequences=vectorize_data(tokenized, vocab=vocab), maxlen=maxlen, padding='post')

  ratio = round(model.predict(tokenized_padded)[0][0] * 100, 2)

  badword_list = []
  for idx, token in enumerate(tokenized[0]):
    pad_token = pad_sequences(sequences=vectorize_data([[token]], vocab=ft_model.wv.vocab), maxlen=35, padding='post')
    token_ratio = round(model.predict(pad_token)[0][0] * 100, 2)
    if token_ratio > 50:
      badword_list.append(token)
  
  return badword_list

def masker(text, badword):
  masked_text = text
  for bad in badword:
    masked_text = masked_text.replace(bad, '*' * len(bad), 1)
  
  return masked_text

def pred_mask(model, text):
  badword = predictor(text, model)
  masked_text = masker(text, badword)
  print('\n')
  print('Original Text: ', text)
  print('Masked Text: ', masked_text)
  print('-' * 80)
  return text, masked_text
