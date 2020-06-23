import numpy
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Flatten
from keras.layers import Dense, SpatialDropout1D
import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
from keras.models import load_model
from keras.utils import to_categorical
from more_itertools import sort_together



class Select_Statement():


    def __init__(self, epoch, batch, lstm_out, number_of_word, embed_dim, max_question_length, number_of_statement):

        self.embed_dim = embed_dim
        self.epoch = epoch
        self.batch = batch
        self.number_of_word = number_of_word
        self.max_question_length = max_question_length
        self.lstm_out = lstm_out
        self.number_of_statement = number_of_statement
        

    
    def create_tokenizer(self):
        ## Need to be created after 
        self.tokenizer = Tokenizer(nb_words = self.number_of_word, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True,split=' ')
        


    def load_tokenizer(self,file_name):

        with open(file_name, 'rb') as handle:
            dict_tokenizer_statement_list = pickle.load(handle)
            self.tokenizer = dict_tokenizer_statement_list['tokenizer']
            self.Y = dict_tokenizer_statement_list['statement']

    def save_tokenizer(self, file_name):

        dict_tokenizer_statement_list = dict()
        dict_tokenizer_statement_list['tokenizer'] = self.tokenizer
        dict_tokenizer_statement_list['statement'] = self.Y

        with open(file_name, 'wb') as handle:
            pickle.dump(dict_tokenizer_statement_list, handle, protocol=pickle.HIGHEST_PROTOCOL)


    def create_model(self):

        self.model = Sequential()
        ## Dropout permet d'enlever des neurones des layers pour eviter le overfitting
        self.model.add(Embedding(self.number_of_word, self.embed_dim, input_length = self.max_question_length, dropout = 0.2))
        self.model.add(LSTM(128))
        self.model.add(Dense(34, activation='sigmoid'))
        self.model.compile(loss = 'binary_crossentropy', optimizer='adam',metrics = ['accuracy'])
        self.model.summary()

    
    def save_model(self,save_name):
        self.model.save(save_name)


    def load_model(self,model_name):

        self.model = load_model(model_name)


    def load_data(self, data_file):

        data_frame = pd.read_csv(data_file)
        columns_list = list()

        for i in data_frame['Statement']:
            columns_list.append(list(i.replace('[','').replace("]","").replace("'","").split(",")))

        data_frame["Statement"] = columns_list


        data_frame["Statement"] = columns_list
        self.tokenizer.fit_on_texts(data_frame['Question'].values)
        X = self.tokenizer.texts_to_sequences(data_frame['Question'].values)
        X = pad_sequences(X,maxlen=self.max_question_length)
        s = data_frame['Statement']
        self.Y = pd.get_dummies(s.apply(pd.Series).stack()).sum(level=0)
        Y = self.Y.values
        X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state = 42)

        return X_train, X_test, Y_train, Y_test


    def train_model(self, data_file, epoch = 20, batch = 1 ):
        
        X_train, X_test, Y_train, Y_test = self.load_data(data_file)

        self.model.fit(X_train, Y_train, batch_size = batch, nb_epoch = epoch, verbose = 1)
        print(X_test)
        


    def ask_question(self, Question):

        tokenize_question = self.tokenizer.texts_to_sequences([Question])
        tokenize_question = pad_sequences(tokenize_question,self.max_question_length)
        result = self.model.predict(tokenize_question)
        max_position = result[0].argmax()
        list_column = list(User.Y.keys())
        list_result = result.tolist()[0]
        
        print(sort_together([list_result,list_column])[1])
        return [list_column,list_result]






if '__main__' == __name__:

    epoch = 250
    batch = 1
    number_of_word = 2000
    lstm_out = 100
    embed_dim = 100
    max_question_length = 30
    number_of_statement = 34

    User = Select_Statement(epoch, batch, lstm_out, number_of_word, embed_dim, max_question_length, number_of_statement)
    User.create_tokenizer()
    User.create_model()
    User.train_model('C:\\Users\\Charles-Antoine\\Desktop\\Statement_learning_by_column\\data_statement.csv', 10, 1)
    User.save_model('model_test.bin')
    User.save_tokenizer('tokenizer_test.bin')
    User.ask_question('coassurance')
