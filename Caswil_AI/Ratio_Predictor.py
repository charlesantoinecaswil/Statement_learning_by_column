
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
from keras.layers.merge import Concatenate
import numpy as np
from nltk.corpus import stopwords
#from Caswil_AI.Other_function import variable, dict_sentence_with_answer, ask_preparation



class variable():

    def __init__(self, name):

        self.name = name

    
    def type(self):

        pass


class  integer_variable(variable):
    
    def __init__(self, name):
        super().__init__(name)
        self.type = "integertype"



class  string_variable(variable):
    
    def __init__(self,name):
        super().__init__(name)
        self.type = "stringtype"



def data_preparation(sentence, bank_of_columns, result):

    column_mentionned = list()
    dict_of_word = dict()

    for columns in bank_of_columns:

        if columns.name in sentence:
            
            column_mentionned.append(columns)
            
            dict_of_word[columns.name] = dict()

            if columns.type == "integertype":

                dict_of_word[columns.name]["type"] = 'integertype'

            if columns.type == "stringtype":

                dict_of_word[columns.name]["type"] = 'not_integertype'

            dict_of_word[columns.name]["word_list"] = list()

            for i in sentence.split(columns.name):

                for column in bank_of_columns:

                    if column.name in i:

                        i = i.replace(column.name,column.type)

                dict_of_word[columns.name]["word_list"].append(i)

    return dict_sentence_with_answer(dict_of_word,result)




## answer 'NUMERATEUR;DENOMINATEUR;VALEUR_NUMÉRIQUE'
def dict_sentence_with_answer(dict_sentence, answer):

    numerateur = answer.split(';')[0].split('|')
    denominateur = answer.split(';')[1].split('|')
    valeur_numerique = answer.split(';')[2].split('|')

    for i in numerateur:

        dict_sentence[i]["Result"] = 'numerateur'

    for i in denominateur:

        dict_sentence[i]["Result"] = 'denominateur'

    for i in valeur_numerique:

        dict_sentence[i]["Result"] = 'valeur_numerique'

    return dict_sentence


## Data cleaning


def ask_preparation(sentence, bank_of_columns):

    column_mentionned = list()
    dict_of_word = dict()

    for columns in bank_of_columns:

        if columns.name in sentence:
            
            column_mentionned.append(columns)
            
            dict_of_word[columns.name] = dict()

            if columns.type == "integertype":

                dict_of_word[columns.name]["type"] = 'integertype'

            if columns.type == "stringtype":

                dict_of_word[columns.name]["type"] = 'not_integertype'

            dict_of_word[columns.name]["word_list"] = list()

            for i in sentence.split(columns.name):

                for column in bank_of_columns:

                    if column.name in i:

                        i = i.replace(column.name,column.type)

                dict_of_word[columns.name]["word_list"].append(i)

    data_frame = pd.DataFrame()
    column_name = list()
    type_de_list = list()
    sentence_before = list()
    sentence_after = list()

    for i in dict_of_word.keys():
        dict_tmp = dict_of_word[i]
        column_name.append(i)
        type_de_list.append(dict_tmp['type'])
        sentence_before.append(dict_tmp['word_list'][0])
        sentence_after.append(dict_tmp['word_list'][1])

    
    data_frame["column_type"] = type_de_list
    data_frame["sentence_before_column"] = sentence_before
    data_frame["sentence_after_column"] = sentence_after
    data_frame["Name"] =  column_name

    return data_frame


            





class define_variable_ratio():


    def __init__(self, epoch, batch, lstm_out, number_of_word, embed_dim, max_question_length, number_of_statement,bank_of_columns):

        self.embed_dim = embed_dim
        self.epoch = epoch
        self.batch = batch
        self.number_of_word = number_of_word
        self.max_question_length = max_question_length
        self.lstm_out = lstm_out
        self.number_of_statement = number_of_statement
        self.bank_of_columns = bank_of_columns
        

    
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

        self.input_1 = Input(shape=(self.max_question_length,))
        self.embedding_layer_1 = Embedding(self.number_of_word, self.embed_dim, input_length = self.max_question_length, dropout = 0.2)(self.input_1)
        self.lstm_layer_1 = LSTM(128)(self.embedding_layer_1)

        self.input_2 = Input(shape=(self.max_question_length,))
        self.embedding_layer_2 = Embedding(self.number_of_word, self.embed_dim, input_length = self.max_question_length, dropout = 0.2)(self.input_2)
        self.lstm_layer_2 = LSTM(128)(self.embedding_layer_2)

        self.input_3 = Input(shape=(self.X_type_length,))
        self.dense_1 = Dense(5)(self.input_3)
        self.dense_2 = Dense(5)(self.dense_1)

        self.concat_layer = Concatenate()([self.lstm_layer_1, self.lstm_layer_2,self.dense_2])
        self.dense_3 = Dense(10)(self.concat_layer)

        self.output = Dense(3, activation='softmax')(self.dense_3)

        self.model = Model(inputs=[self.input_1, self.input_2,self.input_3], outputs=self.output)



    
    def save_model(self,save_name):
        self.model.save(save_name)


    def load_model(self,model_name):

        self.model = load_model(model_name)


    def load_data(self, data_file):

        data_frame_wo_transformation = pd.read_csv(data_file)
        column_type = list()
        sentence_before_column = list()
        sentence_after_column = list()
        result = list()
        print(data_frame_wo_transformation)
        for j, i in data_frame_wo_transformation.iterrows():
            dict_phrase = data_preparation(i["Question"],self.bank_of_columns,i["Statement"])
            for i in dict_phrase.keys():
                column_type.append(dict_phrase[i]["type"])
                sentence_before_column.append(dict_phrase[i]["word_list"][0])
                sentence_after_column.append(dict_phrase[i]["word_list"][1])
                result.append(dict_phrase[i]["Result"])


        

        self.tokenizer.fit_on_texts(sentence_before_column)
        Before_sentence = self.tokenizer.texts_to_sequences(sentence_before_column)
        Before_sentence = pad_sequences(Before_sentence,maxlen=self.max_question_length)

        self.tokenizer.fit_on_texts(sentence_after_column)
        After_sentence = self.tokenizer.texts_to_sequences(sentence_after_column)
        After_sentence = pad_sequences(After_sentence,maxlen=self.max_question_length)

        data_frame = pd.DataFrame()
        data_frame["column_type"] = column_type
        data_frame["sentence_before_column"] = sentence_before_column
        data_frame["sentence_after_column"] = sentence_after_column
        data_frame["Result"] =  result
        

        X_type = pd.get_dummies(data_frame['column_type']).values
        self.X_type = pd.get_dummies(data_frame['column_type']).keys()
        self.X_type_length = len(pd.get_dummies(data_frame['column_type']).keys())
        
        self.tokenizer.fit_on_texts(sentence_after_column)
        X_sentence_before = self.tokenizer.texts_to_sequences(data_frame["sentence_before_column"])
        X_sentence_before = pad_sequences(X_sentence_before,self.max_question_length)

        self.tokenizer.fit_on_texts(sentence_after_column)
        X_sentence_after = self.tokenizer.texts_to_sequences(data_frame["sentence_after_column"])
        X_sentence_after = pad_sequences(X_sentence_after,self.max_question_length)

        self.Y = pd.get_dummies(data_frame['Result'])
        Y = self.Y.values



        return X_type,X_sentence_before,X_sentence_after,Y

        #self.tokenizer.fit_on_texts(sentence_before_column)
        #Before_sentence = self.tokenizer.texts_to_sequences(sentence_before_column)
        #Before_sentence = pad_sequences(Before_sentence,maxlen=self.max_question_length)

        #self.tokenizer.fit_on_texts(sentence_after_column)
        #After_sentence = self.tokenizer.texts_to_sequences(sentence_after_column)
        #After_sentence = pad_sequences(After_sentence,maxlen=self.max_question_length)
        #X = np.asarray(data_frame[['column_type', 'Before_sentence', 'After_sentence']])
        #Y = np.asarray(df['Result'])


       
        #X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state = 42)

        #return X_train, X_test, Y_train, Y_test


    def train_model(self, data_file, epoch = 20, batch = 1 ):
        
        X_type,X_sentence_before,X_sentence_after,Y = self.load_data(data_file)
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
        self.model.fit([X_sentence_before, X_sentence_after, X_type], Y, batch_size = batch, nb_epoch = epoch, verbose = 1)
        
        


    def ask_question(self, Question):

        data_frame = ask_preparation(Question, self.bank_of_columns)
        X_type = list()
        for j in (data_frame['column_type']).values:
            tmp_X_type = list()
            for i in self.X_type:
                if i == j:
                    tmp_X_type.append(1)
                else:
                    tmp_X_type.append(0)
            X_type.append(tmp_X_type)
                
        self.X_type_length = len(pd.get_dummies(data_frame['column_type']).keys())
        
        self.tokenizer.fit_on_texts(data_frame["sentence_before_column"])
        X_sentence_before = self.tokenizer.texts_to_sequences(data_frame["sentence_before_column"])
        X_sentence_before = pad_sequences(X_sentence_before,self.max_question_length)

        self.tokenizer.fit_on_texts(data_frame["sentence_after_column"])
        X_sentence_after = self.tokenizer.texts_to_sequences(data_frame["sentence_after_column"])
        X_sentence_after = pad_sequences(X_sentence_after,self.max_question_length)
        result = self.model.predict([X_sentence_before, X_sentence_after, np.array(X_type)])
        
        
        
        for i,j in zip(np.argmax(result,axis=1),data_frame["Name"]):
             print(list(self.Y.keys())[i],j)







if __name__ == "__main__":
    

    epoch = 250
    batch = 1
    number_of_word = 2000
    lstm_out = 100
    embed_dim = 100
    max_question_length = 30
    number_of_statement = 34
    bank_of_columns = [integer_variable("GROUPE"),integer_variable("CLASSE"),string_variable("Type_de_list_EN"),string_variable("Nom_Anglais"),integer_variable("FRANCHISE_ASSURE_APPLIQUEE_MALADIE")]

    User = define_variable_ratio(epoch, batch, lstm_out, number_of_word, embed_dim, max_question_length, number_of_statement,bank_of_columns)
    User.create_tokenizer()
    X_type,X_sentence_before,X_sentence_after,Y = User.load_data("C:\\Users\\Charles-Antoine Pare\\Documents\\Statement_learning\\ratio_data.csv")
    User.create_model()
    User.train_model("C:\\Users\\Charles-Antoine Pare\\Documents\\Statement_learning\\ratio_data.csv")
    User.ask_question('Je veux le ratio de FRANCHISE_ASSURE_APPLIQUEE_MALADIE selon le Nom_Anglais par Type_de_list_EN')
 