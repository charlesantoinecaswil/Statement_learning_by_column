from os import listdir
from os.path import isfile, join
import numpy
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Concatenate
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
from nltk.corpus import stopwords
import numpy as np


class column_finder():

    def __init__(self, path_output, xml_path,max_length = 100):

        self.csv_output_path = path_output
        self.xml_path = xml_path
        self.data_statement_dict = self.create_dictionnary()
        self.max_length = max_length


    def create_dictionnary(self):
        onlyfiles = [f for f in listdir(self.xml_path) if isfile(join(self.xml_path, f))]

        dict_data_frame_by_statement = dict()

        for statement in onlyfiles:

            csv_name = self.csv_output_path + "\\" + statement.replace('.Caswil.ST.Xml','_OUTPUT.csv')
            try:
                dict_data_frame_by_statement[statement] = pd.read_csv(csv_name)

            except FileNotFoundError:
                continue

        return dict_data_frame_by_statement


    def create_model_data(self):

        column = list()
        ord_letter = list()
        variable_list = list()
        nombre_d_espace = list()
        type_of_charachter_list = list()
        self.column_type_dict = dict()
        self.data_frame_transforme = pd.DataFrame()
        for data_frame in self.data_statement_dict.keys():
            data_frame_tmp = self.data_statement_dict[data_frame]
            self.column_type_dict.update(dict(data_frame_tmp.dtypes))

            for column_tmp_name in data_frame_tmp.keys():

                for variable in data_frame_tmp[column_tmp_name].values:
                    list_letter = list()
                    if str(variable) == 'nan':
                        continue
                    else:
                        
                        for letter in str(variable):
                            list_letter.append(ord(letter))
                    

                    if len(list_letter) != 0:
                        nombre_d_espace.append(str(variable).count(' '))
                        upper = any([i.isupper() for i in str(variable)])
                        lower = any([i.islower() for i in str(variable)])
                        digit = any([i.isdigit() for i in str(variable)])
                        type_of_charachter = ''
                        if upper:
                            type_of_charachter += 'upper'

                        if lower:
                            type_of_charachter += 'lower'

                        if digit:
                            type_of_charachter += 'digit'

                        type_of_charachter_list.append(type_of_charachter)

                        ord_letter.append(list_letter)
                        column.append(column_tmp_name)

        self.data_frame_transforme['ord_letter'] = ord_letter
        self.data_frame_transforme['column'] = column

        self.X = pad_sequences(ord_letter,self.max_length)
        self.X_espace = pd.get_dummies(nombre_d_espace)
        self.X_charachter = pd.get_dummies(type_of_charachter_list)
        self.Y = pd.get_dummies(self.data_frame_transforme['column'])

    
    def create_model(self):

        
        input_1 = Input(shape=(self.max_length,))
        Embedding_1 = Embedding(1000, 100, input_length = self.max_length)(input_1)
        lstm_layer_1 = LSTM(128)(Embedding_1)
        dense_2 = Dense(128)(lstm_layer_1)
        dense_8 = Dense(36)(dense_2)

        input_2 = Input(shape=(len(self.X_espace.keys()),))
        dense_3 = Dense(12)(input_2)
        dense_4 = Dense(12)(dense_3)

        input_3 = Input(shape=(len(self.X_charachter.keys()),))
        dense_5 = Dense(12)(input_3)
        dense_6 = Dense(12)(dense_5)

        concat_layer = Concatenate()([dense_8, dense_4, dense_6])
        dense_7 = Dense(32)(concat_layer)

        output = Dense(len(self.Y.keys()), activation = 'softmax')(dense_7)


        self.model = Model(inputs=[input_1, input_2,input_3], outputs=output)
        self.model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])

    def train_model(self, batch=1, epoch=4):

        self.model.fit([self.X, self.X_espace, self.X_charachter], self.Y.values, batch_size = batch, nb_epoch = epoch, verbose = 1)


    def load_model(self,model_name):

        self.model = load_model(model_name)


    def ask_question(self, Question, nb_statement_to_guess):

        #stop_words = set(stopwords.words('french')) | set(stopwords.words('english'))
        stop_words = set('aucunstopword')
        
        answer_dict = dict()

        index = 1
        word_list = list()

        for i in self.column_type_dict.keys():
            if ((i +' ') in (Question + ' ')):
                
                index_tmp = str(index)
                answer_dict[index_tmp] = dict()
                answer_dict[index_tmp]['Part_of_text'] = i
                answer_dict[index_tmp]['columns_associated'] = i
                answer_dict[index_tmp]['type'] = str(self.column_type_dict[i])
                Question = Question.replace(i + ' ','')
                nb_statement_to_guess += -1
                index += 1

        word_tokens = Question.split(" ")
        filtered_sentence = [w for w in word_tokens if not w in stop_words]

        for i in range(0,nb_statement_to_guess):
            one_word_list = list()
            word_list = list()
            espace = list()
            charachter = list()
            for one_word in range(len(filtered_sentence)):

                one_word_list.append([ord(letter) for letter in filtered_sentence[one_word]])
                word_list.append(filtered_sentence[one_word])
                espace_tmp = [1] + [0]*(len(self.X_espace.keys())-1)
                espace.append(espace_tmp)
                
                if not any([i.isupper() for i in str(filtered_sentence[one_word])]) and not any([i.islower() for i in str(filtered_sentence[one_word])]) and any([i.isdigit() for i in str(filtered_sentence[one_word])]):
                    charachter_tmp = [1,0,0,0]

                if any([i.isupper() for i in str(filtered_sentence[one_word])]) and not any([i.islower() for i in str(filtered_sentence[one_word])]) and not any([i.isdigit() for i in str(filtered_sentence[one_word])]):
                    charachter_tmp = [0,1,0,0]

                if any([i.isupper() for i in str(filtered_sentence[one_word])]) and not any([i.islower() for i in str(filtered_sentence[one_word])]) and any([i.isdigit() for i in str(filtered_sentence[one_word])]):
                    charachter_tmp = [0,0,1,0]

                if any([i.isupper() for i in str(filtered_sentence[one_word])]) and any([i.islower() for i in str(filtered_sentence[one_word])]) and not any([i.isdigit() for i in str(filtered_sentence[one_word])]):
                    charachter_tmp = [0,0,0,1]

                charachter.append(charachter_tmp)




                            

                try:
                    two_word = filtered_sentence[one_word] + ' ' + filtered_sentence[one_word + 1]
                    one_word_list.append([ord(letter) for letter in two_word])
                    word_list.append(two_word)
                    espace_tmp = [0,1] + [0]*(len(self.X_espace.keys())-2)
                    espace.append(espace_tmp)

                    

                    if not any([i.isupper() for i in str(two_word)]) and not any([i.islower() for i in str(two_word)]) and any([i.isdigit() for i in str(two_word)]):
                        charachter_tmp = [1,0,0,0]

                    if any([i.isupper() for i in str(two_word)]) and not any([i.islower() for i in str(two_word)]) and not any([i.isdigit() for i in str(two_word)]):
                        charachter_tmp = [0,1,0,0]

                    if any([i.isupper() for i in str(two_word)]) and not any([i.islower() for i in str(two_word)]) and any([i.isdigit() for i in str(two_word)]):
                        charachter_tmp = [0,0,1,0]

                    if any([i.isupper() for i in str(two_word)]) and any([i.islower() for i in str(two_word)]) and not any([i.isdigit() for i in str(two_word)]):
                        charachter_tmp = [0,0,0,1]
                    charachter.append(charachter_tmp)



                    three_word = filtered_sentence[one_word] + ' ' + filtered_sentence[one_word + 1] + ' ' + filtered_sentence[one_word + 2]
                    one_word_list.append([ord(letter) for letter in three_word])
                    word_list.append(three_word)
                    espace_tmp = [0,0,1] + [0]*(len(self.X_espace.keys())-3)
                    espace.append(espace_tmp)
                    
                    if not any([i.isupper() for i in str(three_word)]) and not any([i.islower() for i in str(three_word)]) and any([i.isdigit() for i in str(three_word)]):
                        charachter_tmp = [1,0,0,0]

                    if any([i.isupper() for i in str(three_word)]) and not any([i.islower() for i in str(three_word)]) and not any([i.isdigit() for i in str(three_word)]):
                        charachter_tmp = [0,1,0,0]

                    if any([i.isupper() for i in str(three_word)]) and not any([i.islower() for i in str(three_word)]) and any([i.isdigit() for i in str(three_word)]):
                        charachter_tmp = [0,0,1,0]

                    if any([i.isupper() for i in str(three_word)]) and any([i.islower() for i in str(three_word)]) and not any([i.isdigit() for i in str(three_word)]):
                        charachter_tmp = [0,0,0,1]
                    charachter.append(charachter_tmp)


                    four_word = filtered_sentence[one_word] + ' ' + filtered_sentence[one_word + 1] + ' ' + filtered_sentence[one_word + 2] + ' ' + filtered_sentence[one_word + 3]
                    one_word_list.append([ord(letter) for letter in four_word])
                    word_list.append(four_word)
                    espace_tmp = [0,0,0,1] + [0]*(len(self.X_espace.keys())-4)
                    espace.append(espace_tmp)

                   
                    if not any([i.isupper() for i in str(four_word)]) and not any([i.islower() for i in str(four_word)]) and any([i.isdigit() for i in str(four_word)]):
                        charachter_tmp = [1,0,0,0]

                    if any([i.isupper() for i in str(four_word)]) and not any([i.islower() for i in str(four_word)]) and not any([i.isdigit() for i in str(four_word)]):
                        charachter_tmp = [0,1,0,0]

                    if any([i.isupper() for i in str(four_word)]) and not any([i.islower() for i in str(four_word)]) and any([i.isdigit() for i in str(four_word)]):
                        charachter_tmp = [0,0,1,0]

                    if any([i.isupper() for i in str(four_word)]) and any([i.islower() for i in str(four_word)]) and not any([i.isdigit() for i in str(four_word)]):
                        charachter_tmp = [0,0,0,1]
                    charachter.append(charachter_tmp)

                except IndexError:
                    continue

            input_1 = pad_sequences(one_word_list, self.max_length)
            input_2 = np.array(espace)
            input_3 = np.array(charachter)
            result = self.model.predict([input_1, input_2, input_3])

            ind = np.unravel_index(np.argmax(result, axis=None), result.shape)



        #for i in range(0,nb_statement_to_guess):
            ind = np.unravel_index(np.argmax(result, axis=None), result.shape)

            column = list(self.Y.keys())[ind[1]]
            word = word_list[ind[0]]
            index_tmp = str(index)
            answer_dict[index_tmp] = dict()
            answer_dict[index_tmp]['Part_of_text'] = word
            answer_dict[index_tmp]['columns_associated'] = column
            answer_dict[index_tmp]['type'] = str(self.column_type_dict[column])
            index += 1
        
            for i in word.split(' '):

                filtered_sentence.remove(i)

        return answer_dict,word_list
            
        

        
        
        

if __name__ == "__main__":
    
    path_output = 'C:\\Users\\Charles-Antoine Pare\\Documents\\AI_Package_Ratio\\AI_Package_Ratio\\app\\result'
    xml_path = 'C:\\Users\\Charles-Antoine Pare\\Documents\\AI_Package_Ratio\\AI_Package_Ratio\\app\\xml'
    User = column_finder(path_output, xml_path)
    aa = User.create_dictionnary()
    User.create_model_data()
    #User.create_model()
    #User.train_model() 
    #User.load("new_model.bin")
    User.load_model('C:\\Users\\Charles-Antoine Pare\\Documents\\Statement_learning_by_column\\Saved_Models\\new_model_ratio.bin')
    User.data_statement_dict
    User.ask_question('3703268718001',1)