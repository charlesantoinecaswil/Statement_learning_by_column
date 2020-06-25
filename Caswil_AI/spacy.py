import en_core_web_sm
import fr_core_news_sm
import pandas as pd
from os import listdir
from os.path import isfile, join
import numpy as np
from spacy.lang.en import English
from spacy.matcher import Matcher
from spacy.pipeline import EntityRuler
import spacy
import random
import pandas as pd
import pickle
from spacy.pipeline import TextCategorizer
import os



class column_finder():

    def __init__(self, path_output, xml_path, pickle_path,max_length = 100):

        self.csv_output_path = path_output ## Les csv output dans le result folder
        self.xml_path = xml_path ## path des statement
        self.data_statement_dict = self.create_dictionnary()
        self.max_length = max_length ## Pas important présentement
        self.pickle_path = pickle_path


    def create_dictionnary(self):
        onlyfiles = [f for f in listdir(self.xml_path) if isfile(join(self.xml_path, f))] ## creation de la liste des statements dans l'app

        dict_data_frame_by_statement = dict()

        for statement in onlyfiles:

            csv_name = self.csv_output_path + "\\" + statement.replace('.Caswil.ST.Xml','_OUTPUT.csv') ## On trouve les csv correspondant au statement de l'app
            try:
                dict_data_frame_by_statement[statement] = pd.read_csv(csv_name)

            except FileNotFoundError:
                continue

        for df in dict_data_frame_by_statement.keys():

            for columns in dict_data_frame_by_statement[df]:

                if dict_data_frame_by_statement[df][columns].dtype == np.dtype('object'): ## On garde seulement les types object pour l'instant
                    pass
                else:
                    del dict_data_frame_by_statement[df][columns]

                    ## On a maintenant un dictionnaire de pd dataframe contenant les outputs des statements

        dict_value = dict() # Ce dictionnaire contiendra comme clés les noms de colonnes et comme valeur un autre dictionnaire contant les statements possèdant la colonne et les valeurs possibles de cette colonne

        for df in dict_data_frame_by_statement.keys():

            for columns in dict_data_frame_by_statement[df]:

                try:
                    dict_value[columns]['source'].append(df)
                    dict_value[columns]['value'] = list(set(dict_value[columns]['value'].append(set(dict_data_frame_by_statement[df][columns].values))))

                except NameError and KeyError:
                    dict_value[columns] = dict()
                    dict_value[columns]['source'] = [df]
                    dict_value[columns]['value'] = list(set(dict_data_frame_by_statement[df][columns].values))


            return dict_value

   
    def pattern_dict(self): ## On va créer un dictionnaire des patterns de valeurs. Les valeurs continuent numériques ne seront pas bien traitées
        
        dictionnaire_value = self.create_dictionnary()
        dict_pattern = dict()

        for column in dictionnaire_value.keys():

            
            try:
                dict_pattern[column]

            except KeyError:
                dict_pattern[column] = list()

            list_of_possible_pattern = list()
            column_name = column.replace('_', ' ')
            column_name_list_of_word = column_name.split(' ')
            pattern = list()
            for word in column_name_list_of_word:
                try:
                    float(word)
                    pattern.append({'IS_DIGIT': True})


                except ValueError:
                    pattern.append({'LOWER': word.lower()})

            dict_pattern[column].append(pattern)

            for value in dictionnaire_value[column]['value']:


                value = str(value)

                pattern = list()
                value = value.replace('_', ' ')
                value = value.replace(',', ' ')
                list_of_value = value.split(' ')

                for word in list_of_value:

                    try:
                        float(word)
                        pattern.append({'IS_DIGIT': True})


                    except ValueError:
                        pattern.append({'LOWER': word.lower()})

                if pattern not in dict_pattern[column]:
                    dict_pattern[column].append(pattern)

        return dict_pattern

    def return_sentence_pattern(self, sentence):
        nlp = spacy.blank("en")
        pattern_dict = self.pattern_dict()
        doc = nlp(sentence.replace(',',' '))
        dict_snswer = dict()
        dict_snswer['entities'] = list()
        for entities in pattern_dict.keys():
            
            matcher = Matcher(nlp.vocab)
            matcher.add(entities,pattern_dict[entities])
            spans = [doc[start:end] for match_id, start, end in matcher(doc)]
            entitie = [(span.start_char, span.end_char, entities) for span in spans]
            dict_snswer['entities'] += (entitie)


        clean_dict = dict()
        clean_dict['entities'] = list()

        all_overlap = list()
        


        for ent in dict_snswer['entities']:
            list_of_overlap = list()
            list_of_overlap.append(ent)
            
            for ent2 in dict_snswer['entities']:

                if ent == ent2:
                    continue
                
                
                entrange = set(range(ent[0],ent[1]))
                ent2range = set(range(ent2[0],ent2[1]))
                if len(entrange.intersection(ent2range)) != 0: 
                    list_of_overlap.append(ent2)

           
                
            all_overlap.append((list_of_overlap))

            
        clean_overlap = list()


        for index1,i in enumerate(all_overlap):
           
            if  not (set(i) in [set(j) for j in clean_overlap]):

                clean_overlap.append(i)

        
       

        for list_of_overlap in clean_overlap:
                               
            for index, element in enumerate(list_of_overlap):
                print(index, doc.text[element[0]:element[1]], element[2])
            choice = int(input('Choose the right element'))
            clean_dict['entities'].append(list_of_overlap[choice])
                
        self.save_observation(doc.text,clean_dict)
        
        
        return clean_dict


    def save_observation(self, sentence, new_dict):

        try:
            file = open(self.pickle_path,'rb')

            train_data_complet = pickle.load(file)
            train_data_complet.append((sentence, new_dict))

            file.close()
        except FileNotFoundError:

            train_data_complet = [(sentence, new_dict)]


        file = open(self.pickle_path,'wb')

        pickle.dump(train_data_complet,file)

        file.close()


    def create_training_data(self, list_of_sentence):
        TRAINING_DATA = list()

        for sentence in list_of_sentence:
            print(sentence)
            dict_ent = self.return_sentence_pattern(sentence)
            
            
            print('Ratio : 0')
            print('Select : 1')
            choice = input('Select 0 or 1')
            dict_ent['cats'] = dict()
            if int(choice) == 0 :
                dict_ent['cats']['Ratio'] = True
                dict_ent['cats']['Select'] = False
            else:
                dict_ent['cats']['Ratio'] = False
                dict_ent['cats']['Select'] = True
            

            
            TRAINING_DATA.append((sentence,dict_ent))


        return TRAINING_DATA


    def create_model(self):

        self.nlp = spacy.blank("en")
        ner = self.nlp.create_pipe("ner")
        textcat = self.nlp.create_pipe("textcat")
        dict_tmp = self.create_dictionnary()


        self.nlp.add_pipe(ner)
        self.nlp.add_pipe(textcat)

        for column in dict_tmp.keys():
            ner.add_label(column)


        textcat.add_label('Select')
        textcat.add_label('Ratio')


    def train_model(self, TRAINING_DATA):

        self.nlp.begin_training()

        for itn in range(40):
    # Shuffle the training data
            random.shuffle(DATA)
            losses = {}

    # Batch the examples and iterate over them
            for batch in spacy.util.minibatch(DATA, size=2):
                texts = [text for text, entities in batch]
                annotations = [entities for text, entities in batch]

        # Update the model
                self.nlp.update(texts, annotations, losses=losses)
                self.nlp.update(texts, annotations, losses=losses)
            print(losses)


    def guess_sentence(self, sentence):

        doc = self.nlp(sentence)


        print(doc.text,doc.cats)
        print("Entities", [(ent.text, ent.label_) for ent in doc.ents])

















path_output = 'Data\\result'
xml_path = 'Data\\xml'
json_path = 'Saved_dict\\tt.pkl'
User = column_finder(path_output, xml_path,json_path)
a = User.create_dictionnary()
User.return_sentence_pattern('')

TEXTS = ['I want the value of CAD equity for the client 45', 'Give me all my CAD fixed income for all my clients', 'I want to see client 90 portfolio', 'What are the investement in mutual funds for client 344', 'what are the differences between client 31 and client 98', 'I want all the clients with more USD equity than CAD', 'Give me the USD', 'Give me the equity', 'What is the financial ratio in the Eq Us - PIM?', 'What is the utilities ratio in the Eq Intl model?', 'How much in percentage is allocated in the Health Care sector in every accounts part of the eq intl model?']
DATA = User.create_training_data(TEXTS)

User.create_model()
User.train_model(DATA)


User.guess_sentence('I want the value of CAD equity for the client 45')
User.guess_sentence('What is the financial ratio in the Eq Us model?')




