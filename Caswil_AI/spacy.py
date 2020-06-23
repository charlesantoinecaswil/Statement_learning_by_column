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
            if choice == 0 :
                dict_ent['cats']['Ratio'] = True
                dict_ent['cats']['Select'] = False
            else:
                dict_ent['cats']['Ratio'] = False
                dict_ent['cats']['Select'] = True
            

            
            TRAINING_DATA.append((sentence,dict_ent))


        return TRAINING_DATA


    def create_model(self, data):

        nlp = spacy.blank("en")
        ner = nlp.create_pipe("ner")
        textcat = nlp.create_pipe("textcat")

        nlp.add_pipe(ner)
        nlp.add_pipe(textcat)





path_output = 'Data\\result'
xml_path = 'Data\\xml'
json_path = 'Saved_dict\\tt.pkl'
User = column_finder(path_output, xml_path,json_path)
a = User.create_dictionnary()
User.return_sentence_pattern('')

TEXTS = ['I want the value of CAD equity for the client 45', 'Give me all my CAD fixed income for all my clients', 'I want to see client 90 portfolio', 'What are the investement in mutual funds for client 344', 'what are the differences between client 31 and client 98', 'I want all the clients with more USD equity than CAD', 'Give me the USD', 'Give me the equity', 'What is the financial ratio in the Eq Us - PIM?', 'What is the utilities ratio in the Eq Intl model?', 'How much in percentage is allocated in the Health Care sector in every accounts part of the eq intl model?']
DATA = User.create_training_data(TEXTS)













patternaccountname = [{"LOWER": "client"}, {"IS_DIGIT": True}]
patternaccountname1 = [{"LOWER": "client"}, {"IS_DIGIT": False}]
patternaccountname2 = [{"LOWER": "clients"}, {"IS_DIGIT": False}]
patterncurrency1 = [{"LOWER": "eur"}]
patterncurrency2 = [{"LOWER": "cad"}]
patterncurrency3 = [{"LOWER": "usd"}]
patterncurrency4 = [{"LOWER": "aud"}]
patternAssetClass1 = [{"LOWER": "cash"},{"LOWER": "and"},{"LOWER": "cash"},{"LOWER": "equivalents"}]
patternAssetClass2 = [{"LOWER": "equity"}]
patternAssetClass3 = [{"LOWER": "fixed"}, {"LOWER": "income"}]
patternAssetClass4 = [{"LOWER": "alternative"}, {"LOWER": "investments"}]
patterSector1 = [{"LOWER": "mutual"}, {"LOWER": "funds"}]
patterSector2 = [{"LOWER": "hedge"}, {"LOWER": "funds"}]
patterSector3 = [{"LOWER": "federal"}, {"LOWER": "government"}]

pattern_list = {'accountname':[patternaccountname,patternaccountname1,patternaccountname2],'currency':[patterncurrency1,patterncurrency2,patterncurrency3,patterncurrency4],'AssetClass':[patternAssetClass1,patternAssetClass2,patternAssetClass3,patternAssetClass4],'Sector':[patterSector1, patterSector2, patterSector3]}

TEXTS = ['I want the value of CAD equity for the client 45', 'Give me all my CAD fixed income for all my clients', 'I want to see client 90 portfolio', 'What are the investement in mutual funds for client 344', "what are the differences between client 31 and client 98",'I want all the clients with more USD equity than CAD','Give me the USD','Give me the equity']
TEXTS = [i.lower() for i in TEXTS]

a.keys()
nlp = spacy.load('en_core_web_sm')
matcher = Matcher(nlp.vocab)
doc = nlp('I want to see client 90 portfolio')
options = {"compact": True, "bg": "#09a3d5",
           "color": "white", "font": "Source Sans Pro"}
spacy.displacy.serve(doc, style="dep", options=options)
spacy.displacy.serve(doc, style='dep')


TRAINING_DATA = dict()

for entities in pattern_list.keys():
    matcher = Matcher(nlp.vocab)
    matcher.add(entities,pattern_list[entities])
# Create a Doc object for each text in TEXTS
    for doc in nlp.pipe(TEXTS):
        TRAINING_DATA[doc.text] = dict()

        for entities in pattern_list.keys():
            matcher = Matcher(nlp.vocab)
            matcher.add(entities,pattern_list[entities])
            # Match on the doc and create a list of matched spans
            spans = [doc[start:end] for match_id, start, end in matcher(doc)]
            # Get (start character, end character, label) tuples of matches
            entities = [(span.start_char, span.end_char, entities) for span in spans]
            # Format the matches as a (doc.text, entities) tuple
            training_example = (doc.text, {"entities": entities})
            # Append the example to the training data
            try:
                TRAINING_DATA[doc.text]['entities'] += (entities)

            except KeyError:
                TRAINING_DATA[doc.text]['entities'] = list()
                TRAINING_DATA[doc.text]['entities'] += (entities)

matcher._patterns
TRAINING_DATA = list(TRAINING_DATA.items())

doc.label_

def on_match(matcher, doc, id, matches):
      print('Matched!', matches,id,matcher,doc)

matcher = Matcher(nlp.vocab)
matcher.add("HelloWorld", on_match, [{"LOWER": "hello"}, {"LOWER": "world"}])
matcher.add("GoogleMaps", on_match, [{"ORTH": "Google"}, {"ORTH": "Maps"}])
doc = nlp("HELLO WORLD on Google Maps.")
matches = matcher(doc.tex)

nlp = spacy.blank("en")
ner = nlp.create_pipe("ner")
textcat = nlp.create_pipe("textcat")
nlp.add_pipe(ner)
nlp.add_pipe(textcat)
ner.add_label("AccountName")
ner.add_label("currency")
ner.add_label("AssetClass")
ner.add_label("Sector")
ner.add_label("Base_Currency_Code")
ner.add_label("TacClassification")
textcat.add_label('Select')
textcat.add_label('Ratio')

nlp.begin_training()

# Loop for 10 iterations
for itn in range(40):
    # Shuffle the training data
    random.shuffle(DATA)
    losses = {}

    # Batch the examples and iterate over them
    for batch in spacy.util.minibatch(DATA, size=2):
        texts = [text for text, entities in batch]
        annotations = [entities for text, entities in batch]

        # Update the model
        nlp.update(texts, annotations, losses=losses)
        nlp.update(texts, annotations, losses=losses)
    print(losses)


test_text = 'I want the ratio of usd equity over all equity'
doc = nlp(test_text)
print(test_text, doc.cats,doc.ents)