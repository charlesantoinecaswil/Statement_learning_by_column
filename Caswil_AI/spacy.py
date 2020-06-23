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
import pickle


class column_finder():

    def __init__(self, path_output, xml_path,max_length,pickle_path = 100):

        self.csv_output_path = path_output ## Les csv output dans le result folder
        self.xml_path = xml_path ## path des statement
        self.data_statement_dict = self.create_dictionnary()
        self.max_length = max_length ## Pas important présentement
        self.picke_path = pickle_path


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
                    pattern.append({'LOWER': word})

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
                        pattern.append({'IS_DIGIT': False})


                    except ValueError:
                        pattern.append({'LOWER': word})

                    dict_pattern[column].append(pattern)

        return dict_pattern

    def return_sentence_pattern(self, sentence):

        pattern_dict = self.pattern_dict()
        doc = nlp(sentence.replace(',',' '))
        dict_snswer = dict()
        dict_snswer['entities'] = list()
        for entities in pattern_dict.keys():
            print(entities)
            matcher = Matcher(nlp.vocab)
            matcher.add(entities,pattern_dict[entities])
            spans = [doc[start:end] for match_id, start, end in matcher(doc)]
            entitie = [(span.start_char, span.end_char, entities) for span in spans]
            dict_snswer['entities'] += (entitie)

        return dict_snswer



    def return_sentence_pattern(self, sentence):

        pattern_dict = self.pattern_dict()
        sentence = sentence.lower()
        doc = nlp(sentence.replace(',',' '))
        dict_snswer = dict()
        dict_snswer['entities'] = list()
        for entities in pattern_dict.keys():
            print(entities)
            matcher = Matcher(nlp.vocab)
            matcher.add(entities,pattern_dict[entities])
            spans = [doc[start:end] for match_id, start, end in matcher(doc)]
            entitie = [(span.start_char, span.end_char, entities) for span in spans]
            dict_snswer['entities'] += (entitie)


        save_observation(sentence)
        return dict_snswer



    def save_observation(self, sentence,new_dict):

        try:
            file = open(self.picke_path,'rb')

            train_data_complet = pickle.load(file)
            train_data_complet.append((sentence, new_dict))

            file.close()
        except FileNotFoundError:
            train_data_complet = [(sentence, new_dict)


        file = open(self.picke_path,'wb')

        pickle.dump(train_data_complet,file)

        file.close()



User.return_sentence_pattern('I coca-cola the value of USD equity for the client 45')




new_dict1 = {'test':1}
new_dict2 = {'test':2}


try:
    file = open(pickle_path,'rb')
    full_data = pickle.load(file)
    file.close()
    full_data.append(new_dict2)

except FileNotFoundError:
    full_data = [new_dict1]


file = open(pickle_path,'wb')
pickle.dump(full_data,file)
file.close()


file = open(pickle_path,'rb')

full_data = pickle.load(file)





        



                    








path_output = 'C:\\Users\\Charles-Antoine Pare\\Documents\\AI_Package_Ratio\\AI_Package_Ratio\\app\\result'
xml_path = 'C:\\Users\\Charles-Antoine Pare\\Documents\\AI_Package_Ratio\\AI_Package_Ratio\\app\\xml'
pickle_path = 'C:\\Users\\Charles-Antoine Pare\\Documents\\AI_Package_Ratio\\AI_Package_Ratio\\test.pkl'
User = column_finder(path_output, xml_path)
User.create_dictionnary()
test = User.pattern_dict()
User.return_sentence_pattern('I coca-cola the value of USD equity for the client 45')

patternaccountname = [{"LOWER": "client"}, {"IS_DIGIT": True}]
patternaccountname1 = [{"LOWER": "client"}, {"IS_DIGIT": False}]
patternaccountname2 = [{"LOWER": "clients"}, {"IS_DIGIT": False}]
patterncurrency1 = [{"LOWER": "eur"}]
patterncurrency2 = [{"LOWER": "cad"}]
patterncurrency3 = [{"LOWER": "usd"}]
patterncurrency4 = [{"LOWER": "a2d"}]
patternAssetClass1 = [{"LOWER": "cash"},{"LOWER": "and"},{"LOWER": "cash"},{"LOWER": "equivalents"}]
patternAssetClass2 = [{"LOWER": "equity"}]
patternAssetClass3 = [{"LOWER": "fixed"}, {"LOWER": "income"}]
patternAssetClass4 = [{"LOWER": "alternative"}, {"LOWER": "investments"}]
patterSector1 = [{"LOWER": "mutual"}, {"LOWER": "funds"}]
patterSector2 = [{"LOWER": "hedge"}, {"LOWER": "funds"}]
patterSector3 = [{"LOWER": "federal"}, {"LOWER": "government"}]

pattern_list = {'accountname':[patternaccountname,patternaccountname1,patternaccountname2],'currency':[patterncurrency1,patterncurrency2,patterncurrency3,patterncurrency4],'AssetClass':[patternAssetClass1,patternAssetClass2,patternAssetClass3,patternAssetClass4],'Sector':[patterSector1, patterSector2, patterSector3]}

TEXTS = ['I want the value of A2D equity for the client 45', 'Give me all my CAD fixed income for all my clients', 'I want to see client 90 portfolio', 'What are the investement in mutual funds for client 344', "what are the differences between client 31 and client 98",'I want all the clients with more USD equity than CAD','Give me the USD','Give me the equity']
TEXTS = [i.lower() for i in TEXTS]

nlp = English()
matcher = Matcher(nlp.vocab)
doc = nlp('je veux le prix en USD deplus de 2000 dollars')


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
nlp.add_pipe(ner)
ner.add_label("accountname")
ner.add_label("currency")
ner.add_label("AssetClass")
ner.add_label("Sector")

nlp.begin_training()

# Loop for 10 iterations
for itn in range(40):
    # Shuffle the training data
    random.shuffle(TRAINING_DATA)
    losses = {}

    # Batch the examples and iterate over them
    for batch in spacy.util.minibatch(TRAINING_DATA, size=2):
        texts = [text for text, entities in batch]
        annotations = [entities for text, entities in batch]

        # Update the model
        nlp.update(texts, annotations, losses=losses)
    print(losses)


doc = nlp('I want the value of vsd EQUITY for the client 45'.lower())

for ent in doc.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)