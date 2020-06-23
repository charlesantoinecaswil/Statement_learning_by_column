

class variable():

    def __init__(self, name):

        self.name = name

    
    def type(self):

        pass


class  integer_variable(variable):
    
    def __init__(self, name):
        super().__init__(name)
        self.type = "integertype"

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
         



class  string_variable(variable):
    
    def __init__(self,name):
        super().__init__(name)
        self.type = "stringtype"


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

data_preparation("Donne moi le ratio de FRANCHISE_ASSURE_APPLIQUEE_MALADIE par CLASSE selon le GROUPE",[integer_variable("FRANCHISE_ASSURE_APPLIQUEE_MALADIE"),integer_variable("CLASSE"),integer_variable("GROUPE")],"CLASSE;GROUPE;FRANCHISE_ASSURE_APPLIQUEE_MALADIE")

## Data cleaning


