from Caswil_AI.Column_Finder import column_finder
from Caswil_AI.Ratio_Predictor import define_variable_ratio, variable, integer_variable, string_variable
from Utils.Constants import *




if __name__ == "__main__":

    question = 'Je veux le ratio Equity pour le CLIENT_065'

    User = column_finder(path_output, xml_path)
    aa = User.create_dictionnary()
    User.create_model_data()
    User.load_model('C:\\Users\\Charles-Antoine Pare\\Documents\\Statement_learning_by_column\\Saved_Models\\new_model_ratio.bin')
    
    User.ask_question(question,2)
    dict_result, list_result = User.ask_question('Je veux le ratio Equity pour le CLIENT_065',2)


    epoch = 250
    batch = 1
    number_of_word = 2000
    lstm_out = 100
    embed_dim = 100
    max_question_length = 30
    number_of_statement = 34
    bank_of_columns = [integer_variable("GROUPE"),integer_variable("CLASSE"),string_variable("Type_de_list_EN"),string_variable("Nom_Anglais"),integer_variable("FRANCHISE_ASSURE_APPLIQUEE_MALADIE")]


    for i in dict_result.values():
        if i['type'] == 'object':
            bank_of_columns.append(string_variable(i['Part_of_text']))
        else:
            bank_of_columns.append(integer_variable(i['Part_of_text']))



    Ratio_determiner = define_variable_ratio(epoch, batch, lstm_out, number_of_word, embed_dim, max_question_length, number_of_statement,bank_of_columns)
    Ratio_determiner.create_tokenizer()
    Ratio_determiner.load_data("C:\\Users\\Charles-Antoine Pare\\Documents\\Statement_learning\\ratio_data.csv")
    Ratio_determiner.create_model()
    Ratio_determiner.train_model(ratio_csv_path)
    Ratio_determiner.ask_question(question)

for i in dict_result.values():
    if i['type'] == 'object':
        bank_of_columns.append(string_variable(i['Part_of_text']))
    else:
        bank_of_columns.append(integer_variable(i['Part_of_text']))
