from Caswil.Parameter_class import Parameter_class, integer
from Caswil.Statement_class import Statement_class
from Caswil_AI.Statement_Predictor import Select_Statement
from Utils.Constants import *
import pickle


## Load all the other constant unique to this model





if __name__ == "__main__":
    Question = input('Que voulez-vous')
    Question_split = Question.split(' ')
    

    print(tokenize_path)

    User = Select_Statement(epoch, batch, lstm_out, number_of_word, embed_dim, max_question_length, number_of_statement)
    try:
        User.load_tokenizer(tokenize_path)
        User.load_model(model_path)

    except FileNotFoundError:
        User.create_tokenizer()
        User.create_model()
        train_data_path = input('Enter the path of your csv data to train the model :')
        User.load_data(train_data_path)
        User.train_model(train_data_path,15)
        User.save_model(model_path)
        User.save_tokenizer(tokenize_path)




    
    Statement = path_xml + User.ask_question(Question)
    Statement_to_open = Statement_class(Statement)
    list_parameters = Statement_to_open.get_all_parameters_involved(path_meta_meta)

    
    for j in list_parameters:
        if j in dict_parameters.keys():
            dict_parameters[j].is_valid(Question_split)
     