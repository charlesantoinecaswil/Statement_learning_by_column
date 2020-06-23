import pickle
import os

epoch = 250
batch = 1
number_of_word = 2000
lstm_out = 100
embed_dim = 100
max_question_length = 30
number_of_statement = 2

path = __file__.replace('Utils\\Constants.py','')




with open(path + '\\Saved_dict\\model_parameter.pickle','rb') as handle:

    model_parameter  = pickle.load(handle)
    model_path = 'Saved_Models\\' + model_parameter['model_name']
    tokenize_path = 'Saved_dict\\' + model_parameter['tokenize_word']
    App_path = model_parameter['App_path']
    path_meta_meta = App_path + '\\META_META.xlsx'
    path_xml = App_path + '\\xml\\'
    parameters_path = App_path + '\\parameters.txt'


try:

    with open(path + '\\Saved_dict\\parameters_dict.pickle','rb') as handle:

        dict_parameters = pickle.load(handle)

except FileNotFoundError:

    dict_parameters = dict()








## Ratio predicteur complet variable

path_output = 'C:\\Users\\Charles-Antoine Pare\\Documents\\AI_Package_Ratio\\AI_Package_Ratio\\app\\result'
xml_path = 'C:\\Users\\Charles-Antoine Pare\\Documents\\AI_Package_Ratio\\AI_Package_Ratio\\app\\xml'
ratio_csv_path = "C:\\Users\\Charles-Antoine Pare\\Documents\\Statement_learning\\ratio_data.csv"





if __name__ == "__main__":
    print(dict_parameters)
    print(path_xml)