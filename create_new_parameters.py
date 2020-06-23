
from Caswil.Parameter_class import integer, Parameter_class
from Utils.Constants import *
import pickle




def create_new_parameter_int(parameter_name, max_len, min_len, dict_parameters):

    dict_parameters[parameter_name] = integer(parameter_name, parameters_path,min_len,max_len)







if __name__ == "__main__":

    parameter_name = input('Enter the parameter name: ')
    max_len = int(input('Enter max value of the parameter: '))
    min_len = int(input('Enter min value of the parameter: '))
    

    try:
        with open('Saved_dict\\parameters_dict.pickle', 'rb') as handle:
            dict_parameters = pickle.load(handle)

        create_new_parameter_int(parameter_name, max_len, min_len, dict_parameters)

        with open('Saved_dict\\parameters_dict.pickle', 'wb') as handle:
            pickle.dump(dict_parameters, handle, protocol=pickle.HIGHEST_PROTOCOL)
        

    except FileNotFoundError:
        dict_parameters = dict()

        create_new_parameter_int(parameter_name, max_len, min_len, dict_parameters)

        with open('Saved_dict\\parameters_dict.pickle', 'wb') as handle:
            pickle.dump(dict_parameters, handle, protocol=pickle.HIGHEST_PROTOCOL)



