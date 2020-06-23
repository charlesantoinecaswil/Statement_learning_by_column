import pickle



if __name__ == "__main__":
    
    model_name = input('Enter the name you want to give to your model') +'.bin'
    tokenize_word = input('Enter the name you want to give to word bank') +'.bin'
    App_path = input('Enter the name of the app path')


    dict_model = dict()
    dict_model['model_name'] = model_name
    dict_model['tokenize_word'] = tokenize_word
    dict_model['App_path'] = App_path
    dict_parameter = dict()


    with open('Saved_dict\\model_parameter.pickle','wb') as handle:
        pickle.dump(dict_model, handle, protocol=pickle.HIGHEST_PROTOCOL)
