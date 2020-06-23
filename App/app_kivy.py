import kivy
from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.scrollview import ScrollView
from kivy.uix.dropdown import DropDown
from kivy.uix.button import Button
from kivy.base import runTouchApp
from kivy.uix.gridlayout import GridLayout
from kivy.uix.textinput import TextInput
from kivy.uix.widget import Widget
from kivy.properties import ObjectProperty
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.image import Image
from kivy.uix.behaviors import ButtonBehavior
import os
import webbrowser
import getpass
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from Utils.Constants import *
from Caswil_AI.Statement_Predictor import Select_Statement
from Caswil.Parameter_class import Parameter_class, integer
from Caswil.Statement_class import Statement_class
from Caswil.Function.RECHERCHE_EXECUTER import RECHERCHE_EXECUTER




# Link Package https://www.youtube.com/watch?v=gt-qndBYrCQ




class MainWindow(Screen):


    def __init__(self, **kwargs):
        super(Screen,self).__init__(**kwargs)

        
        
        


    def generate(self):
        
        self.path = __file__
        

        

class SecondWindow(Screen):

    def __init__(self, **kwargs):
        super(Screen,self).__init__(**kwargs)
        self.path = __file__
        self.User = Select_Statement(epoch, batch, lstm_out, number_of_word, embed_dim, max_question_length, number_of_statement)
        self.User.load_tokenizer(tokenize_path)
        self.User.load_model(model_path)
        self.current_index_parameter = 0 
        



    def INFO(self, question,index):

        self.Question = question
        self.Question_split = self.Question.split(' ')
        self.Result = self.User.ask_question(self.Question)
        self.result.text = 'Open : ' + self.Result
        self.current_index_parameter = index


    def next_question(self):


        if self.current_index_parameter != 0:
            try:
                dict_parameters[list(dict_parameters.keys())[self.current_index_parameter-1]].change_value_to(self.parameter_to_change)
            except IndexError and TypeError:
                pass


        


        
        j = True
        while j == True:
            try:


                try:
                    print('YES')
                    text = dict_parameters[list(dict_parameters.keys())[self.current_index_parameter]].is_valid(self.Question_split)
                    print(text)
                    print(self.current_index_parameter)
                    self.result.text = dict_parameters[list(dict_parameters.keys())[self.current_index_parameter]].is_valid(self.Question_split)[0]
                    self.parameter_to_change = dict_parameters[list(dict_parameters.keys())[self.current_index_parameter]].is_valid(self.Question_split)[1]
                    self.current_index_parameter += 1
                    j=False

                except ValueError:
                    self.current_index_parameter += 1


                
            except (IndexError, TypeError):
                j = False
                self.open_statement()
                self.result.text = 'Go back to search screen'

 

    def open_statement(self):
        
        RECHERCHE_EXECUTER(App_path, self.Result.replace('"','')[0:])
        webbrowser.open(App_path + '/html/'+ getpass.getuser() +'/RECHERCHE.html',new = 0, autoraise = True)
        
        






class ThirdWindow(Screen):

    def __init__(self, **kwargs):
        super(Screen,self).__init__(**kwargs)


    def generate_statement(self,list_statement):

        list_statement = list(list_statement.keys())

        for i in list_statement:
            btn = Button(text=i)
            btn.bind(on_press=self.press)
            self.statement.add_widget(btn)

    
    def press(self, button):
        self.selected_file.text = (button.text).replace('"','')[0:]

        Statement_to_open = Statement_class(path_xml + self.selected_file.text)
        self.parameter_grid.clear_widgets()
        Open_button = Button(text ='Open statement',pos_hint ={'x':.5, 'y':.65},size_hint = (0.3,0.08),on_press = self.open_statement)
        self.main_page.add_widget(Open_button)

        for i in Statement_to_open.get_all_parameters_involved(path_meta_meta):

            if i in dict_parameters.keys():

                
                new_label = Label(text=i)
                test1 = TextInput(text=dict_parameters[i].get_value(),multiline=False,id=i,on_text_validate =self.on_enter)
                self.parameter_grid.add_widget(new_label)
                self.parameter_grid.add_widget(test1)


    def on_enter(self,instance = 'test'):
        
        print(instance)
        try:
            dict_parameters[instance.id].change_value_to(instance.text)

        except AttributeError:
            pass
       



    def open_statement(self,button):
        

        RECHERCHE_EXECUTER(App_path, self.selected_file.text)
        webbrowser.open(App_path + '/html/'+ getpass.getuser() +'/RECHERCHE.html',new = 0, autoraise = True)







        
        
       



class WindowManager(ScreenManager):
    pass
        

sm = WindowManager()
sm.add_widget(MainWindow(name='main'))
sm.add_widget(SecondWindow(name='second'))
sm.add_widget(ThirdWindow(name='third'))



    
kv = Builder.load_file("App\\mynew.kv") 

    
    
class MyMainApp(App):
    def build(self):
        sm = WindowManager()
        sm.add_widget(MainWindow(name='main'))
        sm.add_widget(SecondWindow(name='second'))
        sm.add_widget(ThirdWindow(name='third'))
        return sm


if __name__ == "__main__":

    MyMainApp().run()

