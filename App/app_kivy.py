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


# Link Package https://www.youtube.com/watch?v=gt-qndBYrCQ




class MainWindow(Screen):


    def __init__(self, **kwargs):
        super(Screen,self).__init__(**kwargs)

        
        
        


    

        

class SecondWindow(Screen):

    def __init__(self, **kwargs):
        super(Screen,self).__init__(**kwargs)
        
        



    def INFO(self, question,index):
        pass

        


    def next_question(self):
        pass


      
       




class WindowManager(ScreenManager):
    pass
        



sm = WindowManager()
sm.add_widget(MainWindow(name='main'))
sm.add_widget(SecondWindow(name='second'))




    
kv = Builder.load_file("mynew.kv") 

    
    
class MyMainApp(App):
    def build(self):
        sm = WindowManager()
        sm.add_widget(MainWindow(name='main'))
        sm.add_widget(SecondWindow(name='second'))
        
        return sm


if __name__ == "__main__":

    MyMainApp().run()

