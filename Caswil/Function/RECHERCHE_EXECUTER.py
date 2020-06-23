import webbrowser
import xml.etree.ElementTree as ET
import os
import pandas as pd


## TEST ##

## Ce fichier modifie la view RECHERCHE pour que ça source devienne statement_a_ouvrir

## FIN TEST ##


def RECHERCHE_EXECUTER(path_app, statement_a_ouvrir):

    datapath = 'xml\\' + statement_a_ouvrir
    path_view_RECHERCHE = path_app + '\\xml\\views\\RECHERCHE.Caswil.VW.Xml'

    tree = ET.parse(path_view_RECHERCHE)
    root = root = tree.getroot()

    root[0][1][0][0][0].set('datapath',datapath)

    tree.write(path_view_RECHERCHE)






