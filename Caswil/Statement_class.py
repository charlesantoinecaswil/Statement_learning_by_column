import xml.etree.ElementTree as ET
import os
import pandas as pd



class Statement_class():

    def __init__(self, xml_path):

        self.xml_path = xml_path
        self.tree = ET.parse(self.xml_path)
        self.root = self.tree.getroot()
        self.xml = xml_path.split('\\')[-1]


    
    def get_sources(self):

        source_tmp = list()

        for column in self.root.findall("./statement/sources/source/connection"):
            try:
                source_tmp.append((column.attrib['filename'].split('xml\\'))[-1])

            except AttributeError:
                    continue

        return source_tmp


    def get_columns(self):

        colums_tmp = list()

        for column in self.root.findall("./statement/columns/column"):
            try:
                colums_tmp.append(column.attrib['name'])
            except AttributeError:
                continue

        return colums_tmp
            

    def xml_to_string(self):

         strtoxml = str(ET.tostring(self.root, encoding="utf-8", method="xml"))

         return strtoxml

    def get_all_parameters_involved(self,META_META_XLSX_PATH):
        print(self.xml)
        data_frame_meta_meta = pd.read_excel(META_META_XLSX_PATH)
        data_frame_meta_meta = list(data_frame_meta_meta[data_frame_meta_meta['Statement']==self.xml]['Parameters'])[0].replace('[parameter].','').replace('[','').replace(']','').replace('.','').replace("'",'').split(', ')
        return data_frame_meta_meta



if __name__ == "__main__":
    xml = 'C:\\Users\\Charles-Antoine\\Desktop\\CharlesAntoine_20200120\\app\\xml\\UV_GroupList_Franchises_final.Caswil.ST.Xml'
    TEST = Statement_class(xml)
    meta_meta_path = 'C:\\Users\\Charles-Antoine\\Desktop\\CharlesAntoine_20200120\\app\\META_META.xlsx'
    a = TEST.get_all_parameters_involved(meta_meta_path)
    print(a)