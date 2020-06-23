import xml.etree.ElementTree as ET


class recherche_xml():


    def __init__(self,xml_path):

        self.xml_path = xml_path
        self.tree = ET.parse(xml_path)
        self.root = self.tree.getroot()

    def get_sources(self):

        source_tmp = list()

        for column in self.root.findall("./statement/sources/source/connection"):
            try:
                source_tmp.append((column.attrib['filename'].split('xml\\'))[-1])

            except AttributeError:
                    continue

        return source_tmp



    def change_source(self,new_source):


        for column in self.root.findall("./statement/sources/source/connection"):
            try:
                column.set("filename","xml\\"+ new_source)

            except AttributeError:
                    continue

        self.tree.write(self.xml_path)

    def delete_columns(self):

        self.root[0][1].clear()
        self.tree.write("test.xml")


    def add_column(self,column_name):

        attrib_column = {'groupby':'select',"name":column_name,"visible":""}
        attrib_block = {'value':'[NewSrc1].' + column_name ,"display":'[NewSrc1].' + column_name}

        element  = self.root[0][1].makeelement('column', attrib_column)

        self.root[0][1].append(element)

        for i in self.root[0][1]:
            if i.attrib == attrib_column:
                element = 

        self.tree.write("test.xml")








a = recherche_xml("C:\\Users\\Charles-Antoine\\Desktop\\CharlesAntoine_20200120\\app\\xml\\RECHERCHE.Caswil.ST.Xml")

a.get_sources()
a.delete_columns()

a.add_column("TEST")

a.change_source("UV_GroupList_Franchises_final.Caswil.ST.Xml")