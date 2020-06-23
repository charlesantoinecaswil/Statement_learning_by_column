class Parameter_class():

    def __init__(self, parameter_name,link_parameter_txt):

        self.parameter_name = parameter_name
        self.parameter_txt_version = '[' + parameter_name + ']'
        self.parameters_link = link_parameter_txt


    def get_value(self):

        with open(self.parameters_link) as parameters:

           for lines in parameters:
               if self.parameter_txt_version in lines:
                   actual_value = lines.split(']')[1]
                   actual_value = actual_value.replace('\n','')
                   return actual_value

    
    def change_value_to(self,New_Value):

        with open(self.parameters_link,'r+') as parameters:
            new_txt = str()
            for lines in parameters:
                if self.parameter_txt_version in lines:
                    new_txt += self.parameter_txt_version + New_Value + '\n'

                else:
                    new_txt += lines


            parameters.close()

        with open(self.parameters_link,'w+') as parameters:
            parameters.write(new_txt)
            parameters.close()


class integer(Parameter_class):

    def __init__(self, parameter_name,link_parameter_txt, min_digits,max_digits):

        super().__init__(parameter_name, link_parameter_txt)
        self.only_digits = True
        self.min_digits = min_digits
        self.max_digits = max_digits


    def is_valid(self, list_elements):

        for i in list_elements:

            try:
                int(i)
            except ValueError:
                continue

            if int(self.min_digits) < int(i) < int(self.max_digits):

                print('You want to change ',self.parameter_name, 'from ',self.get_value(), 'to', i)

                text = 'You want to change ' + self.parameter_name + 'from '+ str(self.get_value()) + 'to ' + str(i)
                return [text,i]


    

if __name__ == "__main__":
        
    TEST = integer('SelectGroupe','C:\\Users\\Charles-Antoine\\Desktop\\CharlesAntoine_20200120\\app\\parameters.txt', 1000, 10000)
    TEST.get_value()
    TEST.change_value_to('4501')
    TEST.is_valid(['4701','salut'])