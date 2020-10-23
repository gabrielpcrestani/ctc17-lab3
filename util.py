import csv
from random import randint
from math import log2

class PreProcessing:
    def __init__(self, file_name):
        self.file_name = file_name
        self.data = []
        self.training_data = []
        self.test_data = []
        self.entropy_S = 0
        
    def read_csv(self):
        self.data = []
        with open(self.file_name, newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')
            for row in spamreader:
                list_temp = ', '.join(row).split(',')
                    
                accident = {
                    "Data": list_temp[0].strip(),
                    "Country": list_temp[1].strip(),
                    "Local": list_temp[2].strip(),
                    "Industry Sector": list_temp[3].strip(),
                    "Accident Level": list_temp[4].strip(),
                    "Potential Accident Level": list_temp[5].strip(),
                    "Genre": list_temp[6].strip(),
                    "Employee ou Terceiro": list_temp[7].strip(),
                    "Risco Critico": list_temp[8].strip()
                }
                self.data.append(accident)
        del self.data[0]

        return self.data.copy()

    def separate_data(self):
        data_initial_size = len(self.data)
        test_data_size = int(round(0.2 * data_initial_size , 0))
        for i in range(test_data_size):
            temp = randint(0, len(self.data)-1)
            self.test_data.append(self.data[temp].copy()) 
            del self.data[temp]
        for i in range(data_initial_size - test_data_size):
            self.training_data.append(self.data[i])
        self.read_csv()

        return self.training_data.copy(), self.test_data.copy()

    def entropy(self, data_array):
        # Accident level tem 6 valores possiveis
        accident_level_values = [{'accident_level': 'I', 'count': 0}, {'accident_level': 'II', 'count': 0}, 
        {'accident_level': 'III', 'count': 0}, {'accident_level': 'IV', 'count': 0}, {'accident_level': 'V', 'count': 0}
        , {'accident_level': 'VI', 'count': 0}]
        
        for elem in data_array:
            if elem["Accident Level"] == 'I':
                accident_level_values[0]['count'] += 1
            elif elem["Accident Level"] == 'II':
                accident_level_values[1]['count'] += 1
            elif elem["Accident Level"] == 'III':
                accident_level_values[2]['count'] += 1
            elif elem["Accident Level"] == 'IV':
                accident_level_values[3]['count'] += 1
            elif elem["Accident Level"] == 'V':
                accident_level_values[4]['count'] += 1
            elif elem["Accident Level"] == 'VI':
                accident_level_values[5]['count'] += 1

        #print(len(data_array))
        #print(accident_level_values)
        
        entropy_S = 0

        for elem in accident_level_values:
            if elem['count'] != 0:
                entropy_S -= (elem['count']/len(data_array)) * log2(elem['count']/len(data_array))

        return entropy_S

    def gain(self, attribute):
        temp_array = []
        exists = False

        self.entropy_S = self.entropy(self.data)

        for elem in self.data:
            value = elem[attribute]
            exists = False
            for elem_temp in temp_array:
                if elem_temp['value'] == value:
                    elem_temp['count'] += 1
                    exists = True
                    break
            if (not exists):
                temp_array.append({'value': value, 'count': 1, 'entropy': 0})

        temp_value_array = []
        for elem in temp_array:
            temp_value_array = []
            for row in self.data:
                if row[attribute] == elem['value']:
                    temp_value_array.append(row.copy())
            elem['entropy'] = self.entropy(temp_value_array)

        gain_attribute = self.entropy_S
        for elem in temp_array:
            gain_attribute -= (elem['count'] / len(self.data)) * elem['entropy']

        #return gain_attribute, temp_array.copy()
        return gain_attribute
