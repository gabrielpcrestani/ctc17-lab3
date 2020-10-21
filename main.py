from util import PreProcessing 
from tree import Node     
from decision_tree import DecisionTree, MajorityValue, TestData    


pre_processing = PreProcessing('accident_data.csv')
data = pre_processing.read_csv()
entropia_S = pre_processing.entropy(data)
print(entropia_S)
#print(accident_level_values)
training_data, test_data = pre_processing.separate_data()
print(len(data))
print(len(training_data))
print(len(test_data))

attributes_data = ['Country', 'Local', 'Industry Sector', 'Potential Accident Level', 'Genre', 'Employee ou Terceiro', 'Risco Critico']
#data_gain, data_gain_array = pre_processing.gain('Data')
#country_gain, country_gain_array = pre_processing.gain('Country')
#local_gain, local_gain_array = pre_processing.gain('Local')
#ector_gain, sector_gain_array = pre_processing.gain('Industry Sector')
#potential_gain, potential_gain_array = pre_processing.gain('Potential Accident Level')
#genre_gain, genre_gain_array = pre_processing.gain('Genre')
#employee_gain, employee_gain_array = pre_processing.gain('Employee ou Terceiro')
#risco_gain, risco_gain_array = pre_processing.gain('Risco Critico')


#print(pre_processing.gain('Data'))
#print()
print(pre_processing.gain('Country'))
print()
print(pre_processing.gain('Local'))
print()
print(pre_processing.gain('Industry Sector'))
print()
print(pre_processing.gain('Potential Accident Level'))
print()
print(pre_processing.gain('Genre'))
print()
print(pre_processing.gain('Employee ou Terceiro'))
print()
print(pre_processing.gain('Risco Critico'))

tree = DecisionTree(training_data, attributes_data, MajorityValue(training_data))

#for edge in tree.edges:
#    print(edge.father, edge.child, edge.value)
tree.printTree(tree)

print("Porcentagem de acerto: ", TestData(tree, test_data))
