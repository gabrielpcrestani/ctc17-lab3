from util import PreProcessing 
from tree import Node     
from decision_tree import DecisionTree, MajorityValue, TestDataDecisionTree, TestDataAPriori   

pre_processing = PreProcessing('accident_data.csv')
data = pre_processing.read_csv()
entropia_S = pre_processing.entropy(data)
training_data, test_data = pre_processing.separate_data()
print("Tamanho do set de dados: ", len(data))
print("Tamanho do set de treinamento: ", len(training_data))
print("Tamanho do set de teste:", len(test_data))
print()

attributes_data = ['Country', 'Local', 'Industry Sector', 'Potential Accident Level', 'Genre', 'Employee ou Terceiro', 'Risco Critico']

#print(pre_processing.gain('Data'))
#print()
print("Entropia(S): ", entropia_S)
for attribute in attributes_data:
    print("Ganho(S," + attribute + "): ", pre_processing.gain(attribute))
print()

tree = DecisionTree(training_data, attributes_data, MajorityValue(training_data))
#for edge in tree.edges:
#    print(edge.father, edge.child, edge.value)
tree.printTree(tree)

hit_rate_dt, confusion_matrix_dt, mean_square_error_dt, kappa_statistic_dt = TestDataDecisionTree(tree, test_data)
print("--- DECISION TREE ---")
print("Taxa de acerto: ", hit_rate_dt)
print("Matriz de confusao: ")
for row in confusion_matrix_dt:
    print("\t", row)
print("Erro quadratico medio: ", mean_square_error_dt)
print("Estatistica kappa: ", kappa_statistic_dt)

print("\n--- A PRIORI ---")
hit_rate_ap, confusion_matrix_ap, mean_square_error_ap, kappa_statistic_ap = TestDataAPriori(test_data)
print("Taxa de acerto: ", hit_rate_ap)
print("Matriz de confusao: ")
for row in confusion_matrix_ap:
    print("\t", row)
print("Erro quadratico medio: ", mean_square_error_ap)
print("Estatistica kappa: ", kappa_statistic_ap)
