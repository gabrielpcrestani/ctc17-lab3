from statistics import mode
from tree import Node, Edge
from util import PreProcessing

def DecisionTree(examples = [], attributes = [], standard = ''):
    if not examples:
        return standard
    elif SameClassification(examples):
        return examples[0]['Accident Level']
    elif not attributes:
        return MajorityValue(examples)
    else:
        best = ChooseAttribute(attributes, examples)
        #print(best)
        #print(attributes)
        tree = Node(best)
        m = MajorityValue(examples)

        different_values = DifferentValues(best, examples)
        #print(different_values)
        attributes.pop(attributes.index(best))
        #attributes_minus_best = attributes.pop(attributes.index(best))
        print(attributes)

        for v_i in different_values:
            examples_i = BestEqualsVi(examples, best, v_i)

            #print(examples_i)
            #print(attributes_minus_best)
            #print(m)
            sub_tree = DecisionTree(examples_i, attributes, m)
            tree.add(Edge(tree, sub_tree, v_i))
    return tree


def SameClassification(examples):
    classification = examples[0]['Accident Level']
    for example in examples:
        if (example['Accident Level'] != classification):
            return False

    return True

def MajorityValue(examples):
    classifications = []
    for example in examples:
        classifications.append(example['Accident Level'])
    
    return mode(classifications)

def ChooseAttribute(attributes, examples):
    attributes_gains = []
    pre_processing = PreProcessing('accident_data.csv')
    
    data = pre_processing.read_csv()
    entropia_S = pre_processing.entropy(data)

    training_data, test_data = pre_processing.separate_data()
    
    for attribute in attributes:
        attributes_gains.append(pre_processing.gain(attribute))

    return attributes[attributes_gains.index(max(attributes_gains))]
    #country_gain, country_gain_array = pre_processing.gain('Country')
    #local_gain, local_gain_array = pre_processing.gain('Local')
    #sector_gain, sector_gain_array = pre_processing.gain('Industry Sector')
    #potential_gain, potential_gain_array = pre_processing.gain('Potential Accident Level')
    #genre_gain, genre_gain_array = pre_processing.gain('Genre')
    #mployee_gain, employee_gain_array = pre_processing.gain('Employee ou Terceiro')
    #risco_gain, risco_gain_array = pre_processing.gain('Risco Critico')

def DifferentValues(best, examples):
    values = []
    for example in examples:
        if example[best] not in values:
            values.append(example[best])

    return values 

def BestEqualsVi(examples, best, v_i):
    examples_equals_vi = []
    for example in examples:
        if example[best] == v_i:
            examples_equals_vi.append(example)

    return examples_equals_vi

def TestData(tree, test_data):
    num_correct = 0
    num_test_data = len(test_data)

    for test in test_data:
        edges = tree.edges
        while (True):
            for edge in edges:
                if test[edge.father.attribute] == edge.value:
                    break
            if (edge.child == 'I' or edge.child == 'II' or edge.child == 'III' or edge.child == 'IV' or edge.child == 'V' or edge.child == 'VI'): 
                if edge.child == test['Accident Level']:
                    num_correct += 1
                break
            edges = edge.child.edges

    return num_correct / num_test_data
