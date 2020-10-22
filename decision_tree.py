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
        #print(attributes)

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

def TestDataDecisionTree(tree, test_data):
    all_elements_sum = 0
    hit_rate = 0
    confusion_matrix = \
    [[0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0]]
    p_o = 0
    p_e = 0
    kappa_statistic = 0
    mean_square_error = 0

    for test in test_data:
        edges = tree.edges
        while (True):
            for edge in edges:
                if test[edge.father.attribute] == edge.value:
                    break
            if (edge.child == 'I' or edge.child == 'II' or edge.child == 'III' or edge.child == 'IV' or edge.child == 'V' or edge.child == 'VI'): 
                confusion_matrix[roman_to_int(test['Accident Level']) - 1][roman_to_int(edge.child) - 1] += 1
                break
            edges = edge.child.edges

    for i in range(len(confusion_matrix)):
        for j in range(len(confusion_matrix)):
            all_elements_sum += confusion_matrix[i][j]
        hit_rate += confusion_matrix[i][i]
    hit_rate /= all_elements_sum

    distribution = [0.75, 0.09, 0.07, 0.07, 0.02, 0]
    confusion_matrix_expected_distribution = \
    [[0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0]]
    for i in range(len(confusion_matrix_expected_distribution)):
        for j in range(len(confusion_matrix_expected_distribution)):
            confusion_matrix_expected_distribution[i][j] = round(distribution[j] * sum(confusion_matrix[i]), 2)
    #for row in confusion_matrix_expected_distribution:
    #    print(row)
    p_o = hit_rate
    for i in range(len(confusion_matrix_expected_distribution)):
        p_e += confusion_matrix_expected_distribution[i][i]
    p_e /= all_elements_sum
    kappa_statistic = round((p_o - p_e) / (1 - p_e), 2)

    temp = 0
    for j in range(len(confusion_matrix)):
        temp = sum(confusion_matrix[j])
        for i in range(len(confusion_matrix)):
            temp -= confusion_matrix[i][j]
        mean_square_error += temp ** 2
    mean_square_error /= len(confusion_matrix)
    mean_square_error = round(mean_square_error, 4)

    return hit_rate, confusion_matrix, mean_square_error, kappa_statistic

def TestDataAPriori(test_data):
    classifications_mode = MajorityValue(test_data)

    all_elements_sum = 0
    hit_rate = 0
    confusion_matrix = \
    [[0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0]]
    p_o = 0
    p_e = 0
    kappa_statistic = 0
    mean_square_error = 0

    for test in test_data:
        confusion_matrix[roman_to_int(test['Accident Level']) - 1][roman_to_int(classifications_mode) - 1] += 1

    for i in range(len(confusion_matrix)):
        for j in range(len(confusion_matrix)):
            all_elements_sum += confusion_matrix[i][j]
        hit_rate += confusion_matrix[i][i]
    hit_rate /= all_elements_sum

    distribution = [0.75, 0.09, 0.07, 0.07, 0.02, 0]
    confusion_matrix_expected_distribution = \
    [[0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0]]
    for i in range(len(confusion_matrix_expected_distribution)):
        for j in range(len(confusion_matrix_expected_distribution)):
            confusion_matrix_expected_distribution[i][j] = round(distribution[j] * sum(confusion_matrix[i]), 2)
    #for row in confusion_matrix_expected_distribution:
    #    print(row)
    p_o = hit_rate
    for i in range(len(confusion_matrix_expected_distribution)):
        p_e += confusion_matrix_expected_distribution[i][i]
    p_e /= all_elements_sum
    kappa_statistic = round((p_o - p_e) / (1 - p_e), 2)

    temp = 0
    for j in range(len(confusion_matrix)):
        temp = sum(confusion_matrix[j])
        for i in range(len(confusion_matrix)):
            temp -= confusion_matrix[i][j]
        mean_square_error += temp ** 2
    mean_square_error /= len(confusion_matrix)
    mean_square_error = round(mean_square_error, 4)

    return hit_rate, confusion_matrix, mean_square_error, kappa_statistic

def roman_to_int(input):
    input = input.upper( )
    nums = {'M':1000,
            'D':500,
            'C':100,
            'L':50,
            'X':10,
            'V':5,
            'I':1}
    sum = 0
    for i in range(len(input)):
        value = nums[input[i]]
        if i+1 < len(input) and nums[input[i+1]] > value:
            sum -= value
        else: 
            sum += value
        
    return sum

