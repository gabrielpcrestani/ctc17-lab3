class Node:
    def __init__(self, attribute):
        self.edges = []
        self.attribute = attribute  

    def add(self, edge):
        self.edges.append(edge)    

    def printTree(self, node):
        print("Edge father attribute: ", node.attribute)
        for edge in node.edges:
            if isinstance(edge.child, str):
                print("Edge father: ", edge.father.attribute)
                print("Edge value: ", edge.value)
                print("Edge child: ", edge.child)
                print()
            else:
                print("Value: ", edge.value, "; Son:", edge.child.attribute)
                self.printTree(edge.child)
    
class Edge:
    def __init__(self, father, child, value):
        self.father = father
        self.child = child
        self.value = value
