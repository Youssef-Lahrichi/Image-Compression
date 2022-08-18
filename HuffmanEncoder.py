
class node:
    def  __init__(self, value=0, priority=0, left=None, right=None):
        self.value = value
        self.priority = priority
        self.left = left
        self.right = right

    def __str__(self):
        return "V: " + str(self.value) + " P: " + str(self.priority)


class MinHeap:
    def __init__(self, array=[]):
        self.__array = array

    def __str__(self):
        output = []
        for n in self.__array:
            output.append(str(n))

        return str(output)

    def __getParentIndex(i):
        return int((i - 1)/2)

    def __getRightIndex(i):
        return 2*i+2

    def __getLeftIndex(i):
        return 2*i+1

    def isEmpty(self):
        return len(self.__array) == 0

    def size(self):
        return len(self.__array)
    
    def insert(self, value, priority):
        if type(value) is node:
            self.__array.append(value)
        else:
            self.__array.append(node(value, priority))
        self.__percolateUp(len(self.__array) - 1)

    def removeMin(self):
        if(self.isEmpty()):
            return
        
        lastLeaf = self.__array.pop(-1)
        if self.isEmpty():
            return lastLeaf
        
        root = self.__array[0]
        self.__array[0] = lastLeaf
        self.__percolateDown()
        return root

    def __percolateUp(self, lri):
        if len(self.__array) <= 1:
            return

        lpi = MinHeap.__getParentIndex(lri)
        lastLeaf = self.__array[lri]
        lastParent = self.__array[lpi]
        
        if lastLeaf.priority < lastParent.priority:
            self.__array[lpi] = lastLeaf
            self.__array[lri] = lastParent
            self.__percolateUp(lpi)
        else:
            return


    def __percolateDown(self, ri=0):
        lci = MinHeap.__getLeftIndex(ri)
        rci = MinHeap.__getRightIndex(ri)

        if lci >= len(self.__array) and rci >= len(self.__array):
            return
        elif lci >= len(self.__array):
            min_ci = rci
            min_p = self.__array[rci].priority
        elif rci >= len(self.__array):
            min_ci = lci
            min_p = self.__array[lci].priority
        elif self.__array[lci].priority < self.__array[rci].priority:
            min_ci = lci
            min_p = self.__array[lci].priority
        else:
            min_ci = rci
            min_p = self.__array[rci].priority

        if min_p < self.__array[ri].priority:
            temp = self.__array[ri]
            self.__array[ri] = self.__array[min_ci]
            self.__array[min_ci] = temp
            self.__percolateDown(min_ci)



def create_map_from_tree(node, v2s, s2v, sequence):
    if node.left is None:
        v2s[node.value] = sequence
        s2v[sequence] = node.value
        return

    create_map_from_tree(node.left, v2s, s2v, sequence + "0")
    create_map_from_tree(node.right, v2s, s2v, sequence + "1")

    
def huffmanCode(coeffFrequencyMap):
    thePQ = MinHeap()
    for coeff in coeffFrequencyMap:
        thePQ.insert(coeff, coeffFrequencyMap[coeff])

    while thePQ.size() != 1:
        left = thePQ.removeMin()
        right = thePQ.removeMin()

        newPriority = left.priority + right.priority
        newNode = node(0, newPriority, left, right)
        thePQ.insert(newNode, newPriority)

    v2s = dict()
    s2v = dict()
    create_map_from_tree(thePQ.removeMin(), v2s, s2v, "")
    return v2s, s2v
































    



    

