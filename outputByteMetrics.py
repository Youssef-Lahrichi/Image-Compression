byteOrder = "little"

def getMaxCoeff(compressedOut):
    maxCoeff = 0;
    for coeff in compressedOut:
        if coeff > maxCoeff:
            maxCoeff = coeff

    return maxCoeff

def getMinCoeff(compressedOut):
    minCoeff = 0;
    for coeff in compressedOut:
        if coeff <  minCoeff and coeff != -128:
            minCoeff = coeff
    return minCoeff


def getUniqueCoeffs(compressedOut):
    uniqueCoeffs = set()

    for coeff in compressedOut:
        uniqueCoeffs.add(coeff)
    return len(uniqueCoeffs)

def getCoeffFrequencies(compressedOut):
    coeffFreqMap = dict()

    for coeff in compressedOut:
        if coeff in coeffFreqMap:
            coeffFreqMap[coeff] += 1
        else:
            coeffFreqMap[coeff] = 1

    return coeffFreqMap
