import numpy as np

#L1  = [0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
L1  = [0, 1, 1, 0, 1, 0, 0, 0, 0, 0]
L2  = [1, 0, 0, 1, 0, 0, 0, 0, 0, 0]
#L3  = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
L3  = [0, 1, 0, 0, 0, 0, 1, 0, 0, 0]
L4  = [0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
L5  = [0, 0, 0, 0, 0, 1, 1, 0, 0, 0]
L6  = [0, 0, 0, 0, 0, 0, 1, 1, 0, 0]
L7  = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
L8  = [0, 0, 0, 0, 0, 0, 1, 0, 1, 0]
L9  = [0, 0, 0, 0, 0, 0, 1, 0, 0, 1]
L10 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

L = np.array([L1, L2, L3, L4, L5, L6, L7, L8, L9, L10])

ITERATIONS = 100

def getM(L):
    M = np.zeros([10, 10], dtype=float)
    # number of outgoing links
    c = np.zeros([10], dtype=int)
    
    ## TODO 1 compute the stochastic matrix M
    for i in range(0, 10):
        c[i] = sum(L[i])
    
    for i in range(0, 10):
        for j in range(0, 10):
            if L[j][i] == 0: 
                M[i][j] = 0
            else:
                M[i][j] = 1.0/c[j]
    return M
    
print("Matrix L (indices)")
print(L)    

M = getM(L)

print("Matrix M (stochastic matrix)")
print(M)

### TODO 2: compute pagerank with damping factor q = 0.15
### Then, sort and print: (page index (first index = 1 add +1) : pagerank)
### (use regular array + sort method + lambda function)
def getPR(M, q = 0.15):

    v = np.empty(len(M), dtype = float)
    v.fill(1.0/len(M))
    vCopy = v.copy()

    for _ in range(ITERATIONS):
        for rowI, row in enumerate(M):
            sum1 = 0
            for valI, val in enumerate(row):
                sum1 += val*v[valI]
            vCopy[rowI] = q + (1.0-q)*sum1
        v = vCopy.copy()

    pr = [[i + 1, val] for i, val in enumerate(v)]
    pr.sort(key = lambda val: -val[1])
    return pr

q = 0.15
pr = getPR(M, q)

print("PAGERANK")
print(pr)
    
### TODO 3: compute trustrank with damping factor q = 0.15
### Documents that are good = 1, 2 (indexes = 0, 1)
### Then, sort and print: (page index (first index = 1, add +1) : trustrank)
### (use regular array + sort method + lambda function)
def getTR(M, d_input, q = 0.15):
    d = [val/sum(d_input) for val in d_input]
    dCopy = d.copy()

    for _ in range(ITERATIONS):
        for rowI, row in enumerate(M):
            sum1 = 0
            for valI, val in enumerate(row):
                sum1 += val*d[valI]
            dCopy[rowI] = q*d[rowI] + (1.0-q)*sum1
        d = dCopy.copy()

    tr = [[i + 1, val] for i, val in enumerate(d)]
    tr.sort(key = lambda val: -val[1])
    return tr

d = [1,1,0,0,0,0,0,0,0,0]
q = 0.15
tr = getTR(M,d,q)

print("TRUSTRANK (DOCUMENTS 1 AND 2 ARE GOOD)")
print(tr)
    
### TODO 4: Repeat TODO 3 but remove the connections 3->7 and 1->5 (indexes: 2->6, 0->4) 
### before computing trustrank

L1  = [0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
L3  = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
L = np.array([L1, L2, L3, L4, L5, L6, L7, L8, L9, L10])

M = getM(L)
d = [1,1,0,0,0,0,0,0,0,0]
q = 0.15
tr = getTR(M,d,q)

print("TRUSTRANK REMOVED CONNECTIONS 3->7 AND 1->5")
print(tr)