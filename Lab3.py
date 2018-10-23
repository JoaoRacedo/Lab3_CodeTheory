import numpy as np
import itertools 
from sympy import Symbol
from datetime import datetime
startTime = datetime.now()

#### Functions
'''
Genereate Code C
@param Standar_G_Matrix M, F_{q}^k x
@return return C
'''
def Generate_C(x, q, n, M):
    C = np.zeros((len(x),n), dtype = int)
    for i in range(0,len(C)):
        C[i][:] = x[i].dot(M)%q
    return C

'''
Creates Matrix H
@param Standar_G_Matrix M
@return return H
'''
def Create_H(M,q):
    n = M.shape[1]
    k = M.shape[0]
    #Matrix_I = M[:,range(k)]
    Matrix_P = M[:,range((k),n)]
    P_Transpose = -1*Matrix_P.T
    P_Transpose[P_Transpose < 0] +=q
    Matrix_I = np.identity((n-k), dtype=int)
    H = np.append(P_Transpose,Matrix_I, axis=1)
    return H

'''
Create x vector
@param q (Space), k (# of basis)
@return F_{q}^k
'''
def CreateX(q, k):
    x = np.asarray(list(itertools.product(range(q), repeat = k)))
    return x

'''
Polynomial weight enumerator
@param Code C
@return Polynomial weight enumerator
'''
def Enumerate_Polynomial(Code):
    x_sym = Symbol('x')
    f_sym = 0
    for i in range(len(Code)):
        w = np.count_nonzero(Code[i])
        f_sym += x_sym**w
    return f_sym
        
#### Main

# Variables Initialization

Matrix = np.loadtxt("goley_24.txt", delimiter=',',dtype = int)
k = Matrix.shape[0]
n = Matrix.shape[1]
q = Matrix.max() + 1

# Call of functions
x = CreateX(q,k)
C = Generate_C(x,q,n,Matrix)
G_dual = Create_H(Matrix,q)
f_sym = Enumerate_Polynomial(C)

# Show in Terminal

print("The G Matrix is: \n")
print(Matrix)
print(" ")
print("The Code C is: \n")
print(C)
print(len(C))
print(" ")
print("The G' Matrix is: \n")
print(G_dual)
print(" ")
print("The Polynomial weight enumerator is: \n")
print(f_sym)
print(" ")
print(datetime.now() - startTime)