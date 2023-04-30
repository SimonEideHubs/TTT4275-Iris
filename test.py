import numpy as np
from numba import cuda, jit

@jit(nopython=True)
def simple():
    list = []
    for i in range (1000):
        list.append(1)
    return list

@jit(nopython=True)
def multiply_matrix(A, B):
    C = np.zeros((AM.shape[0], BM.shape[1]))
    for row in range(A.shape[0]): 
        for col in range(A.shape[1]):
            for elt in range(len(B)):
              C[row, col] += A[row, elt] * B[elt, col]
    return C


AM = np.matrix("1 2 3 ; 4 5 6")
BM = np.matrix("7 8 ; 9 10 ; 11 12")

print(AM)
print(BM)


for i in range (1):
    print(multiply_matrix(AM, BM)) 


import numba as nb
import numpy as np

@nb.vectorize(nopython=True)
def nbvectMoment(L,x):
    if x<L/2.0:
        return 0.5*x
    else:
        return 0.5*(L-x)

@nb.jit(nopython=True)
def nbSimpleSpanMoment(L, axles, spacings, step_size):
    travel = L + np.sum(spacings)
    maxmoment = 0
    axle_coords = -np.cumsum(spacings)
    moment_inf = np.empty_like(axles)
    while np.min(axle_coords) < L:
        axle_coords = axle_coords + step_size
        y = nbvectMoment(L,axle_coords)
        for k in range(y.shape[0]):
            if axle_coords[k] >=0 and axle_coords[k] <= L:
                moment_inf[k] = y[k]
            else:
                moment_inf[k] = 0.0   
        moment = np.sum(moment_inf * axles)
        if maxmoment < moment:
            maxmoment = moment
    return np.around(maxmoment,1)

def PosMomentSingle(current_axles, current_spacings):
    data_list = []
    for L in range (1,201):
        L=float(L)        
        if L <= 40:
            step_size = 0.5
        else:
            step_size = 0.5            
        axles = np.array(current_axles, dtype='f')
        spacings = np.array(current_spacings, dtype='f')            
        axles_inv = axles[::-1]
        spacings_inv = spacings[::-1]           
        spacings = np.insert(spacings,0,0)
        spacings_inv = np.insert(spacings_inv,0,0)            
        left_to_right = nbSimpleSpanMoment(L, axles, spacings, step_size)
        right_to_left = nbSimpleSpanMoment(L, axles_inv, spacings_inv, step_size)            
        data_list.append(max(left_to_right, right_to_left))
    return data_list

load_effects = []
for v in range(14,31):
    load_effects.append(PosMomentSingle([8, 32, 32], [14, v]))
load_effects = np.array(load_effects)