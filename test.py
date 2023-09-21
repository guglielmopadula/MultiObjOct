import numpy as np
from copy import deepcopy
objs=[lambda x: np.max(x),lambda x: np.min(x),lambda x: np.sum(x), lambda x: np.prod(x), lambda x: x[0], lambda x: x[1], lambda x: x[2]]


P=np.array(([0,0,0],
           [0,0,1],
           [0,1,0],
           [0,1,1],
           [1,0,0],
           [1,0,1],
           [1,1,0],
           [1,1,1])
           )

def alg_98(A,B,objs):
    flag=False
    for i in range(len(objs)):
        if objs[i](A)>objs[i](B):
            flag=True
        else: 
            if objs[i](B)>objs[i](A):
                return False
    return flag
def alg_100(P,objs):
    F=[]
    for i in range(len(P)):
        tmp=P[i].copy().tolist()
        F.append(tmp)
        F_bak=deepcopy(F)
        for elem in F_bak:
            if elem!=tmp:
                if alg_98(elem,tmp,objs):
                    F.remove(tmp)
                    continue
                else:
                    if alg_98(tmp, elem,objs):
                        F.remove(elem)
        F_bak=deepcopy(F)
    return F

F=alg_100(P,objs)
print(F)

print(alg_98([1,1,1],[1,0,1],objs))