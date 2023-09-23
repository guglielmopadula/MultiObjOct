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
                    break
                else:
                    if alg_98(tmp, elem,objs):
                        F.remove(elem)

        F_bak=deepcopy(F)
    return F

F=alg_100(P.copy(),objs)

def alg_101(P,objs):
    Pprime=P.copy()
    i=1
    R=[]
    PFR=np.zeros(len(P),dtype=np.int64)
    while len(Pprime)>0:
        Pprimebak=Pprime.copy().tolist()
        tmp=alg_100(Pprime,objs)
        R.append(tmp)   
        for elem in tmp:
            j=np.argmin(np.linalg.norm(P-np.array(elem),axis=1))
            PFR[j]=i
            Pprimebak.remove(elem)
        i=i+1
        Pprime=np.array(Pprimebak).copy()
    return R,PFR

R,PFR=alg_101(P.copy(),objs)

def alg_102(P,objs,PFR,R,r):
    sparsity=np.zeros(len(P))
    R_index=[]
    for tmp in R:
        R_index.append([np.argmin(np.linalg.norm(P-np.array(elem),axis=1)) for elem in tmp])
    for r_index in R_index:
        for i in range(len(objs)):
            obj=objs[i]
            tmp=np.zeros(len(r_index))
            for j in range(len(r_index)):
                tmp[j]=obj(P[j])
            indexsorted=np.array(r_index)[np.argsort(tmp)]
            sparsity[indexsorted[0]]=np.inf
            sparsity[indexsorted[-1]]=np.inf
            if len(indexsorted>2):
                for j in range(1,len(indexsorted)-1):
                    sparsity[indexsorted[j]]=sparsity[indexsorted[j]]+(obj(P[indexsorted[j+1]])-obj(P[indexsorted[j-1]]))/r[i]
    return sparsity

r=[1,1,3,1,1,1,1]
sparsity=alg_102(P,objs,PFR,R,r)

def alg_103(P,objs,r,t):
    R,PFR=alg_101(P.copy(),objs)
    sparsity=alg_102(P,objs,PFR,R,r)
    best=np.random.randint(0,len(P))
    for i in range(2,t):
        next=np.random.randint(0,len(P))
        if PFR[next]<PFR[best]:
            best=next
        else:
            if PFR[next]==PFR[best]:
                if sparsity[next]>sparsity[best]:
                    best=next
    return best

best=alg_103(P,objs,r,100)

P=np.array([[0,1,0],[1,0,1]])


def breed(A,r,objs,t):
    tmp=set([])
    for i in range(len(A)):
        for j in range(i+1,len(A)):
            index=np.random.randint(0,len(A[i]))
            Ai_bak=A[i].copy()
            Aj_bak=A[i].copy()
            Ai_bak[index]=A[j,index].copy()
            Aj_bak[index]=A[i,index].copy()
            tmp.add(tuple(Ai_bak.tolist()))
            tmp.add(tuple(Ai_bak.tolist()))
    tmp=np.array([list(elem) for elem in tmp])
    return tmp[alg_103(tmp,objs,r,t)].reshape(1,-1)

def alg_104(build,objs,r,a,t,maxtimes):
    P=build()
    A=set()
    for _ in range(maxtimes):
        Pset=set([tuple(elem) for elem in P.tolist()])
        Pset=Pset.union(A)
        P=np.array([list(elem) for elem in Pset])
        R,PFR=alg_101(P.copy(),objs)
        best=R[0]
        R_index=[]
        for tmp in R:
            R_index.append([np.argmin(np.linalg.norm(P-np.array(elem),axis=1)) for elem in tmp])
        sparsity=alg_102(P,objs,PFR,R,r)
        A=set()
        for i in range(len(R)):
            rsub=R[i]
            r_index=R_index[i]
            if len(A)+len(rsub)>=a:
                tmp=np.array(rsub)
                tmp=tmp[np.argsort(-sparsity[r_index])][a-len(A):]
                for elem in tmp.tolist():
                    A.add(tuple(elem))
                continue
            else:
                for elem in rsub:
                    A.add(tuple(elem))
        A=np.array([list(elem) for elem in A])
        P=breed(A,r,objs,t)
        A=set([tuple(elem) for elem in A.tolist()])
    return best
best=alg_104(lambda: P, objs,r,6,10,100)

def compute_wimpiness(P,objs):
    wimpinnes=np.zeros(len(P))
    for i in range(len(P)):
        for q in P:
            for obj in objs:
                if alg_98(q,P[i],objs):
                    wimpinnes[i]=wimpinnes[i]+1
    return wimpinnes

def alg_105(P,objs):
    distance=np.zeros((len(P),len(P)))
    for i in range(len(P)):
        for j in range(len(P)):
            for obj in objs:
                distance[i,j]=distance[i,j]+(obj(P[i])-obj(P[j]))**2
            distance[i,j]=np.sqrt(distance[i,j])
        distance[i]=np.sort(distance[i])
    return distance

def spea2_fitness(P,objs,k):
    wimpiness=compute_wimpiness(P,objs)
    distance=alg_105(P,objs)
    return 1/(wimpiness+1/(2+distance[:,k]))

def alg_106(P,objs,a):
    A=alg_101(P,objs)[0][0]
    Q=P.copy()
    A=set([tuple(elem) for elem in A])
    Q=set([tuple(elem) for elem in Q.tolist()])
    Q=Q.difference(A)
    Q=np.array([list(elem) for elem in Q])
    if len(A)<a:
        Qfit=spea2_fitness(Q.copy(),objs,1)
        Qtmp=Q[np.flip(np.argsort(Qfit))[:(a-len(A))]]
        Qtmp=set([tuple(elem) for elem in Qtmp.tolist()])
        A=A.union(Qtmp)
        A=np.array([list(elem) for elem in A])
    while len(A)>a:
        A=np.array([list(elem) for elem in A])
        closest=A[0]
        c=0
        distances=alg_105(A,objs)
        for l in range(len(A)):
            for k in range(len(A)):
                if distances[l,k]<distances[c,k]:
                    closest=A[l]
                    c=l
                    break
                else:
                    if distances[l,k]>distances[c,k]:
                        break
        A=set([tuple(elem) for elem in A.tolist()])
        A.remove(tuple(closest.tolist()))
        A=np.array([list(elem) for elem in A])
    A=np.array([list(elem) for elem in A])
    return A

A=alg_106(P,objs,2)

def alg_107(P,a,objs,r,t,maxtime):
    A=set()
    for _ in range(maxtime):
        P=set([tuple(elem) for elem in P.tolist()])
        P=P.union(A)
        P=np.array([list(elem) for elem in P])
        BestFront=alg_101(P,objs)[0][0]
        A=alg_106(P,objs,a)
        P=breed(A,r,objs,t)
        A=set([tuple(elem) for elem in A.tolist()])
    return BestFront



P=np.array(([0,0,0],
           [0,0,1],
           [0,1,0],
           [0,1,1],
           [1,0,0],
           [1,0,1],
           [1,1,0],
           ))


BestFront=alg_107(P,3,objs,r,10,100)

print(BestFront)







