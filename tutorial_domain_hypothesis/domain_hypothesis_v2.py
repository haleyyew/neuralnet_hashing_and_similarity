# Set Unionability
def compute_sets(A, B, threshold, sim_func, gram):
    t = []
    D = []

    # n_a = 0
    # n_b = 0
    n_a = len(A)
    n_b = len(B)    
    n_t = 0
    n_D = 0

    # for i, (k,v) in enumerate(A.items()):
    #     n_a += v
    # for i, (k,v) in enumerate(B.items()):
    #     n_b += v        

    print(len(A),'*',len(B))

    added_b = []
    interval = 50
    for i, (ai,vi) in enumerate(A.items()):

        max_j = -1
        b = None


        if (i+1) % interval == 0:
            print('still processing', ai, i)


        for j, (bj,vj) in enumerate(B.items()):
            score = sim_func(ai, bj, gram)

            if bj == ai:
                # n_t += 1
                # n_D += 1
                max_j = 1
                b = bj
                break

            elif score > threshold:
                # added_b.append(bj)
                # n_t += 1
                # n_D += 1
                # added_b.append(bj)
                # print('here', score)
                if max_j < score:
                    max_j = score
                    b = bj
                continue

                # if B[bj] - vi < 0: 
                #     n_D += -(B[bj] - vi)
                # if B[bj] - vi > 0: 
                #     n_D += B[bj] - vi
            else:
                continue
                # n_D += 1

        if max_j > 0:
            n_t += 1
            n_D += 1            
            added_b.append(b)
        else:
            n_D += 1   




    added_b = list(set(added_b))        
    for b in added_b:
    #     # j = B.index(b)
        del B[b]        # TODO what if b has much more items than a?

    for i, (k,v) in enumerate(B.items()):
        n_D += 1     

    # n_D = len(D)
    # n_t = len(t)

    # print(D)
    # print(n_D)

    print(n_t, n_D)

    return n_a, n_b, D, n_D, t, n_t

import operator as op
from functools import reduce
import sys
def nCr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)


    while numer > sys.maxsize or denom > sys.maxsize:
            print('num', numer)
            print('den', denom)
            if numer > 200000 and denom > 200000:
                numer = numer // 1000
                denom = denom // 1000
            else:
                break
            # raise Exception(str(e))

    ans = numer / denom

    return ans


def hypergeometric(s, n_a, n_b, n_D):
    return nCr(n_a, s) * nCr(n_D - n_a, n_b - s) / nCr(n_D, n_b)

def cdf(n_t, n_a, n_b, n_D):

    cdf_t = 0
    for s in range(n_t):
        p_s_na_nb_nD = hypergeometric(s, n_a, n_b, n_D)
        # print(s, p_s_na_nb_nD)
        cdf_t += p_s_na_nb_nD

    return cdf_t


def example_set():
    A = ['a', 'a', 'b', 'b', 'b']
    B = ['b', 'b', 'c', 'c', 'd']
    B_copy = B.copy()
    n_a, n_b, D, n_D, t, n_t = compute_sets(A, B)

    print(t)
    B = B_copy

    U_set = cdf(n_t, n_a, n_b, n_D)
    print(U_set)

    alpha = 0.95
    if U_set < alpha:
        same_domain = False # reject same domain hypothesis
    else:
        same_domain = True