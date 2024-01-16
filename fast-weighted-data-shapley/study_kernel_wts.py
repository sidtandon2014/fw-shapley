import math
import matplotlib.pyplot as plt
from decimal import Decimal
import numpy as np

def shapley_kernel(N,j):
    return (N-1)/(math.comb(N,j)*j*(N-j))


def shapley_subset_cardinality_wt(N,j): # for the subsets with cardinality j, this function gives the weight
    return 1/(j*(N-j))


def beta_constant(a, b): # using Decimal module to deal with underflow of values
   '''
   the second argument (b; beta) should be integer in this function
   '''
   beta_fct_value=Decimal(1/a)
   for i in range(1,b):
        beta_fct_value=beta_fct_value*Decimal((i/(a+i)))
   return beta_fct_value

def w(M,a,b,j):
    return (M*beta_constant(j+b-1,M-j+a)/beta_constant(a,b))


def w_tilde(M,a,b,j): # use exponent based representation for handling 
    '''
    - w_tilde(j,a,b) = bin(M-1,j-1) * w(j,a,b) = T1 * T2
      We can do this approximation: express both T1 and T2 in the form: <a.bcdef x 10^{g}>
      Now, T1*T2 = (a1.b1c1d1e1f1 * a2.b2c2d2e2f2) * 10^{g1+g2}
    - We use the Decimal module to do this
    '''
    # exp_bc, base_bc = get_in_exp_form(binom_coeff(M-1,j-1))
    # exp_w, base_w = get_in_exp_form(w(M,a,b,j))
    #return ((base_bc*base_w)*math.pow(10,exp_bc+exp_w))
    return Decimal(math.comb(M-1,j-1))*Decimal(w(M,a,b,j))

def beta_shapley_kernel(N,a,b,j):
    return (w(N,a,b,j)/(N-j))*((N-1)/N) # simplified expression so that it has no combinatorial term

def beta_shapley_subset_cardinality_wt(N,a,b,j): # for the subsets with cardinality j, this function gives the weight
    return w_tilde(N,a,b,j)/(j*(N-j))


N,a,b = 2000,1,1
p_shapley = []
p_beta_shapley = []

for i in range(1,N):
    p_shapley.append(shapley_subset_cardinality_wt(N,i))
    p_beta_shapley.append(beta_shapley_subset_cardinality_wt(N,a,b,i))

p_shapley = np.array(p_shapley)
p_shapley = p_shapley/sum(p_shapley)

p_beta_shapley = np.array(p_beta_shapley)
p_beta_shapley = p_beta_shapley/sum(p_beta_shapley)

plt.plot(range(len(p_beta_shapley)),p_beta_shapley,'r')
plt.plot(range(len(p_shapley)),p_shapley,'g')
plt.show()
