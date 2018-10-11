from mpmath import *
from sympy import *
from sympy.stats import *
import time
init_printing()

z = Symbol('z')

mu1 = Symbol('mu_1')
sig1 = Symbol('sigma_1')
mu2 = Symbol('mu_2')
sig2 = Symbol('sigma_2')
mu3 = Symbol('mu_3')
sig3 = Symbol('sigma_3')
mu4 = Symbol('mu_4')
sig4 = Symbol('sigma_4')

mu1_val = 3
mu2_val = 1
mu3_val = 4
mu4_val = 2
sig1_val = 2
sig2_val = 3
sig3_val = 4
sig4_val = 1


minGaussRatio = lambdify( [mu1,mu2,sig1,sig2], -mu1*sig2**2 / (sig1**2 - sig2**2) + mu2*sig1**2 / (sig1**2 - sig2**2) )

eqn = density(Normal("N",mu2,sig2))(z)/density(Normal("N",mu1,sig1))(z) + density(Normal("N",mu3,sig3))(z)/density(Normal("N",mu1,sig1))(z) + density(Normal("N",mu4,sig4))(z)/density(Normal("N",mu1,sig1))(z)
numeqn = eqn.subs(mu1,mu1_val).subs(sig1,sig1_val).subs(mu2,mu2_val).subs(sig2,sig2_val).subs(mu3,mu3_val).subs(sig3,sig3_val).subs(mu4,mu4_val).subs(sig4,sig4_val)

mpeqn = lambdify(z,numeqn.diff(z),"mpmath")
m2 = minGaussRatio(mu1_val,mu2_val,sig1_val,sig2_val)
m3 = minGaussRatio(mu1_val,mu3_val,sig1_val,sig3_val)
m4 = minGaussRatio(mu1_val,mu4_val,sig1_val,sig4_val)
solver = "secant"
root2 = findroot(mpeqn,m2,solver=solver)
root3 = findroot(mpeqn,m3,solver=solver)
root4 = findroot(mpeqn,m4,solver=solver)

print("m2", m2, "m3", m3, "m4", m4)
print("root2",root2,"root3",root3,"root4",root4)

#plot(numeqn,ylim=(0,5))
import code; code.interact(local=locals())