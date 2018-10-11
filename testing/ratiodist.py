# from mpmath import e, pi
from sympy import *
from sympy.stats import *
import code
init_printing()

z = Symbol('z')
c1 = Symbol('c_1')
c2 = Symbol('c_2')
mu1 = Symbol('mu_1')
mu2 = Symbol('mu_2')
sig1 = Symbol('sigma_1')
sig2 = Symbol('sigma_2')

def gaussRatio(z):
	return (c1 * density(Normal("N",mu1,sig1))(z)) / (c2 * density(Normal("N",mu2,sig2))(z))

minGaussRatio = lambdify( [mu1,mu2,sig1,sig2], -mu1*sig2**2 / (sig1**2 - sig2**2) + mu2*sig1**2 / (sig1**2 - sig2**2) )

# Create Ratio of Gaussians
q = gaussRatio(z)
mu1_val = 4
mu2_val = 2
sig1_val = 2
sig2_val = 1

# Plot
qs = q.subs("mu_1",mu1_val).subs("mu_2",mu2_val).subs("sigma_1",sig1_val).subs("sigma_2",sig2_val).subs(c1,1).subs(c2,2)
plot(qs,ylim=(-1,3))

# Solve for Maximum
sln = solve(q.diff(z),z)
print(sln)
minval = minGaussRatio(mu1_val,mu2_val,sig1_val,sig2_val)
print(minval)

code.interact(local=locals())
