from mpmath import *
from sympy import *
from sympy.stats import *

# also make a 3D version

class Gauss:

	x = Symbol('x')

	c1 = Symbol('c_1')
	mu1 = Symbol('mu_1')
	sig1 = Symbol('sigma_1')
	c2 = Symbol('c_2')
	mu2 = Symbol('mu_2')
	sig2 = Symbol('sigma_2')

	minGaussRatio = lambdify( [mu1,mu2,sig1,sig2], (mu2*sig1**2 - mu1*sig2**2) / (sig1**2 - sig2**2) ) # catch sig1==sig2. perhaps with limit?

	gauss = lambdify( [c1,mu1,sig1,x], c1*density(Normal("N",mu1,sig1))(x) )

	def __init__(self,c=[1],mu=[None],sigma=[None],invc=[1],invmu=[None],invsigma=[None]):
		# ( ∑ c * N(mu,sigma) ) / ( ∑ invc * N(invmu,invsigma) )
		# for just a constant, do Gauss(c=[const],mu=[None],sigma=[None])
		self.numC = c if isinstance(c,list) else [c]
		self.numMu = mu if isinstance(mu,list) else [mu]
		self.numSig = sigma if isinstance(sigma,list) else [sigma]
		self.denC = invc if isinstance(invc,list) else [invc]
		self.denMu = invmu if isinstance(invmu,list) else [invmu]
		self.denSig = invsigma if isinstance(invsigma,list) else [invsigma]


	def globalMin(self,approx=False):
		# TODO: catch single numerator single denominator (set approx=True)
		solver = "secant"
		# Initial Guess Calculation
		guess = 0
		totalweighting = 0
		maxDen = (0, None, None)
		for i in range(len(self.denSig)):
			if maxDen[2] == None or self.denSig[i] > maxDen[2]:
				maxDen = (self.denC[i], self.denMu[i], self.denSig[i])
		for i in range(len(self.numSig)):
			if maxDen[2] != None and self.numSig[i] != None:
				weighting = self.numC[i] * (self.numSig[i] / maxDen[2]) ** 2
			else:
				weighting = 1
			if self.numSig[i] != maxDen[2] and maxDen[2] != None:
				if self.numSig[i] == None:
					localmin = maxDen[1]
				else:
					localmin = Gauss.minGaussRatio(self.numMu[i],maxDen[1],self.numSig[i],maxDen[2])
			else:
				localmin = 0
			guess += weighting * localmin
			totalweighting += weighting
		if totalweighting != 0:
			guess = guess / totalweighting
		if approx:
			return guess
		eqn = lambdify([Gauss.x], self.evaluate().diff(Gauss.x), "mpmath")
		root = findroot(eqn,guess,solver=solver)
		return float(root)
		# initial guess is weighted average of minGaussRatios for each ratio, where the weighting factor is c * (sigma1/sigma2)^2
		# check that at least one numerator is larger than the denominator (and that there is a denominator)
		# add constraints, use calculus of variations

	def localMin(self,x):
		return findroot(self.evaluate(Gauss.x),x,solver=solver)

	def evaluate(self,x=None):
		# evaluates at a given x, or symbolically if given Symbol('x'). f[x] calls evaluate.
		if x is None:
			x = Gauss.x
		if len(self.numC) > 0:
			if isinstance(x,Expr):
				num = sum(map(lambda N: N[0]*density(Normal("N",N[1],N[2]))(x) if (N[1] is not None) else N[0], zip(self.numC,self.numMu,self.numSig)))
			else:
				num = sum(map(lambda N: Gauss.gauss(N[0],N[1],N[2],x) if (N[1] is not None) else N[0], zip(self.numC,self.numMu,self.numSig)))
		else:
			num = 1
		if len(self.denC) > 0:
			if isinstance(x,Expr):
				den = sum(map(lambda N: N[0]*density(Normal("N",N[1],N[2]))(x) if (N[1] is not None) else N[0], zip(self.denC,self.denMu,self.denSig)))
			else:
				den = sum(map(lambda N: Gauss.gauss(N[0],N[1],N[2],x) if (N[1] is not None) else N[0], zip(self.denC,self.denMu,self.denSig)))
		else:
			den = 1
		return num / den


	def __add__(self,other):

		if not isinstance(other,Gauss):
			other = Gauss(c=other)

		if self.denC == other.denC and self.denMu==other.denMu and self.denSig==other.denSig:
			# If the denominators match, just add the numerators
			numC = self.numC + other.numC
			numMu = self.numMu + other.numMu
			numSig = self.numSig + other.numSig
			denC = list(self.denC)
			denMu = list(self.denMu)
			denSig = list(self.denSig)
		else:
			# a/b + c/d = (ad + bc) / (bd), where a/b is self and c/d is other
			ad = self.multiplyPoly((self.numC,self.numMu,self.numSig),(other.denC,other.denMu,other.denSig))
			bc = self.multiplyPoly((self.denC,self.denMu,self.denSig),(other.numC,other.numMu,other.numSig))
			bd = self.multiplyPoly((self.denC,self.denMu,self.denSig),(other.denC,other.denMu,other.denSig))
			numC = ad[0] + bc[0]
			numMu = ad[1] + bc[1]
			numSig = ad[2] + bc[2]
			denC = bd[0]
			denMu = bd[1]
			denSig = bd[2]

		return Gauss(c=numC,mu=numMu,sigma=numSig,invc=denC,invmu=denMu,invsigma=denSig)


	def __rmul__(self,other):
		return self.__mul__(other)


	def __mul__(self,other):
		# us multiplyPoly on num and den a/b * c/d = (ac) / (bd)
		if not isinstance(other,Gauss):
			other = Gauss(c=other)

		ac = self.multiplyPoly((self.numC,self.numMu,self.numSig),(other.numC,other.numMu,other.numSig))
		bd = self.multiplyPoly((self.denC,self.denMu,self.denSig),(other.denC,other.denMu,other.denSig))

		numC = ac[0]
		numMu = ac[1]
		numSig = ac[2]
		denC = bd[0]
		denMu = bd[1]
		denSig = bd[2]

		return Gauss(c=numC,mu=numMu,sigma=numSig,invc=denC,invmu=denMu,invsigma=denSig)


	def multiplyPoly(self,poly0,poly1):
		# input is of the form ([c0, c1, ...],[mu0, m1, ...],[sigma0, sigma1, ...])
		c = [None for i in range(len(poly0[0])*len(poly1[0]))]
		mu = [None for i in range(len(poly0[0])*len(poly1[0]))]
		sigma = [None for i in range(len(poly0[0])*len(poly1[0]))]

		for i0 in range(len(poly0[0])):
			for i1 in range(len(poly1[0])):
				ci, mui, sigi = self.multiply((poly0[0][i0],poly0[1][i0],poly0[2][i0]),
									          (poly1[0][i1],poly1[1][i1],poly1[2][i1]))
				i = i0*len(poly1) + i1
				c[i] = ci
				mu[i] = mui
				sigma[i] = sigi

		return (c,mu,sigma)


	def multiply(self,gauss0,gauss1):
		# inputs are of the form (c,mu,sigma)
		c0, mu0, sig0 = gauss0
		c1, mu1, sig1 = gauss1

		c = c0 * c1
		if (mu0 is None) and (mu1 is not None):
			mu = mu1
			sig = sig1
		elif (mu1 is None) and (mu0 is not None):
			mu = mu0
			sig = sig0
		elif (mu1 is None) and (mu0 is None):
			mu = None
			sig = None
		else:
			mu = (mu0 * sig1**2 + mu1 * sig0**2) / (sig0**2 + sig1**2)
			sig = sqrt( (sig0**2 * sig1**2) / (sig0**2 + sig1**2) )

		return (c,mu,sig)


	def __rtruediv__(self,other):
		# other / self

		if not isinstance(other,Gauss):
			other = Gauss(c=other)

		selfNumC = self.numC
		selfNumMu = self.numMu
		selfNumSig = self.numSig
		selfDenC = self.denC
		selfDenMu = self.denMu
		selfDenSig = self.denSig
		self.numC = selfDenC
		self.numMu = selfDenMu
		self.numSig = selfDenSig
		self.denC = selfNumC
		self.denMu = selfNumMu
		self.denSig = selfNumSig
		return self.__mul__(other)


	def __truediv__(self,other):
		# self / other

		if not isinstance(other,Gauss):
			other = Gauss(c=other)

		otherNumC = other.numC
		otherNumMu = other.numMu
		otherNumSig = other.numSig
		otherDenC = other.denC
		otherDenMu = other.denMu
		otherDenSig = other.denSig
		other.numC = otherDenC
		other.numMu = otherDenMu
		other.numSig = otherDenSig
		other.denC = otherNumC
		other.denMu = otherNumMu
		other.denSig = otherNumSig
		return self.__mul__(other)


	def __neg__(self):
		return Gauss(c=list(map(lambda x: -x, self.numC)),
					 mu=list(self.numMu),
					 sigma=list(self.numSig),
					 invc=list(self.denC),
					 invmu=list(self.denMu),
					 invsigma=list(self.denSig))


	def __sub__(self,other):
		return self.__add__(other.__neg__())


	def __eq__(self,other):
		return (self.numC==other.numC and self.numMu==other.numMu and self.numSig==other.numSig) and \
			   (self.denC==other.denC and self.denMu==other.denMu and self.denSig==other.denSig)


	def __ne__(self,other):
		return not self.__eq__(other)


	def __getitem__(self,x):
		return self.evaluate(x)


	def plot(self):
		plot(self.evaluate(Gauss.x))



if __name__ == '__main__':
	g1 = Gauss(mu=[0],sigma=[1])
	g2 = Gauss(mu=[1],sigma=[1])
	import code; code.interact(local=locals())
