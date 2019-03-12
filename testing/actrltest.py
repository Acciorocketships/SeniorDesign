from gekko import GEKKO as solver
import numpy as np
import matplotlib.pyplot as plt

# CV (controlled variable): output or reference trajectory, status=0
# SV (state variable): exclusively something you have a measurement for
# FV (fixed variable): not time dependent, status=1
# MV (manipulated variable): control input, status=1
# Var (variable): normal variables used in equations
# Param (parameter): constant array or value in equations


m = solver()
nt = 100
m.time = np.linspace(0,100,nt)

m.p = m.Array(m.Var,dim=3,value=0)
m.v = m.Array(m.Var,dim=3,value=0)
m.a = m.Array(m.Var,dim=3,value=0)
m.u = m.Array(m.MV,dim=3,value=0,lb=0,ub=20)
for i in range(3):
	m.u[i].STATUS = 1
	m.u[i].DCOST = 0.01
m.g = m.Array(m.Param,dim=3,value=0); m.g[2] = -10
m.r = m.Array(m.CV,dim=3); m.r[:] = [10, 50, 100]
m.cost = m.Var(value=0)

m.Equations([m.p[i].dt() == m.v[i] for i in range(3)] + 
			[m.v[i].dt() == m.a[i] for i in range(3)] + 
			[m.a[i].dt() == (m.u[i] + m.g[i]) for i in range(3)] )
m.Equation( m.cost.dt() == np.sum(np.power(m.u,2) + np.power(m.p-m.r,2)) )

m.Obj(m.cost)
for i in range(3):
	m.fix(m.p[i],nt-1,m.r[i])

m.options.IMODE = 6
m.solve(disp=False)

for i in range(3):
	plt.subplot(3,1,i+1)
	plt.plot(m.time,np.array(m.u[i]),'-g',label='u['+str(i)+']')
	plt.plot(m.time,np.array(m.p[i]),'-b',label='p['+str(i)+']')
	plt.plot(m.time,np.tile(m.r,(nt,1))[:,i],'-r',label='r['+str(i)+']')
	plt.legend()
plt.show()