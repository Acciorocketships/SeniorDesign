from sympy import *
import code
init_printing()

t = Symbol('t')

# Position in the world frame
x = Function('x')(t)
y = Function('y')(t)
z = Function('z')(t)

# Instantaneous, independent velocities in body frame
dxBody = (Function('xb')(t)).diff(t)
dyBody = (Function('yb')(t)).diff(t)
dzBody = (Function('zb')(t)).diff(t)

# Instantaneous, independent angular velocities in body frame
drollBody = (Function('rb')(t).diff(t))
dpitchBody = (Function('pb')(t)).diff(t)
dyawBody = (Function('yb')(t)).diff(t)

# Orientation in the world frame
roll = Function('phi')(t)
pitch = Function('theta')(t)
yaw = Function('psi')(t)

# Control inputs
uRoll = Symbol('R')
uPitch = Symbol('P')
uYaw = Symbol('Y')
uThrust = Symbol('T')

# Constants
mass = Symbol('m')
gravity = Symbol('g')
inertiaX = Symbol('Ix')
inertiaY = Symbol('Iy')
inertiaZ = Symbol('Iz')

eoms = [( -gravity*sin(pitch) + dyawBody*dyBody - dpitchBody*dzBody ),
		( gravity*sin(roll)*cos(pitch) - dyawBody*dxBody + drollBody*dzBody ),
		( -uThrust/mass + gravity*cos(roll)*cos(pitch) + dpitchBody*dxBody - drollBody*dyBody ),
		( 1/inertiaX * (uRoll + (inertiaY-inertiaZ)*dpitchBody*dyawBody) ),
		( 1/inertiaY * (uPitch + (inertiaZ-inertiaX)*drollBody*dyawBody) ),
		( 1/inertiaZ * (uYaw + (inertiaX-inertiaY)*drollBody*dpitchBody) ),
		( drollBody + (dpitchBody*sin(roll) + dyawBody*cos(roll)) * tan(pitch) ),
		( dpitchBody*cos(roll) - dyawBody*sin(roll) ),
		( (dpitchBody*sin(roll) + dyawBody*cos(roll)) * sec(pitch) ),
		( (cos(pitch)*cos(yaw))*dxBody + (-cos(roll)*sin(yaw)+sin(roll)*sin(pitch)*cos(yaw))*dyBody + (sin(roll)*sin(yaw)+cos(roll)*sin(pitch)*cos(yaw))*dzBody ),
		( (cos(pitch)*sin(yaw))*dxBody + (cos(roll)*cos(yaw)+sin(roll)*sin(pitch)*sin(yaw))*dyBody + (-sin(roll)*cos(yaw)+cos(roll)*sin(pitch)*sin(yaw))*dzBody ),
		( (-sin(pitch))*dxBody + (sin(roll)*cos(pitch))*dyBody + (cos(roll)*cos(pitch))*dzBody ) ]

X = [dxBody, dyBody, dzBody, drollBody, dpitchBody, dyawBody, roll, pitch, yaw, x, y, z]

d2xBody = eoms[0].diff(t).subs(dyawBody.diff(t),eoms[5]).subs(dpitchBody.diff(t),eoms[4])
d2yBody = eoms[1].diff(t).subs(dyawBody.diff(t),eoms[5]).subs(drollBody.diff(t),eoms[3])
d2zBody = eoms[2].diff(t).subs(drollBody.diff(t),eoms[3]).subs(dpitchBody.diff(t),eoms[4])

dx = eoms[9].diff(t).subs(dxBody.diff(t),d2xBody).subs(dyBody.diff(t),d2yBody).subs(dzBody.diff(t),d2zBody)
dy = eoms[10].diff(t).subs(dxBody.diff(t),d2xBody).subs(dyBody.diff(t),d2yBody).subs(dzBody.diff(t),d2zBody)
dz = eoms[11].diff(t).subs(dxBody.diff(t),d2xBody).subs(dyBody.diff(t),d2yBody).subs(dzBody.diff(t),d2zBody)

P = solve(d2xBody,uPitch)
R = solve(d2yBody,uRoll)

print('\ndx = ')
pprint(dx)
print('\ndy = ')
pprint(dy)
print('\ndz = ')
pprint(dz)
code.interact(local=locals())



