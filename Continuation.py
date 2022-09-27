import sys
sys.path.append('../')

import autograd.numpy as np
import autograd.numpy.linalg as lg
from autograd import jacobian

import PyCont.NewtonRaphson as nr

def continuation(f, dfdu, dfdp, u0, p0, ds_max, ds, N, a_tol=1.e-8, max_it=10, verbose=False):
	print('\t', '  ', "\t\t\t", 'u', '\t\t', 'p')
	sign = 1.0
	m = u0.size
	samples = []

	u = np.copy(u0)
	p = np.copy(p0)
	for n in range(1, N+1):
		AB = " R" if n < N else "EP"

		# Predictor step:
		# Compute d(u, p)/ds for time-stepping
		fu = dfdu(u, p)
		fp = dfdp(u, p)
		duds, dpds = _computeDerivatives(fu, fp, sign)

		# Corrector step: 
		# create the system and solve it with the newton-raphson method.
		u, p, ds = _adaptiveStepping(f, u, p, duds, dpds, ds, ds_max, m, verbose)
		samples.append(np.concatenate((u, p)))
		print(n, '\t', AB, "\t\t", u, '\t', p, '\t')

	return samples

def _computeDerivatives(fu, fp, sign):
	dudp = lg.solve(fu, -fp)
	dpds = sign*np.sqrt(np.dot(dudp, dudp) + 1.0)
	duds = dudp * dpds

	return duds, dpds

def _adaptiveStepping(f, u0, p0, duds, dpds, ds, m, verbose):
	# Create the non-linear system
	def G(x):
		u_x = x[0:m]
		p_x = x[m:m+1]

		eq1 = f(u_x, p_x)
		eq2 = np.dot(u_x - u0, duds) + (p_x - p0)*dpds - ds
		res = np.concatenate((eq1, eq2))

		return res
	dG = jacobian(G)

	# Do adaptive timestepping
	while True:
		x0 = np.concatenate((u0 + duds*ds, p0 + dpds*ds))
		nr_result = nr.NewtonRaphson(G, dG, x0, max_it=10)

		u = None
		p = None
		if nr_result.success is True:
			ds = 1.2*ds
			u = nr_result.x[0:m]
			p = nr_result.x[m:m+1]
			break
		else:
			ds = 0.5*ds

	return u, p, ds
