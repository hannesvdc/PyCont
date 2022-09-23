import sys
sys.path.append('../')

import autograd.numpy as np
import autograd.numpy.linalg as lg
from autograd import jacobian

import PyCont.NewtonRaphson as nr

def continuation(f, dfdu, dfdp, u0, p0, ds, N, a_tol=1.e-8, max_it=10):
	sign = 1.0
	m = u0.size

	u = np.copy(u0)
	p = np.copy(p0)
	for n in range(1, N+1):
		AB = " R" if n < N else "EP"

		# Compute the derivatives duds and dpds
		fu = dfdu(u, p)
		fp = dfdp(u, p)
		duds, dpds = _computeDerivatives(fu, fp, sign)

		# Test for folds at the current point
		is_fold = _testFold(dpds)
		if is_fold:
			sign = -sign
			AB = "FP"

		# Corrector step: create the system and solve it with the newton-raphson method.
		u, p = _nextStep(f, u, p, duds, dpds, ds, m)
		print(n, '\t', AB, "\t\t", u, '\t', p)


def _computeDerivatives(fu, fp, sign):
	dudp = lg.solve(fu, -fp)
	dp = sign*np.sqrt(np.dot(dudp, dudp) + 1.0)
	du = dudp * dp

	return du, dp

def _nextStep(f, u0, p0, duds, dpds, ds, m):
	# Create the non-linear system
	def G(x):
		v1 = x[0:m-1]
		v2 = x[m-1]

		eq1 = f(v1, v2)
		eq2 = (v1 - u0)*duds + (v2 - p0)*dpds - ds

		res = np.zeros(m)
		res[0:m-1] = eq1
		res[m-1] = eq2

		return res
	dG = jacobian(G)

	# Setup initial point
	x0 = np.array([u + duds*ds, p + dpds*ds])
	nr_result = nr.NewtonRaphson(G, dG, x0, max_it=25)

	u = nr_result.x[0:m-1]
	p = nr_result.x[m-1]

	return u, p

def _testFold(dpds):
	fold_tol = 1.e-5
	return np.abs(dpds) < fold_tol
