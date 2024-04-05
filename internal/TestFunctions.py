import autograd.numpy as np
import autograd.numpy.linalg as lg

import Math as _math


def test_fn_bifurcation(dF, x, l, r, M):
	sys = np.zeros((M+2, M+2))
	sys[0:(M+1),0:(M+1)] = dF(x)
	sys[0:(M+1), M+1]= r
	sys[M+1, 0:(M+1)] = l
	rhs = np.zeros(M+2); rhs[M+1] = 1.0
	y = lg.solve(sys, rhs)

	return y[M+1]

def test_fn_hopf(dF, x, l, r, M):
	Gu = dF(x)[0:M, 0:M]
	I = np.eye(M)
	Gu_I = 2.0 * _math.bialternate(Gu, I)
	K = Gu_I.shape[0]

	sys = np.zeros((K+1, K+1))
	sys[0:K, 0:K] = Gu_I
	sys[K, 0:K] = l
	sys[0:K, K] = r
	rhs = np.zeros(K+1); rhs[K] = 1.0

	y = lg.solve(sys, rhs)
	return y[K]
