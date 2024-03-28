import autograd.numpy as np
import autograd.numpy.linalg as lg


def test_fn_bifurcation(dF, x, l, r, M):
	sys = np.zeros((M+2, M+2))
	sys[0:(M+1),0:(M+1)] = dF(x)
	sys[0:(M+1), M+1]= r
	sys[M+1, 0:(M+1)] = l
	rhs = np.zeros(M+2); rhs[M+1] = 1.0
	y = lg.solve(sys, rhs)

	return y[M+1]

def test_fn_fold(dF, x, M):
	return lg.det(dF(x)[0:M,0:M]) # determinant of Gu
