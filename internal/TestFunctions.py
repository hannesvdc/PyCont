import numpy as np
import scipy.sparse.linalg as slg

def test_fn_bifurcation(dF_w, x, l, r, M):
	def matvec(w):
		el_1 = dF_w(x, w[0:M+1]) + r*w[M+1]
		el_2 = np.dot(l, w[0:M+1])
		return np.append(el_1, el_2)
	sys = slg.LinearOperator((M+2, M+2), matvec=matvec)
	rhs = np.zeros(M+2); rhs[M+1] = 1.0
	y = slg.gmres(sys, rhs)

	return y[M+1]