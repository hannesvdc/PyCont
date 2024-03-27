import sys
sys.path.append("../")

import autograd.numpy as np
import matplotlib.pyplot as plt

import src.PseudoArclengthContinuation as cont

def FoldTest():
	f = lambda x, r: r + x[0]**2
	dfdx = lambda x, r: np.array([[2.0*x[0]]])
	dfdr = lambda x, r: np.array([1.0])

	u0 = np.array([-5.0])
	p0 = -u0[0]**2

	ds_max = 0.01
	ds_min = 1.e-6
	ds = 0.1
	N = 5000
	u_path, r_path, _ = cont.continuation(f, dfdx, dfdr, u0, p0, ds_min, ds_max, ds, N, max_it=10, sign=1.0)
	u_path = np.transpose(u_path)

	fig = plt.figure()
	ax = fig.gca()
	ax.plot(r_path, u_path[0], color='blue', label='Numerical Continuation')
	ax.set_xlabel(r'$r$')
	ax.set_ylabel(r'$u$')
	ax.legend()
	plt.show()	

if __name__ == '__main__':
	FoldTest()





