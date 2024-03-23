import sys
sys.path.append("../")

import autograd.numpy as np
import matplotlib.pyplot as plt

import Continuation as cont

def PitchforkTest():
	f = lambda x, r: r*x[0] + x[0]**3
	dfdx = lambda x, r: np.array([[r + 3.0*x[0]**2]])
	dfdr = lambda x, r: np.array([x[0]])

	u0 = np.array([0.0])
	p0 = -5.0

	ds_max = 0.01
	ds_min = 1.e-6
	ds = 0.1
	N = 5000
	u_path, r_path = cont.continuation(f, dfdx, dfdr, u0, p0, ds_min, ds_max, ds, N, max_it=10)
	u_path = np.transpose(u_path)

	fig = plt.figure()
	ax = fig.gca()
	ax.plot(r_path, u_path[0], color='blue', label='Numerical Continuation')
	ax.set_xlabel(r'$r$')
	ax.set_ylabel(r'$u$')
	ax.legend()
	plt.show()	

if __name__ == '__main__':
	PitchforkTest()