import sys
sys.path.append("../../")

import autograd.numpy as np
import autograd.numpy.linalg as lg
import autograd.numpy.random as rd
import matplotlib.pyplot as plt
from autograd import grad, jacobian, elementwise_grad

import PyCont.Continuation as cont

def BlueSkyTest():
	def f(x, r):
		return x*x + r
	dfdx = lambda x, r: np.array([[2.0*x[0]]])
	dfdr = lambda x, r: np.array([1.0])

	u0 = np.array([-3.0])
	p0 = np.array([-9.0])

	ds = 0.1
	N = 200
	samples = cont.continuation(f, dfdx, dfdr, u0, p0, ds, N)
	samples = np.array(samples)

	r_list = np.linspace(-9.0, 0.0, 1000)
	b1 = -np.sqrt(-r_list)
	b2 = +np.sqrt(-r_list)
	plt.plot(samples[:,1], samples[:,0], color='blue', label='Numerical Continuation')
	plt.plot(r_list, b1, color='red', label='True curve')
	plt.plot(r_list, b2, color='red')
	plt.xlabel(r'$r$')
	plt.ylabel(r'$u$')
	plt.legend()
	plt.show()

if __name__ == '__main__':
	BlueSkyTest()





