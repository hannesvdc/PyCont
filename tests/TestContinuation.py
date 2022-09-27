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

	fig = plt.figure()
	ax = fig.gca()
	r_list = np.linspace(-9.0, 0.0, 1000)
	b1 = -np.sqrt(-r_list)
	b2 = +np.sqrt(-r_list)
	ax.plot(samples[:,1], samples[:,0], color='blue', label='Numerical Continuation')
	ax.plot(r_list, b1, color='red', label='True curve')
	ax.plot(r_list, b2, color='red')
	ax.set_xlabel(r'$r$')
	ax.set_label(r'$u$')
	ax.legend()
	plt.show()	

if __name__ == '__main__':
	BlueSkyTest()





