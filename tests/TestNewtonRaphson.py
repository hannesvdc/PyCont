import sys
sys.path.add("../../")

import autograd.numpy as np
import autograd.numpy.linalg as lg
import autograd.numpy.random as rd
from autograd import grad, jacobian

import PyCont.NewtonRaphson as nr

# Find the zeros of sin(x) = x
def test1():
	def f(x):
		return np.sin(x) - x
	df = grad(f)

	tol = 1.e-10
	x0 = 0.6
	result = nr.NewtonRaphson(f, df, x0, a_tol=tol)

	true_result = 0.0
	abs_error = np.abs(result.x - true_result)
	print('True solution:', true_result)
	print('NR solution:', result.x)
	print('Absolute error', abs_error)

	if abs_error < tol:
		return 1
	else:
		return 0


def test2():
	rng = rd.RandomState()
	A = rng.normal(0.0, 1.0, (3, 3))
	b = rng.normal(0.0, 1.0, 3)

	def f(x):
		return np.dot(A, x) - b
	df = jacobian(f)

	x0 = np.random(0.0, 1.0, 3)
	tol = 1.e-10
	result = nr.NewtonRaphson(f, df, x0, a_tol=tol)

	exact_solution = np.dot(lg.inverse(A), b)
	forward_error = lg.norm(exact_solution - result.x)
	bachward_error = lg.norm(np.dot(A, result.x) - b)
	print('# Iterations', result.iterations)
	print('Forward Error:', forward_error)
	print('Backward Error:', backward_error)

	if backward_error < tol:
		return 1
	else:
		return 0

if __name__ == '__main__':
	print("Test 1")
	res1 = test1()

	#print("Test 2")
	#res2 = test2()





