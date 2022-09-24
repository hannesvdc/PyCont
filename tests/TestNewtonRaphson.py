import sys
sys.path.append("../../")

import autograd.numpy as np
import autograd.numpy.linalg as lg
import autograd.numpy.random as rd
from autograd import grad, jacobian

import PyCont.NewtonRaphson as nr

# Find the zeros of sin(x) = x, this is a second order zero, only linear convergence
def test1():
	def f(u):
		x = u[0]
		y = u[1]
		return np.array([x - y + 1, y - x**2 - 1])
	df = jacobian(f)

	tol = 1.e-10
	u0 = np.array([0.6, 0.6])
	result = nr.NewtonRaphson(f, df, u0, a_tol=tol, max_it=100)

	possible_solutions = np.array([[0.0, 1.0], [-1.0, 2.0], [1.0, 2.0]])
	err = 1000.0
	index = -1
	for i in range(3):
		p = lg.norm(result.x - possible_solutions[i, :])
		if p < err:
			err = p
			index = i
	true_solution = possible_solutions[index, :]

	forward_error = lg.norm(true_solution - result.x)
	backward_error = lg.norm(f(result.x))
	print('# Iterations:', result.iterations)
	print('Forward Error:', forward_error)
	print('Backward Error:', backward_error)

	if backward_error < tol:
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

	x0 = rng.normal(0.0, 1.0, 3)
	tol = 1.e-10
	result = nr.NewtonRaphson(f, df, x0, a_tol=tol)

	exact_solution = np.dot(lg.inv(A), b)
	forward_error = lg.norm(exact_solution - result.x)
	backward_error = lg.norm(np.dot(A, result.x) - b)
	print('# Iterations:', result.iterations)
	print('Forward Error:', forward_error)
	print('Backward Error:', backward_error)

	if backward_error < tol:
		return 1
	else:
		return 0

if __name__ == '__main__':
	n_tests = 2
	good_tests = 0

	print("Test 1")
	res1 = test1()
	good_tests += res1

	print("\nTest 2")
	res2 = test2()
	good_tests += res2

	print("\n\nSuccessfull Tests:", good_tests, "/", n_tests, '!')





