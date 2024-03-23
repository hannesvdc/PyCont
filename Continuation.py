import sys
sys.path.append('../')

import autograd.numpy as np
import autograd.numpy.linalg as lg
import autograd.numpy.random as rd
import scipy.optimize as opt
import scipy.linalg as slg
from autograd import jacobian

import NewtonRaphson as nr

def continuation(G, dGdu, dGdp, u0, p0, ds_min, ds_max, ds, N, a_tol=1.e-8, max_it=10, sign=1.0):
	M = u0.size
	u = np.copy(u0) # Always the previous point on the curve
	p = np.copy(p0)	# Always the previous point on the curve
	u_path = [u]
	p_path = [p]

	print_str = 'Step n :{0:3d}\t u :{1:4f}\t p :{2:4f}'.format(0, lg.norm(u), p)
	print(print_str)

	# Choose intial tangent (guess). Users can specify a sign for a
	# particular continuation direction
	prev_tangent = np.zeros(M+1)
	prev_tangent[M] = -sign

	# Variables for bifurcation detection
	rng = rd.RandomState()
	r = rng.normal(0.0, 1.0, M+1); r = r/lg.norm(r)
	l = rng.normal(0.0, 1.0, M+1); l = l/lg.norm(l)
	prev_tau_test = 0.0
	bifurcation_points = []
	for n in range(1, N+1):
		# Determine the tangent to the curve at current point
		# By solving an underdetermined system with quadratic constraint norm(tau)**2 = 1
		Gu = dGdu(u, p)
		Gp = dGdp(u, p)
		tangent = _computeTangent(Gu, Gp, prev_tangent, M, a_tol)

		# Our implementation uses adaptive timetepping
		while ds > ds_min:
			# Predictor step;
			# compute initial value for Newton-Raphson method
			u_p = u + tangent[0:M] * ds
			p_p = p + tangent[M]   * ds
			x_p = np.append(u_p, p_p)

			# create the extended system and test for bifurcation points
			N = lambda x: np.dot(tangent, x - np.append(u, p)) + ds
			F = lambda x: np.append(G(x[0:M], x[M]), N(x))
			dF = jacobian(F)
			tau_test = tau_bifurcation(dF, x_p, l, r, M)

			# Test for bifurcation point
			if prev_tau_test * tau_test < 0.0: # Bifurcation point detected
				x_singular = _findBifurcationPoint(dF, x_p, l, r, M, a_tol)

				# Also test the Jacobian to be sure
				if lg.norm(x_singular - x_p) < 1.e-1 and np.abs(lg.det(dF(x_singular))) < 1.e-4:
					bifurcation_points.append(x_singular)
					print('Bifurcation Point at', x_singular, '. Aborting')

				#return np.array(u_path), np.array(p_path), bifurcation_points

			# Corrector step: Newton-Raphson
			result = nr.Newton(F, dF, x_p, a_tol=a_tol, max_it=max_it, testCondition=True)

			# Adaptive timestepping
			if result.success:
				u = np.copy(result.x[0:M])
				p = result.x[M]
				u_path.append(u)
				p_path.append(p)

				# Updating the arclength step and tangent vector
				ds = min(1.2*ds, ds_max)
				prev_tangent = np.copy(tangent)
				prev_tau_test = tau_test
				break

			# Decrease arclength if Newton routine needs more than max_it steps
			ds = max(0.5*ds, ds_min)
		
		print_str = 'Step n :{0:3d}\t u :{1:4f}\t p :{2:4f}'.format(n, lg.norm(u), p)
		print(print_str)

	return np.array(u_path), np.array(p_path), bifurcation_points

def _computeTangent(Gu, Gp, prev_tau, M, a_tol):
	# Setup extended jacobian
	DG = np.zeros((M,M+1)) 
	DG[0:M, 0:M] = Gu
	DG[:,M] = Gp

	# Do a version of quadratic programming (can we implement QP?)
	g_tangent = lambda tau: np.append(np.dot(DG, tau), np.dot(tau, tau) - 1.0)
	dg_tangent = jacobian(g_tangent)
	tau = nr.Newton(g_tangent, dg_tangent, prev_tau, a_tol=a_tol, testCondition=False).x

	return tau

def tau_bifurcation(dF, x, l, r, M):
	sys = np.zeros((M+2, M+2))
	sys[0:(M+1),0:(M+1)] = dF(x)
	sys[0:(M+1), M+1]= r
	sys[M+1, 0:(M+1)] = l
	rhs = np.zeros(M+2); rhs[M+1] = 1.0
	y = lg.solve(sys, rhs)

	return y[M+1]

def _findBifurcationPoint(dF, x_p, l, r, M, a_tol):
	# Find the bifurcation point by solving det(dF) = 0 (can become singular as well for pitchforks)
	min_functional = lambda x: tau_bifurcation(dF, x, l, r, M)**2
	min_result = opt.minimize(min_functional, x_p, tol=a_tol)
	x_singular = min_result.x
	return x_singular