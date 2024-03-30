import autograd.numpy as np
import autograd.numpy.linalg as lg
import autograd.numpy.random as rd
import scipy.optimize as opt
from autograd import jacobian

import NewtonRaphson as nr
import internal.TestFunctions as tf

def continuation(G, dGdu, dGdp, u0, p0, initial_tangent, ds_min, ds_max, ds, N, a_tol=1.e-10, max_it=10, sign=1.0):
	M = u0.size
	u = np.copy(u0) # Always the previous point on the curve
	p = np.copy(p0)	# Always the previous point on the curve
	u_path = [u]
	p_path = [p]

	print_str = 'Step n: {0:3d}\t u: {1:4f}\t p: {2:4f}'.format(0, lg.norm(u), p)
	print(print_str)

	# Choose intial tangent (guess). No idea why yet, but we need to
	# negate the tangent to find the actual search direction
	prev_tangent = -initial_tangent / lg.norm(initial_tangent)

	# Variables for bifurcation detection
	rng = rd.RandomState()
	r = rng.normal(0.0, 1.0, M+1); r = r/lg.norm(r)
	l = rng.normal(0.0, 1.0, M+1); l = l/lg.norm(l)
	prev_tau_bifurcation = 0.0
	bifurcation_points = []
	for n in range(1, N+1):
		# Determine the tangent to the curve at current point
		# By solving an underdetermined system with quadratic constraint norm(tau)**2 = 1
		Gu = dGdu(u, p)
		Gp = dGdp(u, p)
		tangent = computeTangent(Gu, Gp, prev_tangent, M, a_tol)

		# Create the extended system for corrector
		N = lambda x: np.dot(tangent, x - np.append(u, p)) + ds
		F = lambda x: np.append(G(x[0:M], x[M]), N(x))
		dF = jacobian(F) # Replace by analytic formula later

		# Test for bifurcation point
		tau_bifurcation = tf.test_fn_bifurcation(dF, np.append(u, p), l, r, M)
		if prev_tau_bifurcation * tau_bifurcation < 0.0: # Bifurcation point detected
			x_singular = _computeBifurcationPoint(dF, np.append(u, p), l, r, M, a_tol)

			# Also test the Jacobian to be sure. If test succesful, return.
			if lg.norm(x_singular - np.append(u, p)) < 1.e-1 and np.abs(lg.det(dF(x_singular))) < 1.e-4:
				bifurcation_points.append(x_singular)
				return np.array(u_path), np.array(p_path), bifurcation_points

		# Our implementation uses adaptive timetepping
		while ds > ds_min:
			# Predictor: Extrapolation
			u_p = u + tangent[0:M] * ds
			p_p = p + tangent[M]   * ds
			x_p = np.append(u_p, p_p)

			# Corrector: Newton-Raphson
			result = nr.Newton(F, dF, x_p, a_tol=a_tol, max_it=max_it)

			# Adaptive timestepping
			if result.success:
				u = result.x[0:M]
				p = result.x[M]
				u_path.append(u)
				p_path.append(p)

				# Updating the arclength step and tangent vector
				ds = min(1.2*ds, ds_max)
				prev_tangent = np.copy(tangent)
				prev_tau_bifurcation = tau_bifurcation
				break

			# Decrease arclength if Newton routine needs more than max_it iterations
			ds = max(0.5*ds, ds_min)
		else:
			# This case should never happpen under normal circumstances
			print('Minimal Arclength Size is too large. Aborting.')
			return u_path, p_path, bifurcation_points
		
		print_str = 'Step n: {0:3d}\t u: {1:4f}\t p: {2:4f}'.format(n, lg.norm(u), p)
		print(print_str)

	return np.array(u_path), np.array(p_path), bifurcation_points

def computeTangent(Gu, Gp, prev_tau, M, a_tol):
	# Setup extended jacobian
	DG = np.zeros((M,M+1)) 
	DG[0:M, 0:M] = Gu
	DG[:,M] = Gp

	# Do a version of quadratic programming (can we implement QP?)
	g_tangent = lambda tau: np.append(np.dot(DG, tau), np.dot(tau, tau) - 1.0)
	dg_tangent = jacobian(g_tangent) # Compute derivative analytically/numerically in future
	tangent = nr.Newton(g_tangent, dg_tangent, prev_tau, a_tol=a_tol).x

	return tangent

def _computeBifurcationPoint(dF, x_p, l, r, M, a_tol):
	# Find the bifurcation point by solving det(dF) = 0 (can become singular as well for pitchforks)
	min_functional = lambda x: tf.test_fn_bifurcation(dF, x, l, r, M)**2
	min_result = opt.minimize(min_functional, x_p, tol=a_tol)
	x_singular = min_result.x
	return x_singular