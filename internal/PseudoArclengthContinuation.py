import numpy as np
import numpy.linalg as lg
import numpy.random as rd
import scipy.sparse.linalg as slg
import scipy.optimize as opt

import internal.TestFunctions as tf

"""
This function computes the tangent to the curve at a given point by solving D_u G * tau = - G_p.
The tangent vector then is [tau, 1] with normalization.

The arguments are:
	- u: The current state variable
	- p: The current parameter
	- Gu_v: The Jacobian of the system with respect to the state variable as a function of u, p, v
	- Gp: The Jacobian of the system with respect to the parameter as a function of u and p
	- prev_tau: The previous tangent vector (used for initial guess)
	- M: The size of the state variable
	- a_tol: The absolute tolerance for the Newton-Raphson solver
"""
def computeTangent(u, p, Gu_v, Gp, prev_tau, M, a_tol):
	DG = slg.LinearOperator((M, M), matvec=lambda v: Gu_v(u, p, v))
	tau = slg.gmres(DG, -Gp(u, p), x0=prev_tau[:M], atol=a_tol)[0]
	tangent = np.append(tau, 1.0)
	tangent = tangent / lg.norm(tangent)

	# Make sure the new tangent vector points in the same rough direction as the previous one
	if np.dot(tangent, prev_tau) < 0.0:
		tangent = -tangent
	return tangent

def continuation(G, Gu_v, Gp, u0, p0, initial_tangent, ds_min, ds_max, ds, N, a_tol=1.e-10, max_it=10):
	r_diff = 1.e-8
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
	prev_tau_value = 0.0
	prev_tau_vector = np.zeros(M+2)
	for n in range(1, N+1):
		# Determine the tangent to the curve at current point
		tangent = computeTangent(u, p, Gu_v, Gp, prev_tangent, M, a_tol)

		# Create the extended system for corrector
		N = lambda x: np.dot(tangent, x - np.append(u, p)) + ds
		F = lambda x: np.append(G(x[0:M], x[M]), N(x))
		dF_w = lambda x, w: (F(x + r_diff * w) - F(x)) / r_diff

		# Test for bifurcation point
		tau_vector, tau_value = tf.test_fn_bifurcation(dF_w, np.append(u, p), l, r, M, prev_tau_vector, a_tol)
		if prev_tau_value * tau_value < 0.0: # Bifurcation point detected
			print('Sign change detected', prev_tau_value, tau_value)
			x_singular = _computeBifurcationPoint(dF_w, np.append(u, p), l, r, M, a_tol, tau_vector)
			print('x_singular:', x_singular)

			# Also test the Jacobian to be sure. If test succesful, return.
			#if lg.norm(x_singular - np.append(u, p)) < 1.e-1 and np.abs(lg.det(dF(x_singular))) < 1.e-4:
			return np.array(u_path), np.array(p_path), [x_singular]

		# Our implementation uses adaptive timetepping
		while ds > ds_min:
			# Predictor: Extrapolation
			u_p = u + tangent[0:M] * ds
			p_p = p + tangent[M]   * ds
			x_p = np.append(u_p, p_p)

			# Corrector: Newton-Raphson
			try:
				x_result = opt.newton_krylov(F, x_p, f_tol=a_tol, maxiter=max_it, verbose=False)
				
				# Bookkeeping for the next step
				u = x_result[0:M]
				p = x_result[M]
				u_path.append(u)
				p_path.append(p)

				# Updating the arclength step and tangent vector
				ds = min(1.2*ds, ds_max)
				prev_tangent = np.copy(tangent)
				prev_tau_value = tau_value
				prev_tau_vector = tau_vector

				break
			except:
				# Decrease arclength if the solver needs more than max_it iterations
				ds = max(0.5*ds, ds_min)
		else:
			# This case should never happpen under normal circumstances
			print('Minimal Arclength Size is too large. Aborting.')
			return u_path, p_path, []
		
		print_str = 'Step n: {0:3d}\t u: {1:4f}\t p: {2:4f}'.format(n, lg.norm(u), p)
		print(print_str)

	return np.array(u_path), np.array(p_path), []

def _computeBifurcationPoint(dF_w, x_p, l, r, M, a_tol, tau_vector):
	# Use the tau_vector found during continuation as fixed initial guess
	def functional(x):
		output = tf.test_fn_bifurcation(dF_w, x, l, r, M, tau_vector, a_tol)
		return output[1]**2 # Return the tf value squared for minimization
	optimize_result = opt.minimize(functional, x_p, tol=a_tol)
	return optimize_result.x
# def _computeBifurcationPoint(dF_w, x_p, l, r, M, a_tol):
# 	# Find the bifurcation point by solving det(dF) = 0 (can become singular as well for pitchforks)
# 	min_functional = lambda x: tf.test_fn_bifurcation(dF_w, x, l, r, M)**2
# 	min_result = opt.minimize(min_functional, x_p, tol=a_tol)
# 	x_singular = min_result.x
# 	return x_singular