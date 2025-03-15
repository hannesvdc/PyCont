import autograd.numpy as np
import scipy.linalg as lg
import scipy.optimize as opt
import scipy as sc
#from autograd import jacobian
np.seterr(all='ignore')
sc.special.seterr(all='ignore')

def _is_zero(x):
    return np.abs(x) < 1.e-4

def _find_all_zeros(f):
    t_range = np.linspace(0.0, 2.0*np.pi, 10**6 + 1)
    
    prev_f_val = f(np.array([1.0, 0.0]))
    roots = []
    for n in range(1, t_range.size):
        x = np.array([np.cos(t_range[n]), np.sin(t_range[n])])
        f_val = f(x)

        if f_val * prev_f_val <= 0.0:
            roots.append(x)
        prev_f_val = f_val

    return roots

# Minimizing the residual of a system is more stable than finding the exact nullspace
def _computeNullspace(Gu, Gp, M):
    phi_0 = np.eye(M)[:,0]
    phi_objective = lambda y: 0.5*np.dot(Gu(y), Gu(y))
    phi_constraint = opt.NonlinearConstraint(lambda y: np.dot(y, y) - 1.0, 0.0, 0.0)
    min_result = opt.minimize(phi_objective, phi_0, constraints=(phi_constraint))
    phi = min_result.x
    print('phi residual', lg.norm(phi_objective(phi)), phi)

    w_objective = lambda y: np.sqrt(np.dot(Gu(y) + Gp, Gu(y) + Gp))
    min_result = opt.minimize(w_objective, np.zeros(M))
    w = min_result.x
    w_1 = np.append(w, 1.0)
    print('w residual', lg.norm(w_objective(w)))

    return phi, w, w_1

# Gu_v takes arguments u, p, v
def _computeCoefficients(Gu_v, Gp, x_s, phi, w, w_1, M):
    r_diff = 1.e-8

    # Compute a
    Gu_phi = lambda x: Gu_v(x[0:M], x[M], phi) # R^{M+1} -> R^M
    a = np.dot(phi, (Gu_phi(x_s + r_diff * np.append(phi, 0.0)) - Gu_phi(x_s)) / r_diff)

    # Compute b
    Gx_w = lambda x: Gu_v(x[0:M], x[M], w) + Gp(x[0:M], x[M])
    b = np.dot(phi, (Gx_w(x_s + r_diff * np.append(phi, 0.0)) - Gx_w(x_s)) / r_diff)

    # Compute c
    c = np.dot(phi, (Gx_w(x_s + r_diff * w_1) - Gx_w(x_s)) / r_diff)

    print('abc', a, b, c)
    return a, b, c

def _solveABSystem(a, b, c):
    #solutions = []
    #f = lambda alpha: a*alpha**2 + 2.0*b*alpha*np.sqrt(1.0 - alpha**2) + c*(1.0 - alpha**2)

    # We go through separate cases for speed and ease of calculations
    # special_case = False
    # if _is_zero(a) and _is_zero(c):
    #     alpha_1 = 0.0
    #     alpha_2 = 1.0
    #     special_case = True
    # elif _is_zero(c):
    #     alpha_1 = 0.0
    #     g = lambda alpha: a*alpha + 2.0*b*np.sqrt(1.0 - alpha**2)
    #     alpha_2 = opt.fsolve(g, 0.5)[0]
    #     special_case = True
    # elif _is_zero(a):
    #     alpha_1 = 0.0
    #     g = lambda alpha: 2.0*b*alpha + c*np.sqrt(1.0 - alpha**2)
    #     alpha_2 = opt.fsolve(g, 0.5)[0]
    #     special_case = True
    # elif _is_zero(a - c): # Double roots are posssible, avoid deflation
    #     alpha_1 = opt.fsolve(f, 0.0)[0]
    #     alpha_2 = np.sqrt(1.0 - alpha_1**2)
    #     special_case = True

    f_full = lambda y: a*y[0]**2 + 2*b*y[0]*y[1] + c*y[1]**2
    #if special_case:
    #    print('alpha', alpha_1, alpha_2, f(alpha_1), f(alpha_2))
    #    solutions.append(np.array([ alpha_1,  np.sqrt(1.0 - alpha_1**2)]))
    #    solutions.append(np.array([ alpha_2,  np.sqrt(1.0 - alpha_2**2)]))
    #    solutions.append(np.array([-alpha_1, -np.sqrt(1.0 - alpha_1**2)]))
    #    solutions.append(np.array([-alpha_2, -np.sqrt(1.0 - alpha_2**2)]))
    #else:
    solutions = _find_all_zeros(f_full)

    return solutions

def branchSwitching(G, Gu_v, Gp, x_s, x_prev):
    print('\nBranch Switching')
    # Setting up variables
    M = x_s.size - 1
    u = x_s[0:M]
    p = x_s[M]

    # Computing necessary coefficients and vectors
    phi, w, w_1 =_computeNullspace(lambda v: Gu_v(u, p, v), Gp(u,p), M)
    a, b, c = _computeCoefficients(Gu_v, Gp, x_s, phi, w, w_1, M)
    solutions = _solveABSystem(a, b, c)

    # Fina all 4 branch tangents
    directions = []
    for n in range(len(solutions)):
        alpha = solutions[n][0]
        beta  = solutions[n][1]

        s = 0.01
        N = lambda x: np.dot(alpha*phi + beta/np.sqrt(1.0)*w, x[0:M] - x_s[0:M]) + beta/np.sqrt(1.0)*(x[M] - x_s[M]) + s
        F_branch = lambda x: np.append(G(x[0:M], x[M]), N(x))

        tangent = np.append(alpha*phi + beta/np.sqrt(1.0)*w, beta/np.sqrt(1.0))
        print('branch tangent', tangent, phi, w)
        x0 = x_s + s * tangent / lg.norm(tangent)
        dir = opt.newton_krylov(F_branch, x0)

        directions.append(dir)

    #Remove the direction where we came from
    inner_prodct = -np.inf
    for n in range(len(directions)):
        inner_pd = np.dot(directions[n]-x_s, x_prev-x_s) / (lg.norm(directions[n]-x_s) * lg.norm(x_prev-x_s))
        if inner_pd > inner_prodct:
            inner_prodct = inner_pd
            idx = n
    directions.pop(idx)
    print('directions', directions)

    # Returning 3 continuation directions
    return directions