import autograd.numpy as np
import scipy.linalg as lg
import scipy.optimize as opt
from autograd import jacobian

import NewtonRaphson as nr

np.seterr(all='ignore')

def _computeNullspace(Gu, Gp):
    ns = lg.null_space(Gu)
    phi = ns[:,0]

    Gu_pinv = lg.pinv(Gu)
    w = np.dot(Gu_pinv, -Gp)
    w_1 = np.append(w, 1.0)

    return phi, w, w_1

def _computeCoefficients(Gu, Gp, x_s, phi, w, w_1, M):
    # Compute a
    Gu_phi = lambda x: np.dot(Gu(x[0:M], x[M]), phi)
    Guu = jacobian(Gu_phi)
    a = np.dot(phi, np.dot(Guu(x_s)[0:M, 0:M], phi))

    # Compute b
    Gx_w = lambda x: np.dot(Gu(x[0:M], x[M]), w) + Gp(x[0:M], x[M])
    GuGx = jacobian(Gx_w)
    b = np.dot(phi, np.dot(GuGx(x_s)[0:M,0:M], phi))

    # Compute c
    GxGx = jacobian(Gx_w)
    c = np.dot(phi, np.dot(GxGx(x_s), w_1))

    print(a, b, c)
    return a, b, c

def _solveABSystem(a, b, c):
    solutions = []
    f = lambda alpha: a*alpha**2 + 2.0*b*alpha*np.sqrt(1.0 - alpha**2) + c*(1.0 - alpha**2)
    alpha_1 = opt.fsolve(f, 0.5)[0] # for some reason, the output of fsolve is an array
    f_deflated = lambda alpha: f(alpha) / (alpha - alpha_1)
    alpha_2 = opt.fsolve(f_deflated, 1.0 - alpha_1)[0] # Use 1 - alpha_1 as initial condition for now. Can we calculate alpha_2 analytically?
    print('alpha', alpha_1, alpha_2, f(alpha_1), f(alpha_2))

    solutions.append(np.array([ alpha_1,  np.sqrt(1.0 - alpha_1**2)]))
    solutions.append(np.array([ alpha_2,  np.sqrt(1.0 - alpha_2**2)]))
    solutions.append(np.array([-alpha_1, -np.sqrt(1.0 - alpha_1**2)]))
    solutions.append(np.array([-alpha_2, -np.sqrt(1.0 - alpha_2**2)]))

    return solutions

def branchSwitching(F, Gu, Gp, x_s, x_prev):
    # Setting up variables
    M = x_s.size - 1
    u = x_s[0:M]
    p = x_s[M]

    # Computing necessary coefficients and vectors
    phi, w, w_1 =_computeNullspace(Gu(u,p), Gp(u,p))
    a, b, c = _computeCoefficients(Gu, Gp, x_s, phi, w, w_1, M)
    solutions = _solveABSystem(a, b, c)

    # Fina all 4 branch tangents
    directions = []
    for n in range(len(solutions)):
        alpha = solutions[n][0]
        beta  = solutions[n][1]

        s = 0.1
        N = lambda x: np.dot(alpha*phi + beta/np.sqrt(2.0)*w, x[0:M] - x_s[0:M]) + beta/np.sqrt(2.0)*(x[M] - x_s[M]) + s
        F_branch = lambda x: np.append(F(x)[0:M], N(x))
        dF_branch = jacobian(F_branch)

        tangent = np.append(alpha*phi + beta/np.sqrt(2.0)*w, beta/np.sqrt(2.0))
        x0 = x_s + s * tangent
        res = nr.Newton(F_branch, dF_branch, x0)

        directions.append(res.x)

    #Remove the direction where we came from
    inner_prodct = -np.inf
    for n in range(len(directions)):
        inner_pd = np.dot(directions[n]-x_s, x_prev-x_s) / (lg.norm(directions[n]-x_s) * lg.norm(x_prev-x_s))
        if inner_pd > inner_prodct:
            inner_prodct = inner_pd
            idx = n
    directions.pop(idx)

    # Returning 3 contnuation directions
    return directions
    

    