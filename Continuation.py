import numpy as np
import numpy.linalg as lg

import internal.PseudoArclengthContinuation as pac
import internal.BranchSwitching as brs
import internal.EigenFunctions as eigf


class ContinuationResult:
    def __init__(self):
        self.branches = []
        self.bifurcation_points = []

def pseudoArclengthContinuation(G, u0, p0, ds_min, ds_max, ds_0, N, tolerance=1.e-10):
    # Create gradient functions
    r_diff = 1.e-8
    Gu_v = lambda u, p, v: (G(u + r_diff * v, p) - G(u, p)) / r_diff
    Gp = lambda u, p: (G(u, p + r_diff) - G(u, p)) / r_diff

    # Compute the initial tangent to the curve
    M = u0.size
    rng = np.random.RandomState()
    result = ContinuationResult()
    random_tangent = rng.normal(0.0, 1.0, M+1)
    tangent = pac.computeTangent(u0, p0, Gu_v, Gp, random_tangent/lg.norm(random_tangent), M, tolerance)

    # Do continuation in both directions of the tangent
    ds = ds_0
    _recursiveContinuation(G, Gu_v, Gp, u0, p0,  tangent, M, ds_min, ds_max, ds, N, tolerance, result)
    _recursiveContinuation(G, Gu_v, Gp, u0, p0, -tangent, M, ds_min, ds_max, ds, N, tolerance, result)

    # Return all found branches and bifurcation points
    return result

def _recursiveContinuation(G, Gu_v, Gp, u0, p0, tangent, M, ds_min, ds_max, ds, N, tolerance, result):
    print('\n\nContinuation on Branch', len(result.branches) + 1)
    
    # Do regular continuation on this branch
    u_path, p_path, bf_points = pac.continuation(G, Gu_v, Gp, u0, p0, tangent, ds_min, ds_max, ds, N, a_tol=tolerance, max_it=10)
    u_path = np.transpose(u_path)[0]
    result.branches.append({'u': u_path, 'p': p_path})

    # If there are no bifurcation points on this path, return
    if len(bf_points) == 0:
       return
    
    # If there are bifurcation points, check if it is unique
    x_singular = bf_points[0]
    for n in range(len(result.bifurcation_points)):
        if lg.norm(x_singular - result.bifurcation_points[n]) / M < 1.e-4:
            return
    result.bifurcation_points.append(x_singular)
    print('Bifurcation Point at', x_singular)
        
    # The bifurcation point is unique, do branch switching
    x_prev = np.append(u_path[-10], p_path[-10]) # x_prev just needs to be a point on the previous path close to the bf point
    directions = brs.branchSwitching(G, Gu_v, Gp, x_singular, x_prev)

    # For each of the branches, run pseudo-arclength continuation
    for n in range(len(directions)):
        x0 = directions[n]
        _recursiveContinuation(G, Gu_v, Gp, x0[0:M], x0[M], x0 - x_singular, M, ds_min, ds_max, ds, N, tolerance, result)