#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 09:25:07 2024

@author: tsiolkovsky
"""

import numpy as np
import scipy as sp
import torch
import torch.distributions as dist
import numpy.linalg as la
import scipy as sp

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
from matplotlib.patches import Ellipse
from matplotlib.patches import Circle
import seaborn as sns

import time


from scipy import integrate
from scipy import interpolate
from scipy.stats import norm
from scipy.stats import multivariate_normal

from mystic.solvers import buckshot
from mystic.solvers import diffev2
from mystic.penalty import quadratic_inequality
from mystic.penalty import barrier_inequality


import pickle

from cartpole import PolicyNet
from cartpole import LyapunovNet

env_params = {
    "gravity": 9.8,
    "masscart": 1.0,
    "masspole": 0.1,
    "total_mass": 1.1,
    "length": 1.0,  
    "tau": 0.05,  # seconds between state updates
    "max_force": 30.0,
    "bounds":[(-5,5),(-100,100),(-1,1),(-10,10)],
    "alpha":1.0,
    "dim":4
}

BOUND_CART = 1.0
BOUND_CART_SPEED = 1.0
BOUND_POLE = 1.0
BOUND_POLE_SPEED = 1.0

piCartW = torch.load('cartpole_state_to_save_policy_seed00.000125True.pt', map_location=lambda storage, loc: storage)
piCartW = piCartW['model_state_dict']

VCartW = torch.load('cartpole_state_to_save_lyap_seed00.000125True.pt', map_location=lambda storage, loc: storage)
VCartW = VCartW['model_state_dict']

piCart = PolicyNet()
piCart.load_state_dict(piCartW)

V = LyapunovNet()
V.load_state_dict(VCartW)

def model(state,action):
    
    gravity = env_params['gravity']
    masscart = env_params['masscart']
    masspole = env_params['masspole']
    total_mass = env_params['total_mass']
    length = env_params['length']
    tau = env_params['tau']
    max_force = env_params['max_force']
    
    if type(state) != torch.Tensor:
        state = torch.tensor(state).float()

    force = action
    force = torch.clamp(force, min = -max_force, max = max_force)
    if len(np.shape(state)) > 1:
        x = state[:, 0:1]
        x_dot = state[:, 1:2]
        theta = state[:, 2:3]
        theta_dot = state[:, 3:4]
    else:
        x = state[0:1]
        x_dot = state[1:2]
        theta = state[2:3]
        theta_dot = state[3:4]

    costheta = torch.cos(theta)
    sintheta = torch.sin(theta)

    temp = masscart + masspole * sintheta**2
    thetaacc = (-force * costheta - masspole * length * theta_dot**2 * costheta * sintheta \
        + total_mass * gravity * sintheta) / (length * temp)
    xacc = (force + masspole * sintheta * (length * theta_dot**2 - gravity * costheta)) / temp

    x_next = x + tau * x_dot
    x_dot_next = x_dot + tau * xacc
    theta_next = theta + tau * theta_dot
    theta_dot_next = theta_dot + tau * thetaacc
    
    if len(np.shape(state)) > 1:
        result = torch.cat((x_next, x_dot_next, theta_next, theta_dot_next), dim = 1)
    else:
        result = torch.tensor([x_next,x_dot_next,theta_next,theta_dot_next])

    return result

# This function computes the covariance noise model for the system in question
# This function may depend on system state, control input/action, set point or
# time. The noise model is assumed to be for a multivariate Gaussian
#
#   returns:
#       S: covariance matrix for a multivariate Gaussian, must be
#          Symmetric Positive Definite

def noiseModel(x,u,r,t):
    
    
    Sig = env_params['alpha']*torch.tensor([[0.01,0.001, 0, 0],[0.001,0.05,0,0],[0,0,0.01,0.001],[0,0,0.001,0.002]])
    
    if len(np.shape(x)) > 1:
        N,dim = np.shape(x)
        noiseSamples = dist.MultivariateNormal(torch.zeros(dim),Sig).sample([N])
    else:
        noiseSamples = dist.MultivariateNormal(torch.zeros(len(x)),Sig).sample()

    return noiseSamples,Sig

# This function returns the set point the system aims for given the current
# state of the system and constraints

def planner(x,c):
    r = [0.,0.,0.,0.]
    #if x[0] > r[0]:
    #    r[0] = x[0]
                
    return r

# This function computes the control action/input to appy to the system given
# the current state and set point

def controller(x,r):
    
    if type(x) != torch.Tensor:
        x = torch.tensor(x).float()
    
    u = piCart(x)
    
    return u.detach()
# This function computes the constraints imposed on the system at the given system
# state and time

def constraints(t):
    a0 = np.array([1,0,0,0])
    b0 = 5
    Hdef0 = (a0,b0)
    
    a1 = np.array([1,0,0,0])
    b1 = -5
    Hdef1 = (a1,b1)
    
    a2 = np.array([0,0,1,0])
    b2 = 1
    Hdef2 = (a2,b2)
    
    a3 = np.array([0,0,1,0])
    b3 = -1
    Hdef3 = (a3,b3)
    # if t < 15:
    #     b = 5
    # elif t >= 15 and t <30:
    #     b = 10
    # elif t>=30 and t<50:
    #     b = 17.5
    # else:
    #     b = 27.5
    
    
    return [Hdef0,Hdef1,Hdef2,Hdef3]

def plotElls(Ell,ri,ax,color,linetype):
    Q,q = Ell
    U,D,V = la.svd(Q)
    
    rx,ry = 1./np.sqrt(D)
    dx,dy = 2*rx,2*ry
    a,b = max(dx,dy)*np.sqrt(ri),min(dx,dy)*np.sqrt(ri)
    e = np.sqrt(a**2-b**2)/a
    
    arcsin = -1. * np.rad2deg(np.arcsin(V[0][0]))
    arccos = np.rad2deg(np.arccos(V[0][1]))
    # Orientation angle (with respect to the x axis counterclockwise).
    alpha = arccos if arcsin > 0. else -1. * arccos
    
    ellipse2 = Ellipse(xy=q, width=a, height=b,edgecolor=color,
        angle=alpha, linestyle = linetype,fc='None', lw=2,zorder=10)
    ax.add_patch(ellipse2)
    
    return 0


def baseRegion(points, tol=0.01):
    """
    Finds the ellipse equation in "center form"
    (x-c).T * A * (x-c) = 1
    """
    N, d = points.shape
    Q = np.column_stack((points, np.ones(N))).T
    err = tol+1.0
    u = np.ones(N)/N
    while err > tol:
        # assert u.sum() == 1 # invariant
        X = np.dot(np.dot(Q, np.diag(u)), Q.T)
        M = np.diag(np.dot(np.dot(Q.T, la.inv(X)), Q))
        jdx = np.argmax(M)
        step_size = (M[jdx]-d-1.0)/((d+1)*(M[jdx]-1.0))
        new_u = (1-step_size)*u
        new_u[jdx] += step_size
        err = la.norm(new_u-u)
        u = new_u
    c = np.dot(u, points)
    A = la.inv(np.dot(np.dot(points.T, np.diag(u)), points)
               - np.multiply.outer(c, c))/d
    return A, c

def monteCarloSimulate(x0,T):
    X_0T = []
    X_0Texp = []
    if type(x0) == torch.Tensor:
        X_0T.append(x0.numpy())
    else:
        X_0T.append(x0)
    x_t = x0
    for t in range(T):
        
        c_t = constraints(t)
        r_t = planner(x_t, c_t)
        u_t = controller(x_t, r_t)
        xExp_next = model(x_t,u_t)
        noiseSamples,Cov = noiseModel(x_t,u_t,r_t,t)
        noise_next = noiseSamples
            
        x_t = xExp_next + noise_next
        X_0T.append(np.array(x_t))
        X_0Texp.append(np.array(xExp_next))
        
    return X_0T,X_0Texp

def regionDefMetric(x,t):
    C = constraints(t)
    r = planner(x,C)
    
    return np.linalg.norm(x-r,axis=1)

def scoreStateVec(tubeInfo,x):
    
    At = tubeInfo[0]
    c = tubeInfo[1]
    v = x-c
    scores = np.einsum('ij,jk,ik->i',v,At,v)
    
    return scores

def scoreStateVecLyapunov(r,x):
    
    return V(torch.tensor(r-x).float()).detach().numpy().flatten()


def scoreState(tubeInfo,x):
    
    At = tubeInfo[0]
    c = tubeInfo[1]
    
    score = np.dot((x-c),np.dot(At,(x-c)))
    
    return score

def measureOfSafety(x,t):
    rs = []
    Cx = constraints(t)
    for Hdef in Cx:
        a = Hdef[0]
        b = Hdef[1]
        
        if b < 0:
           rs.append(b - np.dot(a,x)) 
        else:
            rs.append(np.dot(a,x)-b)
    return max(rs)

def maxBoundaryToConstraint(Et,t):
    
    Qt = Et[0]
    qt = Et[1]
   
    rList = []
    xList = []
    Cx = constraints(t)
    for Hdef in Cx:
        a = Hdef[0]
        b = Hdef[1]
        
        gamma = 2*(b-np.dot(a,qt))/(np.dot(a,np.dot(np.linalg.inv(Qt),a)))
        x = np.linalg.solve(Qt, (gamma/2)*a+np.dot(Qt,qt))
        r = np.dot(x-qt,np.dot(Qt,x-qt))
        rList.append(r)
        xList.append(x)
        r = scoreState(Et, a)
        rList.append(r)
        xList.append(a)
    
    i = np.argmax(rList)
    
    return max(rList),xList[i]

def minBoundaryToConstraint(SigInv,x,t):
    rList = []
    xList = []
    Cx = constraints(t)
    for Hdef in Cx:
        a = Hdef[0]
        b = Hdef[1]
        
        gamma = 2*(b-np.dot(a,x))/(np.dot(a,np.dot(np.linalg.inv(SigInv),a)))
        xs = np.linalg.solve(SigInv, (gamma/2)*a + np.dot(SigInv,x))
        r = np.dot(xs-x,np.dot(SigInv,xs-x))
        if measureOfSafety(x, t) > 0:
            r = 0.0
        rList.append(r)
        xList.append(xs)
    
    i = np.argmin(rList)
    
    return rList[i],xList[i]

def minBoundaryToConstraintVec(SigInv,SigInvBlock,x,t):
    rList = []
    xList = []
    n = len(x)
    Cx = constraints(t)
    indSafe = np.ones(n)*True
    
    for Hdef in Cx:
        a = Hdef[0]
        b = Hdef[1]
        if b < 0:
            indSafe = np.logical_and(indSafe,a@x.T >= b)
        else:
            indSafe = np.logical_and(indSafe,a@x.T <= b)
    
        ax = a@x.T
        aSa = np.dot(a,np.dot(np.linalg.inv(SigInv),a))
        gammas = 2*(b-ax)/(aSa)
        
        bs = np.reshape((gammas[:,None]*a)/2 + (SigInv@x.T).T,env_params['dim']*n)
        
        xsol = np.reshape(sp.sparse.linalg.spsolve(SigInvBlock, bs),(n,env_params['dim']))
        v = xsol - x
        rc = np.einsum('ij,jk,ik->i',v,SigInv,v)
        
        rList.append(rc)
        xList.append(xsol)
    
    rList = np.array(rList)
    xList = np.array(xList)
    inds = np.argmin(rList,axis = 0)
    
    
    rs = np.array([rList[inds[i],i] for i in range(len(inds))])
    xret = [xList[inds[i],i] for i in range(len(inds))]
    rs[np.logical_not(indSafe)] = 0
    
    return rs,xret


def maxBoundaryToConstraintLyapunov(t):
    
    x = torch.tensor([10.,0.,1.,0.])
    r = V(x).detach().numpy()[0]
   
    rList = []
    xList = []
    Cx = constraints(t)
    for Hdef in Cx:
        a = Hdef[0]
        b = Hdef[1]
        if b < 0:
            penalty = lambda x: np.dot(a,x)-b
        else:
            penalty = lambda x: b - np.dot(a,x)
        
        @quadratic_inequality(penalty)
        def pf(x):
            return 0.0
        res = diffev2(lambda x: V(torch.tensor(x).float()).detach().numpy()[0],x0=np.zeros(len(a)),bounds=env_params['bounds'],penalty=pf,npop=40,disp=False,full_output=True)
    
        rList.append(res[1])
        xList.append(res[0])
    
    i = np.argmax(rList)
    
    if rList[i] > r:
        r = rList[i]
        x = xList[i]        
    
    return r,x

def defineRegions(numPfs,numVs,pfTarget,SigInv,t,alpha):
    
    safetyPenalty = lambda x: measureOfSafety(x,t)
    failurePenalty = lambda x: -measureOfSafety(x,t)
    
    Vs = np.zeros(numVs +1)

    @quadratic_inequality(failurePenalty)
    def pf(x):
        return 0.0
    
    VminFail = diffev2(lambda x: V(torch.tensor(x)), np.zeros(env_params['dim']), bounds = env_params['bounds'], penalty=pf,npop=40,disp=False,full_output = True)
    vi = VminFail[1]
    
    
    Vs[0:-1] = np.linspace(0,vi,numVs)
    Vs[-1] = np.inf
    
    vp = lambda x: V(torch.tensor(x)) - Vs[1]
    @quadratic_inequality(vp)
    @quadratic_inequality(safetyPenalty)
    def pf(x):
        return 0.0
    pmax = diffev2(lambda x: minBoundaryToConstraintOverTrajectory(x, t, SigInv), np.zeros(env_params['dim']), bounds = env_params['bounds'],penalty=pf,npop=40,disp=False,full_output=True)  
    pf0,_ = pFailoverHorizon(pmax[0],0,SigInv)
    
    while pf0 > pfTarget:
        Vs[1] = Vs[1]/2
        @quadratic_inequality(vp)
        def pf(x):
            return 0.0
        pmax = diffev2(lambda x: minBoundaryToConstraintOverTrajectory(x, t, SigInv), np.zeros(env_params['dim']), bounds = env_params['bounds'],penalty=pf,npop=40,disp=False,full_output=True)  
        pf0,_ = pFailoverHorizon(pmax[0],0,SigInv)
    
    
    pf0 = pf0*10*alpha/(1-alpha)
    R0 = (-np.inf,np.log10(pf0),0,Vs[1])
    
    Rs = []
    Rs.append(R0)
    env_params['v0'] = Vs[1]
    env_params['pf0'] = pf0
    for i in range(1,len(Vs[1:-1])):
        
        log10pmin = Rs[i-1][1]
        
        vpl = lambda x: Vs[i] - V(torch.tensor(x))
        vpu = lambda x: V(torch.tensor(x)) - Vs[i+1]
        
        @quadratic_inequality(safetyPenalty)
        @quadratic_inequality(vpl)
        @quadratic_inequality(vpu)
        def pf(x):
            return 0.0
        pmax = diffev2(lambda x: minBoundaryToConstraintOverTrajectory(x, tInd, SigInv,ballR=env_params['v0']), np.zeros(env_params['dim']), bounds = env_params['bounds'],penalty=pf,npop=40,disp=False,full_output=True)  
        pf_R,_ = pFailoverHorizon(pmax[0],0,SigInv,ballR=env_params['v0'])
        if pf_R+env_params['pf0'] > pfTarget:
            pfs = np.linspace(log10pmin,np.log10(pf_R+env_params['pf0']),numPfs)
            for j in range(len(pfs)-1):
                Rs.append((pfs[j],pfs[j+1],Vs[i],Vs[i+1]))
            Rs.append()
        else:
            Rs.append((log10pmin,np.log10(pf_R+env_params['pf0']),Vs[i],Vs[i+1]))
        
    vp = lambda x: vi - V(torch.tensor(x))
    @quadratic_inequality(vp)
    @quadratic_inequality(safetyPenalty)
    def pf(x):
        return 0.0
    
    pmin = diffev2(lambda x: -minBoundaryToConstraintOverTrajectory(x, tInd, SigInv,ballR=env_params['v0']), np.zeros(env_params['dim']), bounds = env_params['bounds'],penalty=pf,npop=40,disp=False,full_output=True)  
    pfmin,_ = pFailoverHorizon(pmin[0],0,SigInv,ballR=env_params['v0'])
    
    
    if np.log10(pfmin + env_params['pf0']) < Rs[-1][1]:
        log10pfmin = Rs[-1][1]
        if pfmin < pfTarget:
            pfs = np.linspace(log10pfmin,np.log10(pfTarget),numPfs)
            for j in range(len(pfs)-1):
                Rs.append((pfs[j],pfs[j+1],vi,np.inf))
            Rs.append((np.log10(pfTarget),0,vi,np.inf))
        else:
            Rs.append((np.log10(pfmin),0,vi,np.inf))
    else:
        Rs.append((np.log10(pfmin),0,vi,np.inf))
    
    return Rs

def regionDefinitionsLyapunov(X_t_to_tp1,fxt,t,pr0,ringNum):
    
    
    assert(len(X_t_to_tp1) == 2)    
    N = len(X_t_to_tp1[0])
    indR0 = int(np.ceil(N*pr0))    
    numSamplesForRegion = min(10000,indR0)
    
    xt = X_t_to_tp1[0]
    xtnext = X_t_to_tp1[1]
    
    rmaxTarget,xMaxNext = maxBoundaryToConstraintLyapunov(t+1)
    scores = scoreStateVecLyapunov(planner(xt,constraints(t)), fxt)
    ind = np.argsort(scores)
       
    
    ringBoundaries = np.zeros(ringNum + 2)
    ringBoundaries[1] = scores[ind[indR0]]
    p0 = 0.1
    N = len(xt)
    if np.floor(np.log10(N)) < ringNum:
        p = p0
        i = 2
        while p > 1000/N:
            ringBoundaries[i] = scores[ind[int(np.floor(N*(1-p)))]]
            i = i + 1
            p = p * p0
        ringBoundaries[i:ringNum+1] = np.linspace(scores[ind[-1]],rmaxTarget,ringNum-i+2)[1:]
    else:
        ringBoundaries[1:ringNum+1] = np.linspace(scores[ind[indR0]],rmaxTarget,ringNum)
        
    ringBoundaries[-1] = np.inf
    
    
    return ringBoundaries


def objectiveFun(x,t):
    
    d = int(len(x)/2)
    
    x_t = x[0:d]
    x_next = x[d:]
    
    c_t = constraints(t)
    r_t = planner(x_t,c_t)
    u_t = controller(x_t,r_t)
    xbar_next = model(x_t,u_t).numpy()
    
    s,Sig_t = noiseModel(x_t,u_t,r_t,t)
    Sig_tInv = np.linalg.inv(Sig_t)
    
    r2 = np.dot((x_next - xbar_next),np.dot(Sig_tInv,(x_next-xbar_next)))
    
    return r2


def f(x_t,t):
    c_t = constraints(t)
    r_t = planner(x_t,c_t)
    u_t = controller(x_t,r_t)
    xbar_next = model(x_t,u_t).numpy()
    return xbar_next


def overPwithStencil(Rdef,x,stencil,upperPobj,Nzero,t):
    
    
    pll = Rdef[0]
    plu = Rdef[1]
    vl  = Rdef[2]
    vu  = Rdef[3]
    
        
    N = len(stencil)
        
    C = constraints(t)
    r = planner(x,C)
    u = controller(x,r)
    xd = model(x,u).numpy()
    noiseSample,Cov = noiseModel(x, u, r, t)
    SigInv = np.linalg.inv(Cov)
    L = np.linalg.cholesky(Cov)
    xcs = stencil@L.T + xd
    Nj = 0
    for x in xcs:
        v =V(torch.tensor(x)).detach().numpy()[0]
        if v >= vl and v <=vu:
            if v <= env_params['v0']:
                pl = np.log10(env_params['pf0'])
            else:
                pl = np.log10(pFailoverHorizon(x,t+1, SigInv,ballR=env_params['v0'])[0]+env_params['pf0'])
            if pl >= pll and pl <= plu:
                Nj = Nj + 1
    
    # for s in stencil:
        
    #     xc = np.matmul(L,s) + xd
    #     r = scoreState((Q1,q1), f(xc,t+1))
        
    #     if r >= r10 and r <= r11:
    #         Nj = Nj + 1 
    
    if Nj == 0:
        pUp = Nzero/N
    else:
        pUp = sp.optimize.minimize_scalar(lambda p: np.abs(upperPobj(p,N,Nj)),bounds=(0,1),method='bounded').x
    
    if type(pUp) == np.ndarray:
        pUp = pUp[0]
        
    return pUp

def cumulativePfoverHorizon(x,SigInv,t,tH):
    pf = 0
    x_t = x
    for h in range(tH):
        r,_ =minBoundaryToConstraint(SigInv, x_t, t+h)
        pf = sp.stats.chi2.sf(r,env_params['dim'])*(1-pf) + pf
        
        c_t = constraints(t+h)
        r_t = planner(x_t,c_t)
        u_t = controller(x_t, r_t)
        x_t = model(x_t,u_t).numpy()
    
    return pf
    
            
def upperBoundQuasiMarkovTransitionMatrix(sourceScores,sourceRegionDefs, 
                                destinationScores,destinationRegionDefs,
                                t,Nzero,upperPobj,stencil):
    
    
    M = np.zeros((len(destinationRegionDefs),len(sourceRegionDefs)))
    sourceScoresPl,sourceScoresV = sourceScores
    destinationScoresPl,destinationScoresV = destinationScores
    
    Nsamples = len(stencil)
    d = env_params['dim']
    
    #decision variables are x at time t-1, and x at time t
    penaltySafety = lambda x: measureOfSafety(x[0:d], t)
    
    pChoices = []
    xOpts = []
    ns,Sig = noiseModel(np.zeros(env_params['dim']), 0, 0, 0) #because the noise is the same everywhere
    SigInv = np.linalg.inv(Sig)

    for i in range(len(destinationRegionDefs)):
        Rdef_d = destinationRegionDefs[i]
        
        # target/destination region definitions
        pll_d = Rdef_d[0] # lower boundary on failure probability level
        plu_d = Rdef_d[1] # upper boundary on failure probability level
        vl_d  = Rdef_d[2] # lower boundary on Lyapunov/progress measure level
        vu_d  = Rdef_d[3] # upper boundary on Lyapunov/progress measure level
        
        penaltyDestinationTargetPFLower = lambda x: pll_d - np.log10(pFailoverHorizon(x[d:],t+1,SigInv,ballR=env_params['v0'])[0] + env_params['pf0'])
        penaltyDestinationTargetPFUpper = lambda x: np.log10(pFailoverHorizon(x[d:],t+1,SigInv,ballR=env_params['v0'])[0]+ env_params['pf0']) - plu_d
        
        penaltyDestinationTargetVLower = lambda x: vl_d - V(torch.tensor(x[d:]).float()).detach().numpy()[0]
        penaltyDestinationTargetVUpper = lambda x: V(torch.tensor(x[d:]).float()).detach().numpy()[0] - vu_d
        
        
                
        pChoices_row = []
        xOpt_row = []
        for j in range(len(sourceRegionDefs)):
            Rdef_s = sourceRegionDefs[j]
            
            # origin/source region definitions
            pll_s = Rdef_s[0]
            plu_s = Rdef_s[1]
            vl_s  = Rdef_s[2]
            vu_s  = Rdef_s[3]
            
            penaltySourceTargetPFLower = lambda x: pll_s - np.log10(pFailoverHorizon(x[0:d],t,SigInv,ballR=env_params['v0'])[0]+ env_params['pf0'])
            penaltySourceTargetPFUpper = lambda x: np.log10(pFailoverHorizon(x[0:d],t,SigInv,ballR=env_params['v0'])[0] + env_params['pf0']) - plu_s
            
            penaltySourceTargetVLower = lambda x: vl_s - V(torch.tensor(x[0:d]).float()).detach().numpy()[0]
            penaltySourceTargetVUpper = lambda x: V(torch.tensor(x[0:d]).float()).detach().numpy()[0] - vu_s
            
            indSourcePl = np.logical_and(sourceScoresPl >= pll_s,sourceScoresPl <=plu_s)
            indSourceV = np.logical_and(sourceScoresV >= vl_s,sourceScoresV <= vu_s)
            indSource = np.logical_and(indSourcePl,indSourceV)
            Ns = sum(indSource)
            
            indDestinationPL = np.logical_and(destinationScoresPl[indSource] >= pll_d, destinationScoresPl[indSource] <= plu_d)
            indDestinationV = np.logical_and(destinationScoresV[indSource] >= vl_d,destinationScoresV[indSource] <vu_d)
            indDestionation = np.logical_and(indDestinationPL,indDestinationV)
            Nd = sum(indDestionation)
            
            #DMC count bloating
            pUp = sp.optimize.minimize_scalar(lambda p: np.abs(upperPobj(p,Ns,Nd)),bounds=(0,1),method='bounded').x
            if type(pUp) == np.ndarray:
                pUp = pUp[0]
            
            @quadratic_inequality(penaltySafety)
            @quadratic_inequality(penaltyDestinationTargetPFUpper)
            @quadratic_inequality(penaltyDestinationTargetPFLower)
            @quadratic_inequality(penaltyDestinationTargetVUpper)
            @quadratic_inequality(penaltyDestinationTargetVLower)
            @quadratic_inequality(penaltySourceTargetPFUpper)
            @quadratic_inequality(penaltySourceTargetPFLower)
            @quadratic_inequality(penaltySourceTargetVUpper)
            @quadratic_inequality(penaltySourceTargetVLower)
            

            def pf(x):
                return 0.0
            bounds = env_params['bounds']*2
            result = diffev2(lambda x: objectiveFun(x,t), x0=np.array([-5,0,-1,0,5,0,1,0]), bounds=bounds, penalty=pf, npop=40, disp=False, full_output=True)
            
            pOpt = sp.stats.chi2.sf(objectiveFun(result[0],t),env_params['dim'])
            
            xoptS = result[0][0:d]
            xoptD = result[0][d:]
            
            #Sample about optimal point
            if Ns < Nsamples: 
                pCount = overPwithStencil(Rdef_d, xoptS, stencil, upperPobj, Nzero, t)
            else:
                pCount = pOpt
            
            
            pChoices_row.append((pUp,pOpt,pCount))
            xOpt_row.append((xoptS,xoptD))
            
            M[i,j] = min(pUp,pOpt,pCount)
            
            with open(f'tx entries scratch/p{i}{j}_choices_and_opt.pkl','wb') as file:
                pickle.dump((pUp,pOpt,pCount,xoptS,xoptD))
        
        pChoices.append(pChoices_row)
        xOpts.append(xOpt_row)
    
    return M, pChoices,xOpts

# def upperBoundQuasiMarkovTransitionMatrixLyapunov(sourceScores,sourceRingBoundaries, 
#                                 destinationScores,destinationRingBoundaries,
#                                 t,Nzero,upperPobj,stencil):
    
    
#     M = np.zeros((len(destinationRingBoundaries)-1,len(sourceRingBoundaries)-1))
    
    
    
    
#     d = len(env_params['bounds']) #dimension of the system
#     Nsamples = len(stencil)
    
#     #decision variables are x at time t-1, and x at time t
#     penaltySafety = lambda x: measureOfSafety(x[0:d], t)
    
#     pChoices = []
#     xOpts = []
    
#     for i in range(len(destinationRingBoundaries)-1):
#         rid = destinationRingBoundaries[i]
#         rod = destinationRingBoundaries[i+1]
        
#         penaltyDestinationTargetLower = lambda x: rid - VCart(torch.tensor(f(x[d:],t+1)).float()).detach().numpy()[0]
#         penaltyDestinationTargetUpper = lambda x: VCart(torch.tensor(f(x[d:],t+1)).float()).detach().numpy()[0] - rod
                
#         pChoices_row = []
#         xOpt_row = []
#         for j in range(len(sourceRingBoundaries)-2):
#             ris = sourceRingBoundaries[j]
#             ros = sourceRingBoundaries[j+1]
            
#             penaltySourceTargetLower = lambda x: ris - VCart(torch.tensor(f(x[0:d],t)).float()).detach().numpy()[0]
#             penaltySourceTargetUpper = lambda x: VCart(torch.tensor(f(x[0:d],t))).detach().numpy()[0] - ros
            
#             indSource = np.logical_and(sourceScores >= ris,sourceScores <=ros)
#             Ns = sum(indSource)
#             Nd = sum(np.logical_and(destinationScores[indSource] >= rid, destinationScores[indSource] <= rod))
            
#             #DMC count bloating
#             pUp = sp.optimize.minimize_scalar(lambda p: np.abs(upperPobj(p,Ns,Nd)),bounds=(0,1),method='bounded').x
#             if type(pUp) == np.ndarray:
#                 pUp = pUp[0]
            
#             @quadratic_inequality(penaltySafety)
#             @quadratic_inequality(penaltyDestinationTargetLower)
#             @quadratic_inequality(penaltyDestinationTargetUpper)
#             @quadratic_inequality(penaltySourceTargetLower)
#             @quadratic_inequality(penaltySourceTargetUpper)

#             def pf(x):
#                 return 0.0
#             bounds = env_params['bounds']*2
#             result = diffev2(lambda x: objectiveFun(x,t), x0=np.zeros(len(bounds)), bounds=bounds, penalty=pf, npop=40, disp=False, full_output=True)
            
#             pOpt = sp.stats.chi2.sf(result[1],env_params['dim'])
            
#             xoptS = result[0][0:d]
#             xoptD = result[0][d:]
            
#             #Sample about optimal point
#             if Ns < Nsamples: 
#                 pCount = overPwithStencil((Qdd,cdd,rid,rod), xoptS, stencil, upperPobj, Nzero, t)
#             else:
#                 pCount = pOpt
            
            
#             pChoices_row.append((pUp,pOpt,pCount))
#             xOpt_row.append((xoptS,xoptD))
            
#             M[i,j] = min(pUp,pOpt,pCount)
        
#         pChoices.append(pChoices_row)
#         xOpts.append(xOpt_row)
    
#     return M, pChoices,xOpts

# def lowerBoundQuasiMarkovTransitionMatrix(sourceScores,sourceRingDef,sourceRingBoundaries, 
#                                 destinationScores,destinationRingDef,destinationRingBoundaries,
#                                 t,Nzero,lowerPobj,stencil):
    
    
#     M = np.zeros((len(destinationRingBoundaries)-1,len(sourceRingBoundaries)-1))
    
    
#     #region definition for targets at time t
#     Qsd = sourceRingDef[0]
#     csd = sourceRingDef[1]
    
#     #region definition for targets at time t+1
#     Qdd = destinationRingDef[0]
#     cdd = destinationRingDef[1]
    
#     d = len(csd) #dimension of the system
#     Nsamples = len(stencil)
    
#     #decision variables are x at time t-1, and x at time t
#     penaltySafety = lambda x: measureOfSafety(x[0:d], t)
    
#     pChoices = []
#     xOpts = []
    
#     for i in range(len(destinationRingBoundaries)-1):
#         rid = destinationRingBoundaries[i]
#         rod = destinationRingBoundaries[i+1]
        
#         penaltyDestinationTargetLower = lambda x: rid - scoreState((Qdd,cdd), f(x[d:],t+1))
#         penaltyDestinationTargetUpper = lambda x: scoreState((Qdd,cdd), f(x[d:],t+1)) - rod
                
#         pChoices_row = []
#         xOpt_row = []
#         for j in range(len(sourceRingBoundaries)-2):
#             ris = sourceRingBoundaries[j]*0.5
#             ros = sourceRingBoundaries[j+1]*0.5
            
#             penaltySourceTargetLower = lambda x: ris - scoreState(((Qsd,csd)), f(x[0:d],t))
#             penaltySourceTargetUpper = lambda x: scoreState(((Qsd,csd)), f(x[0:d],t)) - ros
            
#             indSource = np.logical_and(sourceScores >= ris,sourceScores <=ros)
#             Ns = sum(indSource)
#             Nd = sum(np.logical_and(destinationScores[indSource] >= rid, destinationScores[indSource] <= rod))
            
#             if Ns == 0:
#                 pLow = 0
#             #DMC count bloating
#             else:
#                 pLow = sp.optimize.minimize_scalar(lambda p: np.abs(lowerPobj(p,Ns,Nd)),bounds=(0,1),method='bounded').x
#                 if type(pLow) == np.ndarray:
#                     pLow = pLow[0]
                    
#             @quadratic_inequality(penaltySafety)
#             @quadratic_inequality(penaltyDestinationTargetLower)
#             @quadratic_inequality(penaltyDestinationTargetUpper)
#             @quadratic_inequality(penaltySourceTargetLower)
#             @quadratic_inequality(penaltySourceTargetUpper)

#             def pf(x):
#                 return 0.0
#             bounds = env_params['bounds']*2
#             result = diffev2(lambda x: -objectiveFun(x,t), x0=np.zeros(len(bounds)), bounds=bounds, penalty=pf, npop=40, disp=False, full_output=True)
            
#             pOpt = sp.stats.chi2.sf(-result[1],d)
            
#             xoptS = result[0][0:d]
#             xoptD = result[0][d:]
            
#             #Sample about optimal point
#             if Ns < Nsamples: 
#                 pCount = overPwithStencil((Qdd,cdd,rid,rod), xoptS, stencil, lowerPobj, Nzero, t)
#                 if pCount == Nzero/N:
#                     pCount = 0.0
#             else:
#                 pCount = pOpt
            
            
#             pChoices_row.append((pLow,pOpt,pCount))
#             xOpt_row.append((xoptS,xoptD))
            
#             M[i,j] = max(pLow,pOpt,pCount)
        
#         pChoices.append(pChoices_row)
#         xOpts.append(xOpt_row)
    
#     return M, pChoices,xOpts

def pFailoverHorizon(x_t,t,SigInv,pterm=1.0,ballR=0.1):
    xs = []
    xs.append(x_t)
    if measureOfSafety(x_t, t) > 0:
        return 1.0, xs
    
    v = V(torch.tensor(x_t)).detach().numpy()[0]
    pf = 0
    sf = 1
    while v > ballR and pf < pterm:
        c_t = constraints(t)
        r_t = planner(x_t, c_t)
        u_t = controller(x_t, r_t)
        x_t = model(x_t,u_t).numpy()
        xs.append(x_t)
        
        t = t + 1
        v = V(torch.tensor(x_t)).detach().numpy()[0]
        r,_ = minBoundaryToConstraint(SigInv, x_t, t)
        
        pf_t = sp.stats.chi2.sf(r,env_params['dim'])
        pfStep = pf_t*sf
        pf = pf + pfStep
        sf = sf*(1-pf_t)
    
    return pf,xs

def minBoundaryToConstraintOverTrajectory(xt,t,SigInv,ballR=0.1):
    if measureOfSafety(xt,t) > 0:
        return 0
    v = V(torch.tensor(xt)).detach().numpy()[0]
    if v <= ballR:
        return minBoundaryToConstraint(SigInv, f(xt, t),t)[0]
    vi = v
    t0 = t
    ts = []
    xs = []
    while v > ballR and v < 100*vi:
        ts.append(t)
        xs.append(xt)
        ct = constraints(t)
        rt = planner(xt, ct)
        ut = controller(xt, rt)
        xt = model(xt,ut).numpy()
        v = V(torch.tensor(xt)).detach().numpy()[0]
        t = t + 1
    
    if v >= 100*vi:
        rmin = 0
    elif len(ts) < 2:
        rmin = minBoundaryToConstraint(SigInv, xt, t)[0]
    else:
        trajXt = sp.interpolate.CubicSpline(np.array(ts),np.array(xs))
        rminRes = sp.optimize.minimize_scalar(lambda t: minBoundaryToConstraint(SigInv, trajXt(t), t)[0],bounds=(t0,ts[-1]))
        rmin = min(rminRes.fun,minBoundaryToConstraint(SigInv, xs[0], ts[0])[0],minBoundaryToConstraint(SigInv, xs[-1], ts[-1])[0])
    
    return rmin
        
def verifyProbablisticConstraint(Ec,X_0T,X_0Texp,pr0,ringNum):
    Qc,qc,rmax = Ec
    i = 0
    ind = len(X_0Texp)
    # verify that samples across the entire Monte Carlo simulation do not violate the probabilistic constraint
    # if they do, redefine the regions and re-start the check
    
    while(i < len(X_0T)):
        for xt in X_0T:
            stc = scoreStateVec((Qc,qc), xt)
            if max(stc) > rmax:
                td,rb = regionDefinitions(X_0T[i:i+2],X_0Texp[i],i,pr0,ringNum)
                ind = i
                i = 0
                break
            else:
                i = i + 1
    return td,rb,ind

def computeFailureProbability(p0,M,T,pUpSampling_t):
    
    p = p0
    p_t = []
    pfail_t = []
    pfail_t.append(p[-1])
    p_t.append(p)
    
    for t in range(1,T):
        p[-1] = 0
        p = np.matmul(M,p)
        j = 0
        for pi in p:
            if pi > pUpSampling_t[t][j]:
                p[j] = pUpSampling_t[t][j]
            j = j + 1
        
        p_t.append(p)
        pfail_t.append(p[-1])
    
    return p_t,pfail_t

def printMatrix(M):
    rows,cols = np.shape(M)
    text = []
    Mcolors = []
    for i in range(rows):
        textRow = []
        McolorsRow = []
        for j in range(cols):
            pij = M[i,j]
            McolorsRow.append(plt.cm.viridis(pij))
            textRow.append(f'{pij:.3g}')
        Mcolors.append(McolorsRow)
        text.append(textRow)
    fig,ax = plt.subplots(figsize=(10,10))
    Mtx = ax.matshow(M,alpha=0.2)
    ax.set_title('Transition Matrix')
    the_table = ax.table(cellText=text,cellColours = Mcolors,loc='center')
    for i in range(rows):
        for j in range(cols):
            the_table[i,j].set_height(1/cols)
    # plt.savefig(f'transitionMatrix_with_{ringNum}_rings.pdf')
    return 0
    

drawSamples = True
ifNewRingDef = True
newTxMatrices = False

T = 100
tH = 5
d = 4
N = int(1e6)
stencilLength = 10000
pr0 = 0.7
env_params['alpha'] = 0.05
env_params['v0'] = 15
ringNums = [3] #[5,10,15,20,25,35,45,50]
eps_f = 1e-6
ringNumV = 3
X0 = torch.zeros(N,4).uniform_(-0.25,0.25)
stencil = dist.MultivariateNormal(torch.zeros(d),torch.eye(d)).sample([stencilLength]).numpy()
# save the stencil and Monte Carlo Simulated Samples
if drawSamples:
    print(f'Running Monte Carlo Simulation upto time horizon {T+1}...')
    X_0T,X_0Texp = monteCarloSimulate(X0, T)
    with open('data/StdNormStencil_with_{stencilLength}_samples.pkl','wb') as file:
        pickle.dump(stencil, file)
    with open('data/DMCSamples_CartPoleStatic_to_timeHorizon_{T}.pkl','wb') as file:
        pickle.dump(X_0T,file)
    
else:
    with open('data/StdNormStencil_with_{stencilLength}_samples.pkl','rb') as file:
        stencil = pickle.load(file)
    with open('data/DMCSamples_CartPoleStatic_to_timeHorizon_{T}.pkl','rb') as file:
        X_0T = pickle.load(file)

gamma = 1e-9
Nzero = np.ceil(np.abs(-np.log(gamma)))
errTol = lambda x: 1-norm.cdf(x) - gamma
corr = np.abs(np.ceil(sp.optimize.fsolve(errTol,0)))
upperPobj = lambda p,Ni,Nj: Ni*p - corr*np.sqrt(Ni*p*(1-p))-Nj
lowerPobj = lambda p,Ni,Nj: Ni*p + corr*np.sqrt(Ni*p*(1-p))-Nj


for ringNum in ringNums:

        
    if ifNewRingDef:
        print(f'Computing scores and sampling based upper bounds...')
        
        L2Vars = []
        for xt in X_0T[:-2]:
            L2Vars.append(np.var(np.linalg.norm(xt,axis=1)))
        tInd = np.argmax(L2Vars)
        
        dummySample,Sig = noiseModel(np.zeros(env_params['dim']), controller(np.zeros(env_params['dim']),planner(np.zeros(env_params['dim']), constraints(tInd))), planner(np.zeros(env_params['dim']),constraints(tInd)), tInd)
        SigInv = np.linalg.inv(Sig)
        
        regionDefinitions = defineRegions(ringNum,ringNumV,1e-12,SigInv,tInd,0.99)
        
        # v0 = 15
        # alpha = 0.99
        # numPointsOnBoundary = 1000
        # c_tInd = constraints(tInd)
        # j = -1
        # pointsOnIsoLyapunov = np.zeros((stencilLength,env_params['dim']))
        # for i in range(numPointsOnBoundary):
        #     rp = dist.MultivariateNormal(torch.zeros(env_params['dim']),torch.eye(env_params['dim'])).sample().numpy()
        #     if np.mod(i,np.floor(numPointsOnBoundary/len(c_tInd))) == 0:
        #         j = j +1
        #         Hdef = c_tInd[j]
        #         a = Hdef[0]
        #         b = Hdef[1]
        #     gammaB = sp.optimize.minimize_scalar(lambda gamma: np.abs(np.dot(a,gamma*rp)-b))
        #     rdb = torch.tensor(gammaB.x*rp).float()
        #     gamma = sp.optimize.minimize_scalar(lambda gamma: np.abs(VCart(gamma*rdb).detach().numpy()[0]-v0))
        #     pointsOnIsoLyapunov[i,:] = gamma.x*rdb.numpy()
            
        
        # for i in range(numPointsOnBoundary,stencilLength):
        #     rp = dist.MultivariateNormal(torch.zeros(env_params['dim']),torch.eye(env_params['dim'])).sample()
        #     rd = rp/np.linalg.norm(rp)
            
        #     gamma = sp.optimize.minimize_scalar(lambda gamma: np.abs(VCart(gamma*rd).detach().numpy()[0] - v0))
            
        #     pointsOnIsoLyapunov[i,:] = gamma.x*rd.numpy()
        
        # Qv,qv = baseRegion(pointsOnIsoLyapunov)
        
        # VRange = np.zeros(stencilLength)
        # xEllSurface = np.zeros((stencilLength,env_params['dim']))
        # #pfSampling = np.zeros(stencilLength)
        # for i in range(stencilLength):
        #     rp = dist.MultivariateNormal(torch.zeros(env_params['dim']),torch.eye(env_params['dim'])).sample().numpy()
        #     rd = rp/np.linalg.norm(rp)
            
        #     gamma = sp.optimize.minimize_scalar(lambda gamma: np.abs(np.dot(gamma*rd-qv,np.matmul(Qv,gamma*rd-qv))-1))
        #     VRange[i] = VCart(torch.tensor(gamma.x*rd).float()).detach().numpy()[0]
        #     xEllSurface[i,:] = gamma.x*rd
        #     #pfSampling[i] = pFailoverHorizon(gamma.x*rd, 0, SigInv)
        
        # vp = lambda x: VCart(torch.tensor(x).float()).detach().numpy()[0] - v0
        # @quadratic_inequality(vp)
        # def pf(x):
        #     return 0.0
        # rminRes = diffev2(lambda x: minBoundaryToConstraintOverTrajectory(x, tInd, SigInv), xEllSurface[np.argmax(np.linalg.norm(xEllSurface,axis=1))], bounds = env_params['bounds'],penalty=pf,npop=40,disp=False,full_output=True)  
        
        # pf0,_ = pFailoverHorizon(rminRes[0],tInd,SigInv)
        # pf0 = pf0*10*(alpha/(1-alpha))
        # env_params['pf0'] = pf0
        
        # ringBoundariesPl = np.zeros(ringNum+3)
        # ringBoundariesPl[0] = -np.inf
        # ringBoundariesPl[-3] = np.log10(1e-3)
        # ringBoundariesPl[-2] = np.log10(0.5)
        # if np.log10(eps_f)-ringNum-1 > np.log10(pf0):
        #     ringBoundariesPl[1:ringNum] = np.linspace(np.log10(eps_f)-ringNum-1,np.log10(eps_f),ringNum-1)
        # else:
        #     ringBoundariesPl[1:ringNum] = np.linspace(np.log10(pf0),np.log10(eps_f),ringNum-1)
        
        
        xs = X_0T[tInd]
        log10pfs = np.ones(N)*np.log10(env_params['pf0'])
        sourceScoresV = scoreStateVecLyapunov(planner(X_0T[tInd],constraints(tInd)), X_0Texp[tInd])
        for i in range(len(xs)):
            vs = sourceScoresV[i]
            if vs > env_params['v0']:
                pf,_ = pFailoverHorizon(xs[i], tInd, SigInv,ballR=env_params['v0'])
                log10pfs[i] = np.log10(pf+env_params['pf0'])
                
        sourceScoresPf =  log10pfs
        
        sourceScores = (sourceScoresPf,sourceScoresV)
        
        
        xd = X_0T[tInd + 1]
        log10pfd = np.ones(N)*np.log10(env_params['pf0'])
        destinationScoresV = scoreStateVecLyapunov(planner(X_0T[tInd+1],constraints(tInd+1)), X_0Texp[tInd+1])
        for i in range(len(xd)):
            vd = destinationScoresV[i]
            if vd > env_params['v0']:
                pf,_ = pFailoverHorizon(xd[i], tInd+1, SigInv,ballR=env_params['v0'])
                log10pfd[i] = np.log10(pf+env_params['pf0'])
            
        destinationScoresPf = log10pfd
        
        destinationScores = (destinationScoresPf,destinationScoresV)
        
        
        # ringBoundariesV = np.zeros(ringNumV + 2)
        # rbVStart = v0
        # rbVMaxObs = max(sourceScoresV)
        # deltaVs = (rbVMaxObs - rbVStart)/(np.floor(ringNumV/2))

        # ringBoundariesV[1:ringNumV+1] = np.linspace(rbVStart,deltaVs*ringNumV,ringNumV)
        # ringBoundariesV[-1] = np.inf
        
        # # xbsDest = []
        # # xbsSource = []
        # # for fxt in X_0Texp[tInd]:
        # #     r,xb = minBoundaryToConstraint(SigInv, fxt, tInd)
        # #     destinationScores.append(sp.stats.chi2.sf(r,env_params['dim']))
        # #     xbsDest.append(xb)
            
        # # for xt in X_0T[tInd]:
        # #     r,xb = minBoundaryToConstraint(SigInv, xt, tInd)
        # #     sourceScores.append(sp.stats.chi2.sf(r,env_params['dim']))
        # #     xbsSource.append(xb)
           
        # # destinationScores = np.array(destinationScores)
        # # sourceScores = np.array(sourceScores)
        
      
        
        # regionDefinitions = []
        # regionDefinitions.append((ringBoundariesPl[0],ringBoundariesPl[1],0,v0))
        
        # oneStepPfUpperBound = []
        # oneStepPfLowerBound = []
        # for i in range(len(ringBoundariesPl)-1):
        #     for j in range(1,len(ringBoundariesV)-1):    
        #         regionDefinitions.append((ringBoundariesPl[i],ringBoundariesPl[i+1],ringBoundariesV[j],ringBoundariesV[j+1]))
        #         oneStepPfUpperBound.append(10**ringBoundariesPl[i+1])
        #         oneStepPfLowerBound.append(10**ringBoundariesPl[i])
        
        
        
        sourceRegionDefs = regionDefinitions
        destinationRegionDefs = regionDefinitions
        
        
        ############# placing this here for debugging
        
        Mu,pijChoicesU,xOptsU = upperBoundQuasiMarkovTransitionMatrix(sourceScores,sourceRegionDefs, 
                                destinationScores,destinationRegionDefs,
                                tInd,Nzero,upperPobj,stencil)
        
        printMatrix(Mu)
        
        with open(f'data/transitionMatrix_with_{ringNum}_rings_ApproxToApprox_to_time_{T}_upper.pkl','wb') as file:
            pickle.dump(Mu,file)
        # with open(f'data/transitionMatrix_with_{ringNum}_rings_ApproxToApprox_to_time_{T}_lower.pkl','wb') as file:
        #     pickle.dump(Ml,file)
        with open(f'data/transitionMatrixEntryChoices_with_{ringNum}_rings_ApproxToApprox_to_time_{T}_upper.pkl','wb') as file:
            pickle.dump(pijChoicesU,file)
        # with open(f'data/transitionMatrixEntryChoices_with_{ringNum}_rings_ApproxToApprox_to_time_{T}_lower.pkl','wb') as file:
        #     pickle.dump(pijChoicesL,file)
        with open(f'data/transitionMatrixEntryOptimalPoints_with_{ringNum}_rings_ApproxToApprox_to_time_{T}_upper.pkl','wb') as file:
            pickle.dump(xOptsU,file)
        
        #################################
        
        #targetDef,ringBoundaries = regionDefinitions(X_0T[0:2],X_0Texp[0],0,pr0,ringNum)
        #targetDef,ringBoundaries = regionDefinitions(X_0T[tInd:tInd+2],X_0Texp[tInd],tInd,pr0,ringNum)
        #targetDef,ringBoundaries,ind = verifyProbablisticConstraint(targetDef,X_0T,X_0Texp,pr0,ringNum)
        
        #ringBoundariesLyapunov = regionDefinitionsLyapunov(X_0T[tInd:tInd+2], X_0Texp[tInd], tInd, pr0, ringNum)
        t = 0
        
        scores_t = []
        for xt in X_0Texp:
            scoresLyap_t = scoreStateVecLyapunov(planner(X_0T[t],constraints(t)), xt)
            ind = scoresLyap_t > env_params['v0']
            scoresPL_t = np.ones(len(xt))*np.log10(env_params['pf0'])
            scoresUpdate = np.zeros(len(xt[ind]))
            i = 0
            for x in xt[ind]:
                pf,_ = pFailoverHorizon(x, t, SigInv,ballR=env_params['v0'])
                scoresUpdate[i] = np.log10(pf + env_params['pf0'])
                i = i + 1
            scoresPL_t[ind] = scoresUpdate     
            scores_t.append((scoresPL_t,scoresLyap_t))
            t = t + 1
        
        
        
        pUpSampling_t  = []
        pLowSampling_t = []
        pDMC_t = []
        
        
        for t in range(T):
            pU = []
            pL = []
            pDMC = []
            
            sourceScoresPL,sourceScoresLyap = scores_t[t]
            for i in range(len(sourceRegionDefs)):
                rs = sourceRegionDefs[i]
                rpl = rs[0]
                rpu = rs[1]
                rll = rs[2]
                rlu = rs[3]
                
                plCon = np.logical_and(sourceScoresPL >= rpl, sourceScoresPL <=rpu)
                lyapCon = np.logical_and(sourceScoresLyap >=rll, sourceScoresLyap <=rlu)
                
                Nj = sum(np.logical_and(plCon,lyapCon))
                if Nj == 0:
                    pU.append(Nzero/N)
                    pL.append(0)
                else:
                    pUp = sp.optimize.minimize_scalar(lambda p: np.abs(upperPobj(p,N,Nj)),bounds=(0,1),method='bounded').x
                    pLow = sp.optimize.minimize_scalar(lambda p: np.abs(lowerPobj(p,N,Nj)),bounds=(0,1),method='bounded').x
                    if type(pUp) == np.ndarray:
                        pU.append(pUp[0])
                    else:
                        pU.append(pUp)
                    if type(pLow) == np.ndarray:
                        pL.append(pLow[0])
                    else:
                        pL.append(pLow)
                pDMC.append(Nj/N)
            pUpSampling_t.append(pU)
            pLowSampling_t.append(pL)
            pDMC_t.append(pDMC)
        pUpSampling_t = np.array(pUpSampling_t)
        pLowSampling_t = np.array(pLowSampling_t)
        countUpSampling_t = pUpSampling_t*N
        countLowSampling_t = pLowSampling_t*N
        
    
        
        with open(f'data/regionDefinitions_CartPoleStatic_with_{ringNum}_rings_to_time_{T}.pkl','wb') as file:
            pickle.dump(regionDefinitions,file)
        with open(f'data/scores_CartPoleStatic_with_{ringNum}_rings_to_time_{T}.pkl','wb') as file:
            pickle.dump(scores_t,file)
        with open(f'data/DMCSampleUpperBounds_to_time_horizon_{T}.pkl','wb') as file:
            pickle.dump(pUpSampling_t,file)
        with open(f'data/DMCSampleLowerBounds_to_time_horizon_{T}.pkl','wb') as file:
            pickle.dump(pUpSampling_t,file)
        with open(f'data/DMCsamplePDist_to_time_horizon_{T}.pkl','wb') as file:
            pickle.dump(pDMC_t,file)
    else:
       
        with open(f'data/ringBoundaries_CartPoleStatic_with_{ringNum}_rings_to_time_{T}.pkl','rb') as file:
            ringBoundaries_t = pickle.load(file)
        with open(f'data/scores_CartPoleStatic_with_{ringNum}_rings_to_time_{T}.pkl','rb') as file:
            scores_t = pickle.load(file)
        with open(f'data/DMCSampleUpperBounds_to_time_horizon_{T}.pkl','rb') as file:
            pUpSampling_t = pickle.load(file)
        with open(f'data/DMCSampleLowerBounds_to_time_horizon_{T}.pkl','rb') as file:
            pLowSampling_t = pickle.load(file)
        with open(f'data/DMCsamplePDist_to_time_horizon_{T}.pkl','rb') as file:
            pDMC_t = pickle.load(file)
    
    if newTxMatrices:
        print(f'Working on transition matrices with {ringNum} number of rings...')
        quasiMarkovTransitionMatrices_t = []
        pijChoices_t = []
        xOpts_t = []
    

        Mu,pijChoicesU,xOptsU = upperBoundQuasiMarkovTransitionMatrix(sourceScores,sourceRegionDefs, 
                                destinationScores,destinationRegionDefs,
                                tInd,Nzero,upperPobj,stencil)
        
        # Mu,pijChoicesU,xOptsU = upperBoundQuasiMarkovTransitionMatrix(sourceScores,sourceRingDef,sourceRingBoundaries, 
        #                             destinationScores,destinationRingDef,destinationRingBoundaries,
        #                             t,Nzero,upperPobj,stencil)
        
        #Ml,pijChoicesL,xOptsL = lowerBoundQuasiMarkovTransitionMatrix(sourceScores,sourceRingDef,sourceRingBoundaries, 
        #                            destinationScores,destinationRingDef,destinationRingBoundaries,
        #                            t,Nzero,lowerPobj,stencil)
    
            
        #######################################
        #
        # These objects are arranged by row, column
        # so pijChoices[i][j] is the ith row jth column
        #
        ########################################
        Mu[:,-1] = 0.0
        Mu[-1,-1] = 1.0
        #Ml[:,-1] = 0.0
        #Ml[-1,-1] = 1.0
        printMatrix(Mu)
        #printMatrix(Ml)
        
        with open(f'data/transitionMatrix_with_{ringNum}_rings_ApproxToApprox_to_time_{T}_upper.pkl','wb') as file:
            pickle.dump(Mu,file)
        # with open(f'data/transitionMatrix_with_{ringNum}_rings_ApproxToApprox_to_time_{T}_lower.pkl','wb') as file:
        #     pickle.dump(Ml,file)
        with open(f'data/transitionMatrixEntryChoices_with_{ringNum}_rings_ApproxToApprox_to_time_{T}_upper.pkl','wb') as file:
            pickle.dump(pijChoicesU,file)
        # with open(f'data/transitionMatrixEntryChoices_with_{ringNum}_rings_ApproxToApprox_to_time_{T}_lower.pkl','wb') as file:
        #     pickle.dump(pijChoicesL,file)
        with open(f'data/transitionMatrixEntryOptimalPoints_with_{ringNum}_rings_ApproxToApprox_to_time_{T}_upper.pkl','wb') as file:
            pickle.dump(xOptsU,file)
        # with open(f'data/transitionMatrixEntryOptimalPoints_with_{ringNum}_rings_ApproxToApprox_to_time_{T}_lower.pkl','wb') as file:
        #     pickle.dump(xOptsL,file)
            
    else:    
        with open(f'data/transitionMatrix_with_{ringNum}_rings_ApproxToApprox_to_time_{T}.pkl','rb') as file:
            quasiMarkovTransitionMatrices_t = pickle.load(file)
        with open(f'data/transitionMatrixEntryChoices_with_{ringNum}_rings_ApproxToApprox_to_time_{T}.pkl','rb') as file:
            pijChoices_t = pickle.load(file)
        with open(f'data/transitionMatrixEntryOptimalPoints_with_{ringNum}_rings_ApproxToApprox_to_time_{T}.pkl','rb') as file:
            xOpts_t = pickle.load(file)
       
   
    #M = quasiMarkovTransitionMatrices_t[t]
    # numRows,numCols = np.shape(M)
    # for j in range(numCols):
    #     if sum(M[:,j]) < 1.0:
    #         rowInd = np.argmax(M[:,j])
    #         pijChoices = pijChoices_t[t][rowInd][j]
    #         if M[rowInd,j]  < pijChoices[0]:
    #             M[rowInd,j] = pijChoices[0]
    
    p0 = pDMC_t[0]
                
    p_t,pfail_t = computeFailureProbability(p0,Mu,T,pUpSampling_t)
    
    
    
    # #if pfail_t[-1] >= Nzero/N:
        

    
            

