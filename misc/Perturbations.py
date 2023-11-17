import numpy as np
from matplotlib import pyplot as plt
from scipy.linalg import expm


def get_pert(type, sd=1):

    if type == 'gaussian':
        A = np.random.normal(size=(3,3), scale=sd)
        B = np.random.normal(size=(3,3), scale=sd)
        C = A + 1j*B
        d = (C + np.conj(C.T))/2
    if type == 'diag':
        A = np.random.normal(size=(3), scale=sd)
        d = np.diag(A)
    return d


def calculate_pert_traj(X, H, d, l=.01, t=1000):
    # set parameters
    # sd : standard deviation, t : time passed, h : value of hbar, l : modulates size of perturbation
    h = 1
    eig, U = np.linalg.eig(H)
    times = np.linspace(0,t, num=1000)

    # create Hd, the perturbed Hamiltonian

    Hd = H + l*d

    # Change of basis into Hd eigenbasis
    eigd, Ud = np.linalg.eig(Hd)
    eigd = np.real(eigd)
    V = np.linalg.inv(Ud) @ U
    Vt = np.linalg.inv(V)
    Xd = Vt @ X @ V

    # Time evolution

    P = np.array([X.flatten()])
    for time in times:
        Hddiag = np.diag(eigd)
        evol = expm(-1j/h * Hddiag * time)
        F = np.linalg.inv(evol)@Xd@evol

        # Check how much error there is
        if np.linalg.norm(F - np.matrix(F).H) > .001:
            print('not hermitian')

        # Change of basis back to H eigenbasis
        F = V @ F @ Vt

        # formatting
        f = [F.flatten()]
        P = np.concatenate((P,f))

    return P


def graph_traj(X, P):
    data = P
    x=[]
    y=[]
    for k in range(9):
        real = [z.real for z in data[:,k]]
        imag = [z.imag for z in data[:,k]]
        x.append(real)
        y.append(imag)

    Xf = X.flatten()

    re = [z.real for z in Xf]
    im = [z.imag for z in Xf]


    samp_mean = np.mean(data, axis=0)

    figure, axis = plt.subplots(3, 3, figsize=(9.12,9.12))


    axis[0,0].plot(x[0],y[0], linewidth=.3, alpha=1)
    axis[0,0].plot(re[0],im[0], 'ro')


    axis[0,1].plot(x[1],y[1], linewidth=.3, alpha=1)
    axis[0,1].plot(re[1],im[1], 'ro')

    axis[0,2].plot(x[2],y[2], linewidth=.3, alpha=1)
    axis[0,2].plot(re[2],im[2], 'ro')

    axis[1,0].plot(x[3],y[3], linewidth=.3, alpha=1)
    axis[1,0].plot(re[3],im[3], 'ro')

    axis[1,1].plot(x[4],y[4], linewidth=.3, alpha=1)
    axis[1,1].plot(re[4],im[4], 'ro')

    axis[1,2].plot(x[5],y[5], linewidth=.3, alpha=1)
    axis[1,2].plot(re[5],im[5], 'ro')

    axis[2,0].plot(x[6],y[6], linewidth=.3, alpha=1)
    axis[2,0].plot(re[6],im[6], 'ro')

    axis[2,1].plot(x[7],y[7], linewidth=.3, alpha=1)
    axis[2,1].plot(re[7],im[7], 'ro')

    axis[2,2].plot(x[8],y[8], linewidth=.3, alpha=1)
    axis[2,2].plot(re[8],im[8], 'ro')

    re = [z.real for z in samp_mean]
    im = [z.imag for z in samp_mean]
    axis[0,0].plot(re[0],im[0], 'k+')
    axis[0,1].plot(re[1],im[1], 'k+')
    axis[0,2].plot(re[2],im[2], 'k+')
    axis[1,0].plot(re[3],im[3], 'k+')
    axis[1,1].plot(re[4],im[4], 'k+')
    axis[1,2].plot(re[5],im[5], 'k+')
    axis[2,0].plot(re[6],im[6], 'k+')
    axis[2,1].plot(re[7],im[7], 'k+')
    axis[2,2].plot(re[8],im[8], 'k+')



def calculate_pert_evol(X, H, d, basis='energy', l=.01, t=1e18):
    # set parameters
    # sd : standard deviation, t : time passed, h : value of hbar, l : modulates size of perturbation
    h=1
    eig, U = np.linalg.eig(H)


    # create Hd, the perturbed Hamiltonian
    Hd = H + l*d
    
    if basis == 'position':
        U = np.identity(3)

    # Change of basis into Hd eigenbasis
    eigd, Ud = np.linalg.eig(Hd)
    eigd = np.real(eigd)
    V = np.linalg.inv(Ud) @ U
    Vt = np.linalg.inv(V)
    Xd = Vt @ X @ V

    # Time evolution
    Hddiag = np.diag(eigd)
    evol = expm(-1j/h * Hddiag * t)
    F = np.linalg.inv(evol)@Xd@evol

    # Check how much error there is
    if np.linalg.norm(F - np.matrix(F).H) > .001:
        print('not hermitian')

    # Change of basis back to original basis
    F = V @ F @ Vt

    # formatting
    f = [F.flatten()]
    return f


def get_pert_dist(X, H, type, sample_size=1000, basis='energy'):
    P = np.array([X.flatten()])
    for i in range(sample_size):
        d = get_pert(type)
        pert = calculate_pert_evol(X, H, d, basis)
        P = np.concatenate((P,pert))
    return P



def graph_pert_dist(P, X, color=False):
    datak = P[1:,:]
    x=[]
    y=[]
    for k in range(9):
        real = [z.real for z in datak[:,k]]
        imag = [z.imag for z in datak[:,k]]
        x.append(real)
        y.append(imag)

    Xf = X.flatten()

    re = [z.real for z in Xf]
    im = [z.imag for z in Xf]

    samp_meank = np.mean(datak, axis=0)

    figure, axis = plt.subplots(3,3, figsize=(9.12,9.12))

    axis[0,0].scatter(x[0],y[0], alpha=.1)
    axis[0,0].plot(re[0],im[0], 'ro')


    axis[0,1].scatter(x[1],y[1], alpha=.1)
    axis[0,1].plot(re[1],im[1], 'ro')

    axis[0,2].scatter(x[2],y[2], alpha=.1)
    axis[0,2].plot(re[2],im[2], 'ro')

    axis[1,0].scatter(x[3],y[3], alpha=.1)
    axis[1,0].plot(re[3],im[3], 'ro')

    axis[1,1].scatter(x[4],y[4], alpha=.1)
    axis[1,1].plot(re[4],im[4], 'ro')

    axis[1,2].scatter(x[5],y[5], alpha=.1)
    axis[1,2].plot(re[5],im[5], 'ro')

    axis[2,0].scatter(x[6],y[6], alpha=.1)
    axis[2,0].plot(re[6],im[6], 'ro')

    axis[2,1].scatter(x[7],y[7], alpha=.1)
    axis[2,1].plot(re[7],im[7], 'ro')

    axis[2,2].scatter(x[8],y[8], alpha=.1)
    axis[2,2].plot(re[8],im[8], 'ro')


    re = [z.real for z in samp_meank]
    im = [z.imag for z in samp_meank]
    axis[0,0].plot(re[0],im[0], 'k+')
    axis[0,1].plot(re[1],im[1], 'k+')
    axis[0,2].plot(re[2],im[2], 'k+')
    axis[1,0].plot(re[3],im[3], 'k+')
    axis[1,1].plot(re[4],im[4], 'k+')
    axis[1,2].plot(re[5],im[5], 'k+')
    axis[2,0].plot(re[6],im[6], 'k+')
    axis[2,1].plot(re[7],im[7], 'k+')
    axis[2,2].plot(re[8],im[8], 'k+')