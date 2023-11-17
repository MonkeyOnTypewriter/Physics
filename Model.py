import numpy as np
from matplotlib import pyplot as plt
from scipy.linalg import expm

class System:
    def __init__(self, initial_state='random', hamiltonian='triple well', basis='position', pdist='diag'):

        # set the initial state
        if initial_state == 'random':
            initial_state = np.random.normal(-1, 1, size=(3,)) + 1j*np.random.normal(-1, 1, size=(3,))
            initial_state = initial_state/np.linalg.norm(initial_state)

        self.D0 = np.outer(np.conj(initial_state),initial_state)

        # set the hamiltonian
        if hamiltonian == 'triple well':
            s = 1
            d = .5
            hamiltonian = np.array([[s,-d/3,-d/3],
                                    [-d/3,s,-d/3],
                                    [-d/3,-d/3,s]])
            
        self.H0 = hamiltonian

        # set the basis
        self.basis = basis

        # set the perturbation distribution
        self.pdist = pdist

        # initialize data
        self.Hd = None




        self.data = [self.D0.flatten()]

        if basis == 'energy':
            eig, U = np.linalg.eig(self.H0)
            Ut = np.linalg.inv(U)
            D0 = Ut @ self.D0 @ U
            self.data = [D0.flatten()]

    def perturb(self, strength=.01):

        if self.pdist == 'diag':
            A = np.random.normal(size=(3), scale=1)
            d = np.diag(A)

        self.Hd = self.H0 + strength*d
    

    
    def evolve(self, time):

        # Change of basis into Hd eigenbasis
        eigd, Ud = np.linalg.eig(self.Hd)
        eigd = np.real(eigd)
        V = np.linalg.inv(Ud)
        Vt = np.linalg.inv(V)
        D0_Hd = Vt @ self.D0 @ V

        Hddiag = np.diag(eigd)
        Evol = expm(-1j * Hddiag * time)
        Dt_Hd = np.linalg.inv(Evol)@ D0_Hd @Evol

        # Check how much error there is
        if np.linalg.norm(Dt_Hd - np.matrix(Dt_Hd).H) > .001:
            print('much error')

        # Change of basis back to position
        Dt_P = V @ Dt_Hd @ Vt

        # Change basis to desired basis
        if self.basis == 'energy':
            eig, U = np.linalg.eig(self.H0)
            Ut = np.linalg.inv(U)
            Dt_P = Ut @ Dt_P @ U

        # Add to data
        Dt_P = [Dt_P.flatten()]
        self.data = np.concatenate((self.data, Dt_P, ))


    
    def graph(self, type, title='', color=False, centered=False):

        x=[]
        y=[]
        for k in range(9):
            real = [z.real for z in self.data[:,k]]
            imag = [z.imag for z in self.data[:,k]]
            x.append(real)
            y.append(imag)

        D0 = self.D0.flatten()
        re = [z.real for z in D0]
        im = [z.imag for z in D0]
        

        figure, axis = plt.subplots(3, 3, figsize=(9,9))
        figure.suptitle(title)

        index = None
        if color == True:
            index = np.arange(len(self.data))

        if centered == True:

            D0 = self.D0.flatten()
            re = [z.real for z in D0]
            im = [z.imag for z in D0]

            for i in range(3):
                for j in range(3):
                    if i != j:
                        center_point = [0,0]
                        axis[i,j].set_xlim(center_point[0] - .5, center_point[0] + .5)
                        axis[i,j].set_ylim(center_point[1] - .5, center_point[1] + .5)
                    else:
                        center_point = [re[4*i],im[4*i]]
                        axis[i,j].set_xlim(center_point[0] - .5, center_point[0] + .5)
                        axis[i,j].set_ylim(center_point[1] - .5, center_point[1] + .5)


    
        if type == 'trajectory' and color == False:

            axis[0,0].plot(x[0],y[0], linewidth=.3)
            axis[0,1].plot(x[1],y[1], linewidth=.3)
            axis[0,2].plot(x[2],y[2], linewidth=.3)
            axis[1,0].plot(x[3],y[3], linewidth=.3)
            axis[1,1].plot(x[4],y[4], linewidth=.3)
            axis[1,2].plot(x[5],y[5], linewidth=.3)
            axis[2,0].plot(x[6],y[6], linewidth=.3)
            axis[2,1].plot(x[7],y[7], linewidth=.3)
            axis[2,2].plot(x[8],y[8], linewidth=.3)

        if type == 'trajectory' and color == True:
            axis[0,0].scatter(x[0],y[0], marker='.', alpha=1, c=index, cmap='viridis')
            axis[0,1].scatter(x[1],y[1], marker='.', alpha=1, c=index, cmap='viridis')
            axis[0,2].scatter(x[2],y[2], marker='.', alpha=1, c=index, cmap='viridis')
            axis[1,0].scatter(x[3],y[3], marker='.', alpha=1, c=index, cmap='viridis')
            axis[1,1].scatter(x[4],y[4], marker='.', alpha=1, c=index, cmap='viridis')
            axis[1,2].scatter(x[5],y[5], marker='.', alpha=1, c=index, cmap='viridis')
            axis[2,0].scatter(x[6],y[6], marker='.', alpha=1, c=index, cmap='viridis')
            axis[2,1].scatter(x[7],y[7], marker='.', alpha=1, c=index, cmap='viridis')
            axis[2,2].scatter(x[8],y[8], marker='.', alpha=1, c=index, cmap='viridis')

        if type == 'states':

            axis[0,0].scatter(x[0],y[0], alpha=.3, c=index, cmap='viridis')
            axis[0,1].scatter(x[1],y[1], alpha=.3, c=index, cmap='viridis')
            axis[0,2].scatter(x[2],y[2], alpha=.3, c=index, cmap='viridis')
            axis[1,0].scatter(x[3],y[3], alpha=.3, c=index, cmap='viridis')
            axis[1,1].scatter(x[4],y[4], alpha=.3, c=index, cmap='viridis')
            axis[1,2].scatter(x[5],y[5], alpha=.3, c=index, cmap='viridis')
            axis[2,0].scatter(x[6],y[6], alpha=.3, c=index, cmap='viridis')
            axis[2,1].scatter(x[7],y[7], alpha=.3, c=index, cmap='viridis')
            axis[2,2].scatter(x[8],y[8], alpha=.3, c=index, cmap='viridis')


        # plot initial state
        D0 = self.data[0,:]
        re = [z.real for z in D0]
        im = [z.imag for z in D0]

        axis[0,0].plot(re[0],im[0], 'ro')
        axis[0,1].plot(re[1],im[1], 'ro')
        axis[0,2].plot(re[2],im[2], 'ro')
        axis[1,0].plot(re[3],im[3], 'ro')
        axis[1,1].plot(re[4],im[4], 'ro')
        axis[1,2].plot(re[5],im[5], 'ro')
        axis[2,0].plot(re[6],im[6], 'ro')
        axis[2,1].plot(re[7],im[7], 'ro')
        axis[2,2].plot(re[8],im[8], 'ro')

        #plot sample mean
        samp_mean = np.mean(self.data[1:,:], axis=0)

        re = [z.real for z in samp_mean]
        im = [z.imag for z in samp_mean]
        axis[0,0].plot(re[0],im[0], 'kX')
        axis[0,1].plot(re[1],im[1], 'kX')
        axis[0,2].plot(re[2],im[2], 'kX')
        axis[1,0].plot(re[3],im[3], 'kX')
        axis[1,1].plot(re[4],im[4], 'kX')
        axis[1,2].plot(re[5],im[5], 'kX')
        axis[2,0].plot(re[6],im[6], 'kX')
        axis[2,1].plot(re[7],im[7], 'kX')
        axis[2,2].plot(re[8],im[8], 'kX')

    def save(self, file_name):
        plt.savefig(f'modl/{file_name}.png')
        plt.close()
        