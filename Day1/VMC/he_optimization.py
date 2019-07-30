from slaterwf import ExponentSlaterWF
from wavefunction import JastrowWF,MultiplyWF
from hamiltonian import Hamiltonian
from metropolis import metropolis_sample
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
def func(x, a, b, c):
    return a*(x-b)**2+c
if __name__=="__main__":
    # These are good parameters to start with.
    nconfig=1000
    ndim=3
    nelec=2
    nstep=100
    tau=0.2

    # All the quantities we will keep track of.
    # You'll want to populate thes lists.
    #df={}
    #quantities=['kinetic','electron-nucleus','electron-electron']
    #for i in quantities:
    #    df[i]=[]
    #for i in ['alpha','beta','acceptance']:
    #    df[i]=[]
    ke = []
    vion = []
    vele = []
    potential = []
    eloc = []
    acc = []
    virial = []
    vele = []
    venu = []
    ham=Hamiltonian(Z=2) # Helium
    # Best Slater determinant.
    n = 51
    ast = 1.5
    aen = 2.5
    bst = -0.5
    ben = 1.5
    alphas = np.linspace(ast, aen, n)
    betas =  np.linspace(bst, ben, n)
    for alpha in alphas:
        ewf = ExponentSlaterWF(alpha=alpha)
        pos = np.random.randn(nelec, ndim, nconfig)
        pos, _ = metropolis_sample(pos=pos, wf=ewf, tau=tau, nstep=nstep)
        acc.append(_)
        ke.append(np.mean(-0.5*np.sum(ewf.laplacian(pos), axis=0)))
        vele.append(np.mean(ham.pot_ee(pos)))
        venu.append(np.mean(ham.pot_en(pos)))
        potential.append(np.mean(ham.pot(pos)))
        eloc.append(ke[-1]+potential[-1])
        virial.append(potential[-1]/ke[-1])
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    ax.plot(alphas, eloc)
    x = alphas
    popt, pcov = curve_fit(func, x, eloc)
    a, b, c = popt
    x = np.linspace(ast, aen, n*10)
    y = func(x, a, b, c)
    minidx = np.argmin(y)
    ax.plot(x,y)
    ax.axvline(x=x[minidx], color='k')
    ax.axhline(y=y[minidx], color='k')
    ax.set_xlabel("Alpha values")
    ax.set_ylabel("Local Energy (Ha)")
    ax.set_title("Alpha optimization")
    print("Min Energy: {}".format(y[minidx]))
    #.show()
    df = pd.DataFrame.from_dict({'kinetic': ke, 'electron-electron': vele, 'electron-nucleus': venu, 
                                 'potential': potential, 'eloc': eloc,
                                 'alpha': alpha, 'acceptance': acc, 'virial': virial})
    df.to_csv("helium_alpha.csv",index=False)
    ke = []
    vion = []
    vele = []
    potential = []
    eloc = []
    acc = []
    virial = []
    vele = []
    venu = []
    alph = []
    bet = []
    # Best Slater-Jastrow.
    for alpha in alphas:
        for beta in betas:
            alph.append(alpha)
            bet.append(beta)
            wf = MultiplyWF(ExponentSlaterWF(alpha), JastrowWF(beta))
            pos = np.random.randn(nelec, ndim, nconfig)
            pos, _ = metropolis_sample(pos=pos, wf=wf, tau=tau, nstep=nstep)
            acc.append(_)
            ke.append(np.mean(-0.5*np.sum(wf.laplacian(pos), axis=0)))
            vele.append(np.mean(ham.pot_ee(pos)))
            venu.append(np.mean(ham.pot_en(pos)))
            potential.append(np.mean(ham.pot(pos)))
            eloc.append(ke[-1]+potential[-1])
            virial.append(potential[-1]/ke[-1])

    df = pd.DataFrame.from_dict({'kinetic': ke, 'electron-electron': vele, 'electron-nucleus': venu, 
                                 'potential': potential, 'eloc': eloc,
                                 'alpha': alph, 'beta': bet, 'acceptance': acc, 'virial': virial})
    min_e = []
    for alpha in alphas:
        data = df.groupby('alpha').get_group(alpha)
        c = -0.5
        #print(a,b,c)
        #print(data.to_string())
        popt, pcov = curve_fit(func, data['beta'], data['eloc'], p0=[a,b,c])
        a, b, c = popt
        #x = np.linspace(ast, aen, n*10)
        y = func(betas, a, b, c)
        idxmin = np.argmin(y)
        min_e.append(y[idxmin])
        #print("Min Energy with beta: {}".format(data.iloc[idxmin, 'eloc']))
    fig = plt.figure(2)
    ax = fig.add_subplot(111)
    ax.plot(alphas, min_e)
    popt, pcov = curve_fit(func, alphas, min_e)
    a, b, c = popt
    x = np.linspace(ast, aen, n*10)
    y = func(x, a, b, c)
    idxmin = np.argmin(y)
    ax.plot(x,y)
    ax.axvline(x[idxmin])
    ax.axhline(y[idxmin])
    #df.groupby('alpha').get_group(alphas[5]).plot('alpha', 'eloc')
    #min_e = df.groupby('alpha').apply(lambda x: x.loc[x['eloc'].idxmin()])
    #print(min_e[['beta', 'eloc']].to_string())
    #min_e.plot('beta', 'eloc')
    plt.show()
    df.to_csv("helium_beta.csv",index=False)
            

