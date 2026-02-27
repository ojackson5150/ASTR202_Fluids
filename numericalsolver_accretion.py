'''
@author: Ollie Jackson
@date: 02-25-2026

GAI Usage: Used ChatGPT to ask two conceptual questions about how to numerically solve
1. What do the boundary conditions "let the gas flow off the grid" mean?
2. What does "splitting the operator mean"

GAI output here: https://chatgpt.com/share/699fa484-573c-8007-b84c-718707df3a2e
'''

import numpy as np
import matplotlib.pyplot as plt

#using the Lax-Friedrich method for advection, implicit for diffusion

#advection parameters
Nsteps = 1500
dt = 0.1 #for stability making sure dt < dx/u
dx = 1
nu = 0.075 #viscosity

Ngrid = 50
x = np.arange(0.1, Ngrid*1., dx)/Ngrid #starting at 0.1 to avoid a radius of 0
alpha = -(9*nu / (4*x)) * dt/dx #derived in attached homework

#diffusion parameters
beta = 3*nu * dt / (dx)**2 #derived in attached homework
A = np.eye(Ngrid, k = 1) * (-beta) + np.eye(Ngrid)* (1+ 2*beta) + np.eye(Ngrid, k = -1) * (-beta)

#initial conditions
Sigma = np.exp(-((x - x.mean())**2)/(2*0.02**2)) #initial Gaussian function that is sharp - width = 0.02

#setting up figure
plt.ion()
fig, ax = plt.subplots(1,1)
pl, = ax.plot(x, Sigma, 'ro', markersize = 2)
ax.set_ylim([0,0.35])
ax.set_xlabel('x = r/R0')
ax.set_ylabel(r'$\Sigma$')
fig.canvas.draw()

for ct in range(Nsteps):

    #advection
    Sigma_advect = Sigma.copy()
    
    #uses alpha[1:-1] to match size of Sigma_advect[2:]
    #same implementation as used in in-class example
    Sigma_advect[1:Ngrid-1] = 0.5*(Sigma_advect[2:] + Sigma_advect[: -2]) - alpha[1:-1]*(Sigma_advect[2:] - Sigma_advect[:-2])

    #impose condition that gas flows off grid at both ends
    #Asked chatgpt what this boundary condition means - see attached GAI script
    #gas flowing off grid means that there is no spatial gradient at the boundaries

    Sigma_advect[0] = Sigma_advect[1]
    Sigma_advect[-1] = Sigma_advect[-2]
    
    #diffusion - sample implementation used in in-class example
    Sigma = np.linalg.solve(A, Sigma_advect)

    #impose boundary conditions again
    Sigma[0] = Sigma[1]
    Sigma[-1] = Sigma[-2]

    pl.set_ydata(Sigma)
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.01)
