import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

"""

This program solves 1D heat equation using 
finite difference implicit method.

"""


def lhs_operator(N, sigma):
    """
    Computes the NxN matrix, A for Dirichlet boundary
    conditons.
    
    N: number of points on the grid.
    sigma: value of D*dt/dx**2
    """
    
    # setup the diagonal of the matrix
    Diagonal = np.diag((2.0+1.0/sigma)*np.ones(N))
    # Setup the upper diagonal of the operator.
    Upper_Diagonal = np.diag(-1.0*np.ones(N-1),k=1)
    # Setup the lower diagonal of the operator.
    Lower_Diagonal = np.diag(-1.0*np.ones(N-1), k=-1)
    # Assemble the operator
    A = Diagonal + Upper_Diagonal + Lower_Diagonal
    return A

def rhs_vector(u, sigma):
    """
    Computes the rhs column matrix b of the 
    1D heat equation using Dirichlet 
    conditions
    
    u: temperature as 1D array
    sigma: value of D*dt/dx**2
    """
    
    b = u[1:-1]/sigma
    # Set Dirichlet condition
    b[0] = b[0] + u[0]
    b[-1] = b[-1] + u[-1]
    return b

def btcs_implicit(u0, nt, dt, dx, D):
    """
    Computes the temperature along the rod 
    after a given number of time steps.
    
    u0: initial temperature array
    nt: number of time steps 
    dt: time step size
    dx: distance between consecutive grids
    D: diffusion coefficient
    """
    
    sigma = D*dt/dx**2
    # create an implicit operator of the system
    A = lhs_operator(len(u0)-2, sigma)
    # Integrate in time
    u = u0.copy()
    for n in range(nt):
        # Generate the rhs of the system
        b = rhs_vector(u, sigma)
        # solve the system
        u[1:-1] = linalg.solve(A,b)
    return u

def Plot_Field(x, u, nt):
    plt.figure(figsize=(6.0,4.0))
    plt.xlabel("Distance [m]")
    plt.ylabel("Temperature [C]")
    plt.plot(x, u, color="C0", linestyle="-",linewidth=2)
    plt.title("t=%d"%nt)
    plt.savefig("heat%d.png"%nt)

if __name__ == "__main__":

    # Set parameters.
    L = 1.0 # length of the rod
    nx = 51 # number of grids on the rod
    dx = L / (nx-1) # distance between the grid points
    D = 1.22e-3 # Diffusion coefficient

    # Define the grids on the rod
    x = np.linspace(0., L, nx)

    # Set the initial temperature along the rod.
    u0 = np.zeros(nx)
    u0[0] = 100.0
    u0[-1] = 0.0

    # Plot the initial field
    Plot_Field(x, u0, 0)

    # Set the time step 
    sigma = 0.5
    dt = sigma * dx**2 / D

    nt = 1000

    u = btcs_implicit(u0, nt, dt, dx, D)

    # Plot the final field.
    Plot_Field(x, u, nt)

