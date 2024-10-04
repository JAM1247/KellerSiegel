import jax
import jax.numpy as jnp
from jax import jit
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# Parameters
L = 10.0 # domain length
N = 64 # Change this to change Spatial Step Size 
dx = L / N # Spatial step size
x = jnp.linspace(0, L, N, endpoint=False)  # x axis 

dt = 5e-5 # time step
T = 5.0 # total simulation time
num_steps = int(T / dt) # timestep 

alpha = 5.0 # Chemotactic sensitivity
lambda_ = 1.0 # Diffusion length
rho0 = 1.0 # Average initial cell density

# Initial conditions
key = jax.random.PRNGKey(0)
eta = 0.01 * jax.random.normal(key, shape=(N,))
rho = rho0 + eta  # Initial cell density with small perturbation
w = jnp.zeros(N)  # Initial chemoattractant concentration


# Computing using central differences with periodic boundaries
def gradient(u):
    return (jnp.roll(u, -1) - jnp.roll(u, 1)) / (2 * dx)

# Computing using central differences with periodic boundaries
def laplacian(u):
    return (jnp.roll(u, -1) - 2 * u + jnp.roll(u, 1)) / dx**2

# Computing using central differences with periodic boundaries
def divergence(f):
    return gradient(f)


#Creating a Laplacian matrix with periodic boundary conditions.
def create_laplacian_matrix(N, dx, periodic=True):    
    diagonal = -2.0 * jnp.ones(N) / dx**2
    off_diagonal = jnp.ones(N - 1) / dx**2
    L = jnp.diag(diagonal) + jnp.diag(off_diagonal, k=1) + jnp.diag(off_diagonal, k=-1)
    if periodic:
        # Implementing the periodic boundary conditions
        L = L.at[0, -1].set(1.0 / dx**2)
        L = L.at[-1, 0].set(1.0 / dx**2)
    return L

# Initializing Laplacian and Identity matrices
L_matrix = create_laplacian_matrix(N, dx)
I_matrix = jnp.eye(N)

# Defining system matrices for implicit solves
A_rho = I_matrix - dt * L_matrix 
A_w = (lambda_**2) * L_matrix - I_matrix

# I used the upwind scheme to compute chemostatic flux, I was having trouble with the calculation ouptut and I found this to be a common solution on Numpy and Jax forums
@jit
def upwind_flux(rho, w):
    #I computed the chemotactic flux using an upwind scheme to prevent negative densities.
    grad_w = gradient(w)
    # Upwind scheme: Use rho[i-1] if grad_w >= 0, else use rho[i]
    flux = jnp.where(
        grad_w >= 0,
        alpha * jnp.roll(rho, 1) * grad_w, # Upwind value when grad_w >= 0
        alpha * rho * grad_w # Upwind value when grad_w < 0
    )
    return flux

# Time-stepping loops to compute Upwind Scheme and Mass Conservation
rho_list = [] # Array to hold rho values
t_list = [] # storing time values for graphing rho values 

for step in range(num_steps):
    # Computing chemotactic flux using Upwind Scheme
    flux = upwind_flux(rho, w)

    # Computing divergence of flux 
    div_flux = (flux - jnp.roll(flux, 1)) / dx

    # Computing RHS for rho update
    RHS_rho = rho - dt * div_flux

    # Solve for rho(t+1)
    rho_new = jnp.linalg.solve(A_rho, RHS_rho)

    # Solving for w(t+1)
    RHS_w = -rho
    w_new = jnp.linalg.solve(A_w, RHS_w)

    # Updating variables 
    rho = rho_new
    w = w_new

    # Enforcing  positiv densities and conserving mass
    negative_mass = -jnp.sum(rho[rho < 0]) * dx  
    rho = jnp.maximum(rho, 0.0)
    rho += negative_mass / L  # Redistributes negative mass uniformly

    # Checking for NaNs or Infs (There should no longer be any, this was mostly for debugging)
    if jnp.isnan(rho).any() or jnp.isinf(rho).any():
        print(f"NaN or Inf detected in rho at step {step}")
        break
    if jnp.isnan(w).any() or jnp.isinf(w).any():
        print(f"NaN or Inf detected in w at step {step}")
        break

    # Saving results for plotting every 100 steps
    if step % 100 == 0:
        rho_list.append(rho)
        t_list.append(step * dt)
        total_mass = jnp.sum(rho) * dx
        print(f"Step {step}/{num_steps}: Total Mass = {total_mass:.4f}")

# Converting results to NumPy arrays for graphing
rho_array = jnp.stack(rho_list)
t_array = jnp.array(t_list)

# Creating a meshgrid for plotting
x_np = np.array(x)
T_mesh_np = np.array(t_array)[:, np.newaxis]

# Plotting the kymograph with adjusted color scale
plt.figure(figsize=(12, 6))
pcm = plt.pcolormesh(
    x_np,
    T_mesh_np[:, 0],
    np.array(rho_array),
    shading='auto',
    cmap='viridis',
    vmin=0.95,
    vmax=1.05
)
plt.colorbar(pcm, label='Cell Density ρ(x, t)')
plt.xlabel('Spatial coordinate x')
plt.ylabel('Time t')
plt.title(f'Kymograph of Cell Density ρ(x, t) using SBDF1 Method\na = {alpha}, λ = {lambda_}')
plt.tight_layout()
plt.show()

# Plotting snapshots of rho, I mostly used this to monitor the equation during debugging but
# I'll leave it here because I think that it is helpful data to have
plt.figure(figsize=(12, 6))
for idx, time in enumerate(t_list):
    plt.plot(x_np, np.array(rho_array)[idx], label=f't={time:.2f}')
plt.xlabel('Spatial coordinate x')
plt.ylabel('Cell Density ρ(x)')
plt.title('Snapshots of Cell Density at Various Times')
plt.legend(loc='upper right', fontsize='small', ncol=2)
plt.tight_layout()
plt.show()
