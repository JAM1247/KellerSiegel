import jax
import jax.numpy as jnp
from jax import jacfwd, jit
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import scipy.sparse

# Parameters
L = 10.0 # domain length
N = 64 # Change this to change Spatial Step Size 
dx = L / N # Spatial step size
x = jnp.linspace(0, L, N, endpoint=False)  # x axis 

dt = 1e-6 # time step
T = 0.05 # total simulation time
num_steps = int(T / dt)

alpha = 1.0 # Chemotactic sensitivity
lambda_ = 0.5 # Diffusion length

# Differential operators using periodic boundary conditions
def shift_left(u):
    return jnp.roll(u, -1)

def shift_right(u):
    return jnp.roll(u, 1)

@jit
def gradient(u):
    return (shift_left(u) - shift_right(u)) / (2 * dx)

@jit
def laplacian(u):
    return (shift_left(u) - 2 * u + shift_right(u)) / dx**2

def initialize_w(rho):
    # defining matrix
    A = lambda_**2 * construct_laplacian_matrix() - np.eye(N)
    rho_np = np.array(rho) # Converting rho to a NumPy array
    w = np.linalg.solve(A, -rho_np)
    w = jnp.array(w)  # Converting w back to a Numpy array
    return w

# Initializing state with spatially varying initial condition
def initialize_state(key):
    x_center = L / 2
    sigma = L / 10
    rho = 1.0 + 0.5 * jnp.exp(-((x - x_center) ** 2) / (2 * sigma ** 2))
    # Adding small random perturbations
    key, subkey = jax.random.split(key)
    eta = 0.05 * jax.random.normal(subkey, shape=(N,))
    rho += eta
    # Ensureing all densities are positive
    rho = jnp.maximum(rho, 0.0)
    # Initialize w by solving the steady-state chemoattractant equation
    w = initialize_w(rho)
    return jnp.concatenate([rho, w])

# Constructing the Laplacian matrix with periodic boundary conditions
def construct_laplacian_matrix():
    e = np.ones(N)
    diagonals = [e, -2 * e, e]
    offsets = [-1, 0, 1]
    laplacian_matrix = scipy.sparse.diags(diagonals, offsets, shape=(N, N), format='csr').toarray()
    # Applying periodic boundary conditions
    laplacian_matrix[0, -1] = 1
    laplacian_matrix[-1, 0] = 1
    laplacian_matrix /= dx**2
    return laplacian_matrix

# Defining the residual function using a conservative flux computation
@jit
def compute_flux_divergence(rho, w):
    # Computing chemotactic flux at cell interfaces
    grad_w = gradient(w)
    flux = alpha * rho * grad_w
    flux_interface = 0.5 * (flux + shift_left(flux))
    # Calculating divergence of flux using conservative form
    div_flux = (flux_interface - shift_right(flux_interface)) / dx
    return div_flux

@jit
def residual(u_new, rho_prev):
    rho_new, w_new = u_new[:N], u_new[N:]

    # Computing laplacian
    laplacian_rho_new = laplacian(rho_new)
    laplacian_w_new = laplacian(w_new)

    # Computing divergence of flux
    div_flux_new = compute_flux_divergence(rho_new, w_new)

    # Residual for rho and w
    f_rho = rho_new - rho_prev - dt * (-laplacian_rho_new + div_flux_new)
    f_w = lambda_**2 * laplacian_w_new - w_new + rho_new

    return jnp.concatenate([f_rho, f_w])

# Defining the Jacobian of the residual using automatic differentiation
jacobian_residual = jacfwd(lambda u, rho_prev: residual(u, rho_prev))

# Newton's method with damping and adjusted parameters
def newton_solver(u_guess, rho_prev, max_iter=50, tol=1e-5, initial_damping=1.0, damping_reduction=0.8, min_damping=1e-4, reg_param=1e-6):
    u = u_guess
    damping = initial_damping
    prev_res_norm = None
    for i in range(max_iter):
        res = residual(u, rho_prev)
        res_norm = jnp.linalg.norm(res)

        # Checking for NaNs or Infs (there shouldn't be any, this was for debugging but I'm leaving it here since it's helpful)
        if jnp.isnan(res_norm) or jnp.isinf(res_norm):
            print("Residual norm is NaN or Inf. Terminating Newton's method.")
            return u, res_norm

        # Checking for convergence
        if res_norm < tol:
            return u, res_norm

        # Checking for stagnation
        if prev_res_norm is not None and abs(prev_res_norm - res_norm) / res_norm < 1e-6:
            return u, res_norm

        prev_res_norm = res_norm

        # Computing Jacobian
        J = jacobian_residual(u, rho_prev)

        # Checking for NaNs or Infs in Jacobian
        if jnp.isnan(J).any() or jnp.isinf(J).any():
            print("Jacobian contains NaNs or Infs, Newton's method failded")
            return u, res_norm

        # Regularizing tne Jacobian
        J_reg = J + reg_param * jnp.eye(int(2 * N))

        # Solving for delta_u = J_reg * delta_u = res
        try:
            delta_u = jnp.linalg.solve(J_reg, res)
        except jnp.linalg.LinAlgError:
            # If singular, we use least squares
            delta_u = jnp.linalg.lstsq(J_reg, res, rcond=None)[0]
            print("    Jacobian is singular; using least squares solution.")

        # Tentative update with damping
        u_tentative = u - damping * delta_u

        # Computing residual for tentative update
        res_tentative = residual(u_tentative, rho_prev)
        res_tentative_norm = jnp.linalg.norm(res_tentative)

        # Checking if residual has decreased
        if res_tentative_norm < res_norm:
            # Accepting the tentative update
            u = u_tentative
            # Resetuing damping
            damping = initial_damping
        else:
            # Reducing damping
            damping *= damping_reduction
            if damping < min_damping:
                print(f"    Damping below minimum threshold at iteration {i}. Terminating Newton's method.")
                return u, res_norm  

    print(f"Newton's method did not converge within {max_iter} iterations. Final residual norm: {res_norm:.2e}")
    return u, res_norm

# Enforcing mass conservation
def enforce_mass_conservation(rho, total_mass_initial):
    total_mass_current = jnp.sum(rho) * dx
    mass_error = total_mass_current - total_mass_initial
    rho -= mass_error / L  # Adjusting densities uniformly
    return rho

# Time-stepping function
def time_step(u_prev, total_mass_initial):
    rho_prev = u_prev[:N]
    # Usinh previous solution as initial guess
    u_guess = u_prev

    # Solving for [rho_new, w_new] using Newton's method
    u_new, res_norm = newton_solver(u_guess, rho_prev)

    # Ensuring densities are positive and conserve tjeir mass
    rho_new = u_new[:N]
    negative_mass = jnp.sum(rho_new[rho_new < 0]) * dx
    rho_new = jnp.maximum(rho_new, 0.0)
    rho_new += negative_mass / L  # Redistributing negative mass uniformly
    rho_new = enforce_mass_conservation(rho_new, total_mass_initial)
    w_new = u_new[N:]
    u_new = jnp.concatenate([rho_new, w_new])

    return u_new

def run_simulation(initial_state, num_steps):
    step_interval = max(1, num_steps // 100) 
    num_snapshots = num_steps // step_interval + 1

    # Initializing snapshots
    rho_snapshots = np.zeros((num_snapshots, N))
    t_snapshots = np.zeros((num_snapshots,))

    # Setting initial snapshots
    rho_snapshots[0] = np.array(initial_state[:N])
    t_snapshots[0] = 0.0

    u_prev = initial_state
    snap_idx = 1

    total_mass_initial = jnp.sum(u_prev[:N]) * dx

    for step in range(1, num_steps + 1):
        u_new = time_step(u_prev, total_mass_initial)

        # Extracting rho_new
        rho_new = u_new[:N]

        # Monitoring total cell mass
        total_mass = jnp.sum(rho_new) * dx

        # Saving snapshots and printing mass at intervals (this was mostly for debugging when my mass wasn't conserved but I'll leave it)
        if step % step_interval == 0 or step == 1:
            print(f"Step {step}/{num_steps}, Total cell mass: {total_mass:.4f}")

        if step % step_interval == 0:
            rho_snapshots[snap_idx] = np.array(rho_new)
            t_snapshots[snap_idx] = step * dt
            snap_idx += 1

        u_prev = u_new

    return u_new, rho_snapshots, t_snapshots

# Main simulation loop and graphing
def main():
    key = jax.random.PRNGKey(0)
    initial_state = initialize_state(key)

    # Run the simulation
    print("Starting simulation...")
    final_state, rho_snapshots, t_snapshots = run_simulation(initial_state, num_steps)
    print("Simulation complete.")

    # Convert snapshots to NumPy arrays for plotting
    rho_array = rho_snapshots
    t_array = t_snapshots

    # Create a meshgrid for plotting
    X, T_mesh = np.meshgrid(np.array(x), t_array)

    # Plot the kymograph with adjusted color scale
    plt.figure(figsize=(12, 6))
    pcm = plt.pcolormesh(X, T_mesh, rho_array, shading='auto', cmap='viridis', vmin=np.min(rho_array), vmax=np.max(rho_array))
    plt.colorbar(pcm, label='Cell Density ρ(x, t)')
    plt.xlabel('Spatial coordinate x')
    plt.ylabel('Time t')
    plt.title(f'Kymograph of Cell Density ρ(x, t)\nα = {alpha}, λ = {lambda_}')
    plt.tight_layout()
    plt.show()

    # Plot snapshots at various times
    plt.figure(figsize=(12, 6))
    num_snapshots = rho_array.shape[0]
    snapshot_indices = np.linspace(0, num_snapshots - 1, num=5).astype(int)

    for idx in snapshot_indices:
        plt.plot(x, rho_array[idx], label=f't = {t_array[idx]:.5f}')

    plt.xlabel('Spatial coordinate x')
    plt.ylabel('Cell Density ρ(x, t)')
    plt.title('Snapshots of Cell Density ρ(x, t) at Various Times')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
