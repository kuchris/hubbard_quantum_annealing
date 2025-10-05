"""
Hubbard Model Quantum Annealing Simulation
Based on: "Quantum speed-up for solving the one-dimensional Hubbard model using quantum annealing"
arXiv:2510.02141

This implementation simulates quantum annealing for the 1D Hubbard model using
classical quantum circuit simulation techniques.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.sparse import kron, identity, csr_matrix
from scipy.sparse.linalg import eigsh
import time
from typing import Tuple, List

class HubbardQuantumAnnealing:
    """
    Implements quantum annealing for the 1D Hubbard model following the approach
    described in the paper arXiv:2510.02141
    """
    
    def __init__(self, L: int, t_h: float = 1.0, U: float = 4.0):
        """
        Initialize the Hubbard model quantum annealing simulator.
        
        Args:
            L: Number of lattice sites
            t_h: Hopping parameter
            U: On-site interaction strength
        """
        self.L = L
        self.t_h = t_h
        self.U = U
        self.n_qubits = 2 * L  # spin-up and spin-down for each site
        
        # Pauli matrices
        self.I = np.array([[1, 0], [0, 1]], dtype=complex)
        self.X = np.array([[0, 1], [1, 0]], dtype=complex)
        self.Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        self.Z = np.array([[1, 0], [0, -1]], dtype=complex)
        
        # Pre-compute single-qubit operators for efficiency
        self._setup_operators()
    
    def _setup_operators(self):
        """Set up the many-body operators needed for the simulation."""
        # Create single-qubit operators for each qubit
        self.X_ops = []
        self.Y_ops = []
        self.Z_ops = []
        self.I_ops = []
        
        for i in range(self.n_qubits):
            # Create list of identity matrices
            ops_x = [self.I] * self.n_qubits
            ops_y = [self.I] * self.n_qubits
            ops_z = [self.I] * self.n_qubits
            ops_i = [self.I] * self.n_qubits
            
            # Replace with Pauli matrix at position i
            ops_x[i] = self.X
            ops_y[i] = self.Y
            ops_z[i] = self.Z
            
            # Create tensor products
            self.X_ops.append(self._tensor_product(ops_x))
            self.Y_ops.append(self._tensor_product(ops_y))
            self.Z_ops.append(self._tensor_product(ops_z))
            self.I_ops.append(self._tensor_product(ops_i))
    
    def _tensor_product(self, matrices):
        """Compute tensor product of list of matrices."""
        result = matrices[0]
        for mat in matrices[1:]:
            result = np.kron(result, mat)
        return result
    
    def _jordan_wigner_hopping(self, i: int, j: int, spin: str) -> np.ndarray:
        """
        Create hopping term using Jordan-Wigner transformation.
        
        Args:
            i, j: Site indices
            spin: 'up' or 'down'
        """
        if spin == 'up':
            qubit_i = i
            qubit_j = j
        else:  # spin down
            qubit_i = i + self.L
            qubit_j = j + self.L
        
        # Jordan-Wigner transformation: c†_i c_j = (X_i - iY_i)/2 * Π Z_k * (X_j + iY_j)/2
        # For simplicity, we'll use the XX + YY form from the paper
        
        if i < j:
            # String of Z operators between i and j
            z_string = self.I_ops[0]  # Start with identity
            for k in range(i, j):
                if spin == 'up':
                    z_string = z_string @ self.Z_ops[k]
                else:
                    z_string = z_string @ self.Z_ops[k + self.L]
        else:
            z_string = self.I_ops[0]  # Identity matrix
        
        # Hopping term: -(t_h/2) * (X_i X_j + Y_i Y_j) * Z_string
        hopping = -self.t_h/2 * (self.X_ops[qubit_i] @ self.X_ops[qubit_j] + 
                                self.Y_ops[qubit_i] @ self.Y_ops[qubit_j])
        
        return hopping
    
    def _interaction_term(self) -> np.ndarray:
        """Create on-site interaction term U * n_i_up * n_i_down."""
        interaction = np.zeros((2**self.n_qubits, 2**self.n_qubits), dtype=complex)
        
        for i in range(self.L):
            # n_i_up = (1 - Z_i)/2
            n_up = (self.I_ops[i] - self.Z_ops[i]) / 2
            # n_i_down = (1 - Z_{i+L})/2  
            n_down = (self.I_ops[i + self.L] - self.Z_ops[i + self.L]) / 2
            
            interaction += self.U * n_up @ n_down
        
        return interaction
    
    def hamiltonian(self, s: float) -> np.ndarray:
        """
        Create the annealing Hamiltonian at parameter s.
        
        Args:
            s: Annealing parameter (0 to 1)
        """
        H = np.zeros((2**self.n_qubits, 2**self.n_qubits), dtype=complex)
        
        # Hopping terms (always present)
        for i in range(self.L - 1):
            for spin in ['up', 'down']:
                H += self._jordan_wigner_hopping(i, i + 1, spin)
        
        # Interaction term (scaled by s)
        H += s * self._interaction_term()
        
        return H
    
    def prepare_initial_state(self, N_up: int, N_down: int) -> np.ndarray:
        """
        Prepare the initial ground state of the non-interacting Hamiltonian.
        For simplicity, we'll use a computational basis state with the right particle numbers.
        
        Args:
            N_up: Number of spin-up particles
            N_down: Number of spin-down particles
        """
        # Create a simple initial state with particles in the lowest available states
        state = np.zeros(2**self.n_qubits, dtype=complex)
        
        # For simplicity, put particles in the first N_up and N_down sites
        # This is a rough approximation of the true ground state
        basis_index = 0
        for i in range(N_up):
            basis_index += 2**i
        for i in range(N_down):
            basis_index += 2**(i + self.L)
        
        state[basis_index] = 1.0
        return state
    
    def evolve_step(self, state: np.ndarray, H: np.ndarray, dt: float) -> np.ndarray:
        """
        Evolve the state by one time step using the time evolution operator.
        
        Args:
            state: Current quantum state
            H: Hamiltonian
            dt: Time step
        """
        # U = exp(-i * H * dt)
        U = expm(-1j * H * dt)
        return U @ state
    
    def quantum_annealing(self, N_up: int, N_down: int, T_A: float, 
                         n_steps: int = 100) -> Tuple[np.ndarray, List[float]]:
        """
        Perform quantum annealing simulation.
        
        Args:
            N_up: Number of spin-up particles
            N_down: Number of spin-down particles  
            T_A: Total annealing time
            n_steps: Number of time steps
            
        Returns:
            final_state: Final quantum state after annealing
            energies: List of energies during evolution
        """
        dt = T_A / n_steps
        state = self.prepare_initial_state(N_up, N_down)
        energies = []
        
        print(f"Starting quantum annealing for L={self.L}, T_A={T_A}, steps={n_steps}")
        
        for step in range(n_steps):
            s = (step + 0.5) / n_steps  # Mid-point for better accuracy
            H = self.hamiltonian(s)
            
            # Calculate energy
            energy = np.real(np.conj(state) @ H @ state)
            energies.append(energy)
            
            # Evolve state
            state = self.evolve_step(state, H, dt)
            
            if step % 10 == 0:
                print(f"Step {step}/{n_steps}, s={s:.3f}, Energy={energy:.6f}")
        
        return state, energies
    
    def exact_ground_state_energy(self, N_up: int, N_down: int) -> float:
        """
        Calculate exact ground state energy using diagonalization (for small systems).
        
        Args:
            N_up: Number of spin-up particles
            N_down: Number of spin-down particles
        """
        H_final = self.hamiltonian(1.0)
        
        # For small systems, we can find the lowest eigenvalue
        if 2**self.n_qubits <= 2**12:  # Keep it manageable
            eigenvalues = eigsh(csr_matrix(H_final), k=1, which='SA', 
                              return_eigenvectors=False)
            return eigenvalues[0]
        else:
            print("System too large for exact diagonalization")
            return None
    
    def calculate_residual_energy(self, final_state: np.ndarray, 
                                exact_energy: float) -> float:
        """Calculate the residual energy after annealing."""
        H_final = self.hamiltonian(1.0)
        final_energy = np.real(np.conj(final_state) @ H_final @ final_state)
        return final_energy - exact_energy


def run_simulation():
    """Run a complete simulation demonstration."""
    print("=" * 60)
    print("Hubbard Model Quantum Annealing Simulation")
    print("Based on arXiv:2510.02141")
    print("=" * 60)
    
    # Parameters from the paper
    L = 4  # Start small for demonstration
    t_h = 1.0
    U = 4.0  # U/t_h = 4 as in the paper
    
    # Half-filled case: 2N_down = N = L
    N = L
    N_down = L // 2
    N_up = L - N_down
    
    print(f"System parameters: L={L}, U/t_h={U/t_h:.1f}")
    print(f"Particle numbers: N_up={N_up}, N_down={N_down}")
    
    # Create simulator
    simulator = HubbardQuantumAnnealing(L, t_h, U)
    
    # Get exact ground state energy
    print("\nCalculating exact ground state energy...")
    exact_energy = simulator.exact_ground_state_energy(N_up, N_down)
    if exact_energy is not None:
        print(f"Exact ground state energy: {exact_energy:.6f}")
    else:
        print("Cannot calculate exact energy for this system size")
        return
    
    # Test different annealing times
    T_A_values = [5, 10, 20, 40]
    residual_energies = []
    
    print(f"\nTesting different annealing times...")
    
    for T_A in T_A_values:
        print(f"\n--- T_A = {T_A} ---")
        start_time = time.time()
        
        final_state, energies = simulator.quantum_annealing(
            N_up, N_down, T_A, n_steps=50
        )
        
        residual = simulator.calculate_residual_energy(final_state, exact_energy)
        residual_energies.append(residual)
        
        elapsed = time.time() - start_time
        print(f"Final energy: {energies[-1]:.6f}")
        print(f"Residual energy: {residual:.6f}")
        print(f"Computation time: {elapsed:.2f} seconds")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    
    plt.subplot(1, 2, 1)
    plt.semilogy(T_A_values, np.abs(residual_energies), 'o-', linewidth=2, markersize=8)
    plt.xlabel('Annealing Time $T_A$')
    plt.ylabel('|Residual Energy|')
    plt.title('Quantum Annealing Performance')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    # Plot energy evolution for the best case
    best_T_A = T_A_values[np.argmin(np.abs(residual_energies))]
    final_state, energies = simulator.quantum_annealing(N_up, N_down, best_T_A, n_steps=100)
    
    plt.plot(energies, linewidth=2)
    plt.axhline(y=exact_energy, color='r', linestyle='--', label='Exact ground state')
    plt.xlabel('Time Step')
    plt.ylabel('Energy')
    plt.title(f'Energy Evolution (T_A = {best_T_A})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('hubbard_annealing_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n" + "=" * 60)
    print("Simulation Complete!")
    print(f"Best residual energy: {min(np.abs(residual_energies)):.6f}")
    print(f"Optimal annealing time: {T_A_values[np.argmin(np.abs(residual_energies))]}")
    print("Results saved to 'hubbard_annealing_results.png'")
    print("=" * 60)


if __name__ == "__main__":
    run_simulation()
