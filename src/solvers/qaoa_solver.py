"""
Quantum Approximate Optimization Algorithm (QAOA) Solver for CSAPR.
Utilizes Qiskit 2.x and Qiskit-Aer with CVaR objective, depth p=1,2,3,
metrics profiling (depth, 2-qubit gates, evaluations), and feasibility repair.
"""

import time
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
from scipy.optimize import minimize

from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import Sampler

from repair_engine import repair_solution

def qubo_to_ising(Q: np.ndarray, offset: float) -> Tuple[SparsePauliOp, float]:
    """
    Transforms QUBO minimization: min x^T Q x + offset, x in {0, 1}^N
    to Ising Hamiltonian: H = sum_i h_i Z_i + sum_{i < j} J_{ij} Z_i Z_j + ising_offset
    via x_i = (I - Z_i) / 2.
    
    Qiskit Little-Endian convention: qubit 0 is the rightmost character in Pauli string.
    """
    N = Q.shape[0]
    # Ensure Q is symmetric
    Q_sym = (Q + Q.T) / 2.0
    
    linear_coeffs = np.zeros(N)
    for i in range(N):
        linear_coeffs[i] = Q[i, i]
        
    h = np.zeros(N)
    J = np.zeros((N, N))
    ising_offset = offset

    for i in range(N):
        # Q_ii * (I - Z_i) / 2
        ising_offset += 0.5 * Q[i, i]
        h[i] -= 0.5 * Q[i, i]
        
        for j in range(i + 1, N):
            q_ij = Q_sym[i, j]
            # 2 * q_ij * x_i * x_j = 2 * q_ij * (I - Z_i - Z_j + Z_i Z_j) / 4
            ising_offset += 0.5 * q_ij
            h[i] -= 0.5 * q_ij
            h[j] -= 0.5 * q_ij
            J[i, j] += 0.5 * q_ij

    pauli_list = []
    # Single-qubit Z terms
    for i in range(N):
        if abs(h[i]) > 1e-10:
            pauli_str = ["I"] * N
            pauli_str[N - 1 - i] = "Z"
            pauli_list.append(("".join(pauli_str), h[i]))
            
    # Two-qubit ZZ terms
    for i in range(N):
        for j in range(i + 1, N):
            if abs(J[i, j]) > 1e-10:
                pauli_str = ["I"] * N
                pauli_str[N - 1 - i] = "Z"
                pauli_str[N - 1 - j] = "Z"
                pauli_list.append(("".join(pauli_str), J[i, j]))

    if not pauli_list:
        pauli_list.append(("I" * N, 0.0))

    cost_op = SparsePauliOp.from_list(pauli_list)
    return cost_op, float(ising_offset)

class QAOACSAPRSolver:
    def __init__(
        self,
        num_qubits: int,
        p: int = 1,
        cvar_alpha: float = 0.50,
        shots: int = 2048,
        seed: int = 42
    ):
        self.num_qubits = num_qubits
        self.p = p
        self.cvar_alpha = max(0.01, min(1.0, cvar_alpha))
        self.shots = shots
        self.seed = seed
        self.backend = AerSimulator(method="statevector", seed_simulator=seed)
        self.sampler = Sampler(backend_options={"method": "statevector", "seed_simulator": seed})

    def solve(
        self,
        problem,
        Q: np.ndarray,
        offset: float,
        initial_point: Optional[np.ndarray] = None,
        max_iter: int = 60
    ) -> Dict[str, Any]:
        t_start = time.perf_counter()
        
        # 1. Map to Ising Hamiltonian
        t_ham_start = time.perf_counter()
        cost_op, ising_offset = qubo_to_ising(Q, offset)
        t_ham = time.perf_counter() - t_ham_start
        
        # 2. Build QAOA Ansatz Circuit
        ansatz = QAOAAnsatz(cost_operator=cost_op, reps=self.p)
        ansatz_measured = ansatz.copy()
        ansatz_measured.measure_all()
        
        # Transpile against target simulator to get realistic gate metrics
        pass_mgr = generate_preset_pass_manager(optimization_level=1, backend=self.backend)
        transpiled_circuit = pass_mgr.run(ansatz_measured)
        
        circuit_depth = transpiled_circuit.depth()
        gate_counts = transpiled_circuit.count_ops()
        two_qubit_gates = gate_counts.get("cx", 0) + gate_counts.get("cz", 0) + gate_counts.get("ecr", 0)
        
        # 3. Variational Optimization Loop
        evaluations_count = 0
        loss_history = []
        
        # Initial parameters: [gamma_1..gamma_p, beta_1..beta_p]
        if initial_point is None or len(initial_point) != 2 * self.p:
            rng = np.random.default_rng(self.seed)
            init_params = rng.uniform(0.0, np.pi, size=2 * self.p)
        else:
            init_params = np.array(initial_point, dtype=float)

        def loss_function(params: np.ndarray) -> float:
            nonlocal evaluations_count
            evaluations_count += 1
            
            # Execute circuit
            job = self.sampler.run(ansatz_measured, parameter_values=[params], shots=self.shots)
            result = job.result()
            quasi_dist = result.quasi_dists[0]
            
            # Compute energy of each sampled bitstring
            energies = []
            probs = []
            
            for int_state, prob in quasi_dist.items():
                # Convert integer to binary vector (length N)
                bit_str = format(int_state, f"0{self.num_qubits}b")
                # Qiskit Little-Endian: qubit 0 is bit_str[-1]
                x_vec = np.array([int(bit_str[self.num_qubits - 1 - i]) for i in range(self.num_qubits)])
                e_val = problem.evaluate_qubo_energy(x_vec, Q, offset)
                energies.append(e_val)
                probs.append(prob)
                
            energies = np.array(energies)
            probs = np.array(probs)
            
            # Sort by energy
            sort_idx = np.argsort(energies)
            sorted_e = energies[sort_idx]
            sorted_p = probs[sort_idx]
            
            # CVaR calculation
            cvar_prob_sum = 0.0
            cvar_energy = 0.0
            for e, p_val in zip(sorted_e, sorted_p):
                take_p = min(p_val, self.cvar_alpha - cvar_prob_sum)
                cvar_energy += take_p * e
                cvar_prob_sum += take_p
                if cvar_prob_sum >= self.cvar_alpha - 1e-9:
                    break
                    
            cvar_cost = cvar_energy / max(1e-6, cvar_prob_sum)
            loss_history.append(float(cvar_cost))
            return float(cvar_cost)

        t_opt_start = time.perf_counter()
        opt_res = minimize(
            loss_function,
            init_params,
            method="COBYLA",
            options={"maxiter": max_iter, "rhobeg": 0.5, "tol": 1e-3}
        )
        t_opt = time.perf_counter() - t_opt_start
        optimal_params = opt_res.x

        # 4. Final Sampling at Optimal Variational Parameters
        final_job = self.sampler.run(ansatz_measured, parameter_values=[optimal_params], shots=self.shots)
        final_dist = final_job.result().quasi_dists[0]

        best_sample_energy = float("inf")
        best_sample_x = None
        best_feasible_energy = float("inf")
        best_feasible_x = None
        
        feasible_sample_probability = 0.0
        
        sample_records = []
        for int_state, prob in final_dist.items():
            bit_str = format(int_state, f"0{self.num_qubits}b")
            x_vec = np.array([int(bit_str[self.num_qubits - 1 - i]) for i in range(self.num_qubits)])
            energy = problem.evaluate_qubo_energy(x_vec, Q, offset)
            cost = problem.evaluate_classical_objective(x_vec)
            is_feas, details = problem.check_feasibility(x_vec)
            
            if is_feas:
                feasible_sample_probability += prob
                if energy < best_feasible_energy:
                    best_feasible_energy = energy
                    best_feasible_x = x_vec
                    
            if energy < best_sample_energy:
                best_sample_energy = energy
                best_sample_x = x_vec
                
            sample_records.append((x_vec, energy, cost, is_feas, prob))

        t_total = time.perf_counter() - t_start

        # 5. Feasibility Repair
        if best_feasible_x is not None:
            final_solution_vector = best_feasible_x
            was_repaired = False
            final_feas = True
        else:
            final_solution_vector, was_repaired, rep_details = repair_solution(
                best_sample_x,
                problem.n,
                problem.m,
                problem.demands,
                problem.capacities,
                problem.cost_matrix
            )
            final_feas = rep_details["final_feasible"]

        final_obj = problem.evaluate_classical_objective(final_solution_vector)
        final_qubo_energy = problem.evaluate_qubo_energy(final_solution_vector, Q, offset)

        return {
            "solver": "QAOA",
            "p": self.p,
            "cvar_alpha": self.cvar_alpha,
            "shots": self.shots,
            "seed": self.seed,
            "runtime_total_sec": round(t_total, 4),
            "runtime_opt_sec": round(t_opt, 4),
            "evaluations_count": evaluations_count,
            "circuit_depth": circuit_depth,
            "two_qubit_gates": two_qubit_gates,
            "optimal_parameters": [round(float(v), 5) for v in optimal_params],
            "feasible_sample_probability": round(feasible_sample_probability, 4),
            "best_raw_energy": round(best_sample_energy, 4),
            "best_feasible_found": best_feasible_x is not None,
            "was_repaired": was_repaired,
            "final_solution_vector": final_solution_vector,
            "final_objective": round(final_obj, 4),
            "final_qubo_energy": round(final_qubo_energy, 4),
            "final_feasible": final_feas
        }
