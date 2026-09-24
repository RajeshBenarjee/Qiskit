"""
Classical Optimization Solvers for CSAPR:
1. Uniform Random Sampler (Mandatory Sanity Baseline)
2. Classical Simulated Annealing (QUBO Minimization)
"""

import time
import numpy as np
from typing import Dict, Any, Tuple
from repair_engine import repair_solution

class UniformRandomSampler:
    """
    Uniform Random Sampling baseline:
    Samples random binary bitstrings uniformly from {0, 1}^N to establish
    the statistical null hypothesis / sanity baseline for quantum and heuristic algorithms.
    """
    def __init__(self, num_vars: int, seed: int = 42):
        self.num_vars = num_vars
        self.rng = np.random.default_rng(seed)

    def sample(
        self,
        problem,
        Q: np.ndarray,
        offset: float,
        num_samples: int = 1000
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        best_feasible_cost = float("inf")
        best_feasible_vector = None
        best_overall_energy = float("inf")
        best_overall_vector = None
        
        feasible_count = 0
        samples_evaluated = 0

        for _ in range(num_samples):
            # Uniform random bitstring in {0, 1}^N
            x = self.rng.integers(0, 2, size=self.num_vars)
            energy = problem.evaluate_qubo_energy(x, Q, offset)
            cost = problem.evaluate_classical_objective(x)
            is_feas, details = problem.check_feasibility(x)
            
            samples_evaluated += 1

            if energy < best_overall_energy:
                best_overall_energy = energy
                best_overall_vector = x

            if is_feas:
                feasible_count += 1
                if cost < best_feasible_cost:
                    best_feasible_cost = cost
                    best_feasible_vector = x

        runtime = time.perf_counter() - t0
        feas_rate = feasible_count / max(1, num_samples)

        # Repair best sampled solution if no strictly feasible sample was found
        repaired_vector = None
        repaired_cost = None
        if best_feasible_vector is not None:
            final_vector = best_feasible_vector
            final_cost = best_feasible_cost
            final_feasible = True
        else:
            final_vector, _, rep_det = repair_solution(
                best_overall_vector,
                problem.n,
                problem.m,
                problem.demands,
                problem.capacities,
                problem.cost_matrix
            )
            final_cost = problem.evaluate_classical_objective(final_vector)
            final_feasible = rep_det["final_feasible"]

        return {
            "solver": "UniformRandomSampler",
            "samples": num_samples,
            "runtime_sec": round(runtime, 5),
            "feasibility_rate": round(feas_rate, 4),
            "best_raw_energy": round(best_overall_energy, 4),
            "final_objective": round(final_cost, 4),
            "final_vector": final_vector,
            "final_feasible": final_feasible
        }

class SimulatedAnnealingSolver:
    """
    Classical Simulated Annealing for QUBO minimization:
    Explores configuration space using single-bit flips with Metropolis-Hastings acceptance.
    """
    def __init__(
        self,
        num_vars: int,
        initial_temp: float = 100.0,
        cooling_rate: float = 0.98,
        max_steps: int = 5000,
        seed: int = 42
    ):
        self.num_vars = num_vars
        self.t_init = initial_temp
        self.cooling = cooling_rate
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)

    def solve(
        self,
        problem,
        Q: np.ndarray,
        offset: float
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        
        # Initialize randomly or with a greedy valid assignment
        current_x = np.zeros(self.num_vars, dtype=int)
        for i in range(problem.n):
            j_rand = self.rng.integers(0, problem.m)
            current_x[i * problem.m + j_rand] = 1
            
        current_energy = problem.evaluate_qubo_energy(current_x, Q, offset)
        
        best_x = current_x.copy()
        best_energy = current_energy
        temp = self.t_init

        step = 0
        while step < self.max_steps and temp > 1e-4:
            step += 1
            # Pick a variable to flip
            flip_idx = self.rng.integers(0, self.num_vars)
            candidate_x = current_x.copy()
            candidate_x[flip_idx] = 1 - candidate_x[flip_idx]
            
            candidate_energy = problem.evaluate_qubo_energy(candidate_x, Q, offset)
            delta_e = candidate_energy - current_energy

            # Metropolis acceptance criterion
            if delta_e < 0 or self.rng.random() < np.exp(-delta_e / max(temp, 1e-8)):
                current_x = candidate_x
                current_energy = candidate_energy
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_x = current_x.copy()

            temp *= self.cooling

        runtime = time.perf_counter() - t0
        raw_feas, raw_details = problem.check_feasibility(best_x)
        raw_cost = problem.evaluate_classical_objective(best_x)

        # Repair if infeasible
        if raw_feas:
            final_x = best_x
            final_cost = raw_cost
            final_feas = True
        else:
            final_x, _, rep_det = repair_solution(
                best_x,
                problem.n,
                problem.m,
                problem.demands,
                problem.capacities,
                problem.cost_matrix
            )
            final_cost = problem.evaluate_classical_objective(final_x)
            final_feas = rep_det["final_feasible"]

        return {
            "solver": "SimulatedAnnealing",
            "steps": step,
            "runtime_sec": round(runtime, 5),
            "best_energy": round(best_energy, 4),
            "raw_feasible": raw_feas,
            "final_objective": round(final_cost, 4),
            "final_vector": final_x,
            "final_feasible": final_feas
        }
