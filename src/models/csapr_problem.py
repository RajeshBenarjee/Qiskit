"""
Capacitated Shelter Assignment Problem under Risk (CSAPR) Problem Model.
General formulation for n zones and m shelters.
Ensures consistent cost matrices and rigorous constraint checking across all solvers.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
import pulp

class CSAPRProblem:
    def __init__(
        self,
        zone_ids: List[str],
        zone_names: List[str],
        demands: List[float],
        shelter_ids: List[str],
        shelter_names: List[str],
        capacities: List[float],
        cost_matrix: np.ndarray,
        lambda_assignment: float = 100.0,
        lambda_capacity: float = 50.0
    ):
        self.zone_ids = zone_ids
        self.zone_names = zone_names
        self.demands = np.array(demands, dtype=float)
        self.shelter_ids = shelter_ids
        self.shelter_names = shelter_names
        self.capacities = np.array(capacities, dtype=float)
        self.cost_matrix = np.array(cost_matrix, dtype=float)
        
        self.n = len(zone_ids)
        self.m = len(shelter_ids)
        self.num_vars = self.n * self.m
        
        self.lambda_assign = lambda_assignment
        self.lambda_cap = lambda_capacity

    def var_index(self, i: int, j: int) -> int:
        return i * self.m + j

    def index_to_var(self, idx: int) -> Tuple[int, int]:
        return divmod(idx, self.m)

    def evaluate_classical_objective(self, x: np.ndarray) -> float:
        cost = 0.0
        for i in range(self.n):
            for j in range(self.m):
                idx = self.var_index(i, j)
                if x[idx] > 0.5:
                    cost += self.cost_matrix[i, j]
        return float(cost)

    def check_feasibility(self, x: np.ndarray) -> Tuple[bool, Dict[str, Any]]:
        assignment_violations = []
        assigned_shelters = {}
        for i in range(self.n):
            assigned = [j for j in range(self.m) if x[self.var_index(i, j)] > 0.5]
            if len(assigned) != 1:
                assignment_violations.append((i, self.zone_names[i], assigned))
            else:
                assigned_shelters[i] = assigned[0]

        capacity_violations = []
        shelter_loads = {j: 0.0 for j in range(self.m)}
        for i, j in assigned_shelters.items():
            shelter_loads[j] += self.demands[i]

        for j in range(self.m):
            if shelter_loads[j] > self.capacities[j] + 1e-5:
                overload = shelter_loads[j] - self.capacities[j]
                capacity_violations.append((j, self.shelter_names[j], shelter_loads[j], self.capacities[j], overload))

        is_feasible = (len(assignment_violations) == 0) and (len(capacity_violations) == 0)
        return is_feasible, {
            "assignment_violations": assignment_violations,
            "capacity_violations": capacity_violations,
            "shelter_loads": shelter_loads,
            "assigned_shelters": assigned_shelters
        }

    def solve_exact_milp(self) -> Dict[str, Any]:
        prob = pulp.LpProblem("CSAPR_Exact_MILP", pulp.LpMinimize)
        x_vars = {}
        for i in range(self.n):
            for j in range(self.m):
                x_vars[i, j] = pulp.LpVariable(f"x_{i}_{j}", cat=pulp.LpBinary)

        prob += pulp.lpSum(self.cost_matrix[i, j] * x_vars[i, j] for i in range(self.n) for j in range(self.m))

        for i in range(self.n):
            prob += pulp.lpSum(x_vars[i, j] for j in range(self.m)) == 1, f"OneHot_Zone_{i}"

        for j in range(self.m):
            prob += pulp.lpSum(self.demands[i] * x_vars[i, j] for i in range(self.n)) <= self.capacities[j], f"Capacity_Shelter_{j}"

        solver = pulp.PULP_CBC_CMD(msg=False)
        prob.solve(solver)

        status_str = pulp.LpStatus[prob.status]
        if status_str != "Optimal":
            return {"status": status_str, "feasible": False, "objective_value": None, "solution_vector": None}

        sol_vector = np.zeros(self.num_vars, dtype=int)
        for i in range(self.n):
            for j in range(self.m):
                if pulp.value(x_vars[i, j]) > 0.5:
                    sol_vector[self.var_index(i, j)] = 1

        obj_val = float(pulp.value(prob.objective))
        is_feas, details = self.check_feasibility(sol_vector)

        return {
            "status": status_str,
            "feasible": is_feas,
            "objective_value": obj_val,
            "solution_vector": sol_vector,
            "assignment_details": details
        }

    def build_qubo(self) -> Tuple[np.ndarray, float]:
        N = self.num_vars
        Q = np.zeros((N, N), dtype=float)
        offset = 0.0

        # 1. Base Linear Travel & Risk Cost
        for i in range(self.n):
            for j in range(self.m):
                u = self.var_index(i, j)
                Q[u, u] += self.cost_matrix[i, j]

        # 2. Assignment One-Hot Penalty: lambda_assign * sum_i (sum_j x_ij - 1)^2
        for i in range(self.n):
            offset += self.lambda_assign
            for j in range(self.m):
                u = self.var_index(i, j)
                Q[u, u] -= self.lambda_assign
                for k in range(j + 1, self.m):
                    v = self.var_index(i, k)
                    Q[u, v] += 2.0 * self.lambda_assign

        # 3. Capacity Overload Penalization:
        # Penalizes pairs (i, k) assigned to shelter j where co-location exceeds threshold
        for j in range(self.m):
            K_j = self.capacities[j]
            mean_demand = np.mean(self.demands)
            max_allowed = int(np.floor(K_j / max(1.0, mean_demand)))
            
            # Pairwise quadratic overload terms
            for i in range(self.n):
                for k in range(i + 1, self.n):
                    if self.demands[i] + self.demands[k] > K_j:
                        u = self.var_index(i, j)
                        v = self.var_index(k, j)
                        Q[u, v] += self.lambda_cap * 2.0
                    else:
                        # Soft concentration penalty
                        u = self.var_index(i, j)
                        v = self.var_index(k, j)
                        pair_weight = (self.demands[i] * self.demands[k]) / (K_j**2)
                        Q[u, v] += self.lambda_cap * pair_weight

        return Q, float(offset)

    def evaluate_qubo_energy(self, x: np.ndarray, Q: np.ndarray, offset: float) -> float:
        return float(x.T @ Q @ x + offset)
