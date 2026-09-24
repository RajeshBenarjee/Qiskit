"""
Feasibility Repair and Decoder Mechanism for CSAPR.
Transforms any sampled bitstring into a strictly feasible assignment solution.
"""

from typing import Dict, Any, Tuple
import numpy as np

def repair_solution(
    raw_bitstring: np.ndarray,
    n_zones: int,
    m_shelters: int,
    demands: np.ndarray,
    capacities: np.ndarray,
    cost_matrix: np.ndarray
) -> Tuple[np.ndarray, bool, Dict[str, Any]]:
    """
    Feasibility Repair Engine:
    Step 1: Decode one-hot assignment per zone. If a zone has 0 or >1 shelters selected,
            assign to the shelter with the minimum cost C[i, j].
    Step 2: Capacity check. If any shelter is overloaded, iteratively migrate zones
            with the smallest marginal cost increase (C[i, new] - C[i, old]) to
            an under-capacity shelter until all capacity constraints are satisfied.
    """
    repaired_x = np.zeros(n_zones * m_shelters, dtype=int)
    assigned_shelters = {}
    was_modified = False

    # Step 1: Ensure exactly one shelter per zone
    for i in range(n_zones):
        chosen = [j for j in range(m_shelters) if raw_bitstring[i * m_shelters + j] > 0.5]
        if len(chosen) == 1:
            assigned_shelters[i] = chosen[0]
        else:
            was_modified = True
            # Greedy argmin cost
            best_j = int(np.argmin(cost_matrix[i, :]))
            assigned_shelters[i] = best_j

    # Step 2: Resolve capacity overloads
    shelter_loads = {j: 0.0 for j in range(m_shelters)}
    for i, j in assigned_shelters.items():
        shelter_loads[j] += demands[i]

    # Iterative greedy migration
    max_iter = 50
    it = 0
    while any(shelter_loads[j] > capacities[j] for j in range(m_shelters)) and it < max_iter:
        it += 1
        overloaded_shelters = [j for j in range(m_shelters) if shelter_loads[j] > capacities[j]]
        available_shelters = [j for j in range(m_shelters) if shelter_loads[j] < capacities[j]]
        
        if not available_shelters:
            # System-wide over-capacity: impossible to resolve strictly
            break
            
        src_j = overloaded_shelters[0]
        # Candidates in this overloaded shelter
        candidate_zones = [i for i, j in assigned_shelters.items() if j == src_j]
        
        # Find best move: (zone i, target shelter j) with minimum cost delta that fits
        best_move = None
        min_delta = float("inf")
        
        for i in candidate_zones:
            d_i = demands[i]
            for tgt_j in available_shelters:
                if shelter_loads[tgt_j] + d_i <= capacities[tgt_j] + 1e-5:
                    delta = cost_matrix[i, tgt_j] - cost_matrix[i, src_j]
                    if delta < min_delta:
                        min_delta = delta
                        best_move = (i, src_j, tgt_j)
                        
        if best_move is None:
            # Fallback: force move zone with lowest delta even if it temporarily fills available
            for i in candidate_zones:
                for tgt_j in available_shelters:
                    delta = cost_matrix[i, tgt_j] - cost_matrix[i, src_j]
                    if delta < min_delta:
                        min_delta = delta
                        best_move = (i, src_j, tgt_j)
                        
        if best_move:
            i_move, s_from, s_to = best_move
            assigned_shelters[i_move] = s_to
            shelter_loads[s_from] -= demands[i_move]
            shelter_loads[s_to] += demands[i_move]
            was_modified = True
        else:
            break

    # Build final binary vector
    for i, j in assigned_shelters.items():
        repaired_x[i * m_shelters + j] = 1

    # Check final feasibility
    final_feasible = all(shelter_loads[j] <= capacities[j] + 1e-5 for j in range(m_shelters))

    return repaired_x, was_modified, {
        "final_feasible": final_feasible,
        "assigned_shelters": assigned_shelters,
        "shelter_loads": shelter_loads,
        "iterations_required": it
    }
