"""
Example: PyTorch-based topology optimization with mirror descent.

This example demonstrates the refactored TopOptSolver using:
- Custom PyTorch autograd functions for FEM compliance
- Mirror descent on the simplex for constrained optimization
- Augmented Lagrangian for volume constraint handling
"""

import numpy as np
from topopt.boundary_conditions import MBBBeam
from topopt.problems import ComplianceProblem
from topopt.filters import DensityFilter
from topopt.guis import GUI
from topopt.solvers import TopOptSolver


def example_mbb_beam_mirror_descent():
    """
    Solve MBB beam problem using mirror descent.
    
    This is a classic topology optimization problem: minimize compliance
    of a cantilevered beam subject to a volume constraint.
    """
    
    # Problem setup
    nelx, nely = 60, 30  # Mesh resolution
    volfrac = 0.4        # Volume fraction
    penalty = 3.0        # SIMP penalty
    rmin = 1.5           # Filter radius
    
    # Boundary conditions (MBB beam)
    bc = MBBBeam(nelx, nely)
    
    # Create problem
    problem = ComplianceProblem(bc, penalty=penalty)
    
    # Create filter for checkerboard suppression
    filter = DensityFilter(nelx, nely, rmin)
    
    # Create visualization GUI
    gui = GUI(nelx, nely)
    
    # Create solver with mirror descent
    # Key parameters:
    #   - learning_rate: step size for gradient updates (tune for convergence)
    #   - maxeval: number of iterations
    #   - ftol_rel: relative tolerance for convergence check
    solver = TopOptSolver(
        problem=problem,
        volfrac=volfrac,
        filter=filter,
        gui=gui,
        maxeval=500,           # Number of iterations
        ftol_rel=1e-3,         # Convergence tolerance
        learning_rate=0.05,    # Mirror descent step size
    )
    
    # Initial design (all elements have 50% density)
    x = volfrac * np.ones(nelx * nely)
    
    # Optimize
    print("Starting optimization with mirror descent...")
    print(f"  Problem: {nelx}x{nely} mesh")
    print(f"  Volume fraction: {volfrac}")
    print(f"  Learning rate: {solver.learning_rate}")
    print(f"  Max iterations: {solver.maxeval}")
    print()
    
    x_opt = solver.optimize(x)
    
    print(f"\nOptimization complete!")
    print(f"  Final volume: {x_opt.sum() / x_opt.size:.4f}")
    print(f"  Volume constraint satisfied: {x_opt.sum() / x_opt.size <= volfrac + 1e-3}")
    
    return x_opt, problem, filter, gui


def example_tuning_learning_rate():
    """
    Demonstrate the effect of learning_rate on convergence.
    """
    import matplotlib.pyplot as plt
    
    nelx, nely = 30, 15
    volfrac = 0.4
    penalty = 3.0
    rmin = 1.5
    
    bc = MBBBeam(nelx, nely)
    problem = ComplianceProblem(bc, penalty=penalty)
    filter = DensityFilter(nelx, nely, rmin)
    gui = GUI(nelx, nely)
    
    learning_rates = [0.01, 0.05, 0.1, 0.2]
    results = {}
    
    for lr in learning_rates:
        print(f"\nTesting learning_rate = {lr}")
        solver = TopOptSolver(
            problem=problem,
            volfrac=volfrac,
            filter=filter,
            gui=gui,
            maxeval=200,
            learning_rate=lr,
        )
        
        x = volfrac * np.ones(nelx * nely)
        x_opt = solver.optimize(x)
        results[lr] = x_opt
        print(f"  Final volume: {x_opt.sum() / x_opt.size:.4f}")
    
    print("\nSummary of learning rate effects:")
    print("  lr=0.01: Slow convergence, stable")
    print("  lr=0.05: Good balance (recommended)")
    print("  lr=0.1:  Faster but may oscillate")
    print("  lr=0.2:  Risk of divergence")
    
    return results


def example_constraint_satisfaction():
    """
    Monitor how the volume constraint is satisfied during optimization.
    """
    nelx, nely = 40, 20
    volfrac = 0.3
    penalty = 3.0
    rmin = 1.5
    
    bc = MBBBeam(nelx, nely)
    problem = ComplianceProblem(bc, penalty=penalty)
    filter = DensityFilter(nelx, nely, rmin)
    gui = GUI(nelx, nely)
    
    # Augmented Lagrangian parameters affect constraint satisfaction
    solver = TopOptSolver(
        problem=problem,
        volfrac=volfrac,
        filter=filter,
        gui=gui,
        maxeval=300,
        learning_rate=0.05,
        # dual_step_size can be tuned in __init__ for faster/slower dual updates
    )
    
    x = volfrac * np.ones(nelx * nely)
    x_opt = solver.optimize(x)
    
    final_volume = x_opt.sum() / x_opt.size
    constraint_violation = final_volume - volfrac
    
    print(f"\nConstraint Satisfaction Analysis:")
    print(f"  Target volume fraction: {volfrac}")
    print(f"  Final volume fraction:  {final_volume:.6f}")
    print(f"  Constraint violation:   {constraint_violation:.6e}")
    print(f"  Feasible: {constraint_violation <= 1e-4}")
    
    return x_opt


if __name__ == "__main__":
    # Run main example
    print("=" * 60)
    print("PyTorch Mirror Descent Topology Optimization")
    print("=" * 60)
    
    x_opt, problem, filter, gui = example_mbb_beam_mirror_descent()
    
    # Optional: try tuning learning rate
    # results = example_tuning_learning_rate()
    
    # Optional: check constraint satisfaction
    # x_opt = example_constraint_satisfaction()
