"""
Example: Beta-distributed topology optimization with implicit differentiation.

This example demonstrates the BetaSolverWithImplicitDiff using:
- Beta distribution parameterization for each element
- Implicit differentiation through FEM analysis
- Uncertainty quantification via Beta parameters
"""

import numpy as np
from topopt.boundary_conditions import MBBBeamBoundaryConditions
from topopt.problems import ComplianceProblem
from topopt.filters import DensityBasedFilter
from topopt.guis import GUI
from topopt.solvers import BetaSolverWithImplicitDiff


def example_beta_mbb_beam():
    """
    Solve MBB beam using Beta-distributed design variables.
    
    Each element density ρ_e ~ Beta(α_e, β_e) where α_e, β_e are optimized
    via implicit differentiation, which avoids solving additional FEM systems.
    """
    
    # Problem setup
    nelx, nely = 60, 30
    volfrac = 0.4
    penalty = 3.0
    rmin = 1.5
    
    # Boundary conditions
    bc = MBBBeamBoundaryConditions(nelx, nely)
    
    # Create problem
    problem = ComplianceProblem(bc, penalty=penalty)
    
    # Create filter
    filter_obj = DensityBasedFilter(nelx, nely, rmin)
    
    # Create GUI
    gui = GUI(problem, title="Beta MBB Beam Example")
    
    # Create Beta solver with implicit differentiation
    solver = BetaSolverWithImplicitDiff(
        problem=problem,
        volfrac=volfrac,
        filter=filter_obj,
        gui=gui,
        maxeval=200,           # Typically needs fewer iterations than mirror descent
        learning_rate=0.01,    # Adam learning rate
        n_samples=100          # Samples for Monte Carlo expectation
    )
    
    # Initial design (not used by Beta solver, but required by interface)
    x = volfrac * np.ones(nelx * nely)
    
    # Optimize
    print("Starting optimization with Beta-distributed variables...")
    print(f"  Problem: {nelx}x{nely} mesh")
    print(f"  Volume fraction: {volfrac}")
    print(f"  Samples per iteration: {solver.n_samples}")
    print()
    
    x_opt = solver.optimize(x)
    
    print(f"\nOptimization complete!")
    print(f"  Final volume: {x_opt.mean():.4f}")
    print(f"  Volume constraint satisfied: {x_opt.mean() <= volfrac + 1e-3}")
    
    return x_opt, solver, problem, filter_obj, gui


def example_uncertainty_quantification():
    """
    Demonstrate uncertainty quantification in topology optimization.
    
    After optimization, we can extract confidence intervals and variance
    for each element's density, showing which design choices are uncertain.
    """
    
    nelx, nely = 40, 20
    volfrac = 0.3
    penalty = 3.0
    rmin = 1.5
    
    bc = MBBBeamBoundaryConditions(nelx, nely)
    problem = ComplianceProblem(bc, penalty=penalty)
    filter_obj = DensityBasedFilter(nelx, nely, rmin)
    gui = GUI(problem, title="UQ Example")
    
    solver = BetaSolverWithImplicitDiff(
        problem, volfrac, filter_obj, gui,
        maxeval=150,
        learning_rate=0.01,
        n_samples=100
    )
    
    x = volfrac * np.ones(nelx * nely)
    x_opt = solver.optimize(x)
    
    # Get confidence intervals
    lower, upper = solver.get_confidence_intervals(percentile=95)
    
    # Get variance
    variance = solver.get_design_variance()
    
    print("\nUncertainty Quantification Analysis:")
    print("=" * 60)
    print(f"\nDesign Statistics:")
    print(f"  Mean density: {x_opt.mean():.4f}")
    print(f"  Std deviation: {x_opt.std():.4f}")
    print(f"\n95% Confidence Intervals:")
    print(f"  Lower bound mean: {lower.mean():.4f}")
    print(f"  Upper bound mean: {upper.mean():.4f}")
    print(f"  Interval width: {(upper.mean() - lower.mean()):.4f}")
    print(f"\nVariance Statistics:")
    print(f"  Mean variance: {variance.mean():.6f}")
    print(f"  Max variance: {variance.max():.6f}")
    print(f"  Min variance: {variance.min():.6f}")
    
    # Identify uncertain elements (high variance)
    uncertain_idx = np.argsort(variance)[-10:]  # Top 10 most uncertain
    print(f"\nTop 10 most uncertain elements:")
    print(f"  Indices: {uncertain_idx}")
    print(f"  Densities: {x_opt[uncertain_idx]}")
    print(f"  Variances: {variance[uncertain_idx]}")
    
    return x_opt, solver


def example_compare_solvers():
    """
    Compare Beta solver with standard mirror descent.
    """
    from topopt.solvers import TopOptSolver
    
    nelx, nely = 30, 15
    volfrac = 0.4
    penalty = 3.0
    rmin = 1.5
    
    bc = MBBBeamBoundaryConditions(nelx, nely)
    problem = ComplianceProblem(bc, penalty=penalty)
    filter_obj = DensityBasedFilter(nelx, nely, rmin)
    gui = GUI(problem, title="Solver Comparison")
    
    x_init = volfrac * np.ones(nelx * nely)
    
    print("Comparing solvers on 30x15 MBB beam...")
    print("=" * 60)
    
    # Mirror descent solver
    print("\n1. Standard Mirror Descent:")
    solver_md = TopOptSolver(problem, volfrac, filter_obj, gui, 
                            maxeval=100, learning_rate=0.05)
    x_md = solver_md.optimize(x_init.copy())
    
    # Beta solver
    print("\n2. Beta with Implicit Differentiation:")
    solver_beta = BetaSolverWithImplicitDiff(
        problem, volfrac, filter_obj, gui,
        maxeval=100, learning_rate=0.01, n_samples=50
    )
    x_beta = solver_beta.optimize(x_init.copy())
    
    # Compare
    print("\n" + "=" * 60)
    print("Comparison:")
    print(f"  Mirror Descent volume: {x_md.mean():.4f}")
    print(f"  Beta volume: {x_beta.mean():.4f}")
    print(f"  Difference: {abs(x_md.mean() - x_beta.mean()):.6f}")
    
    # Get confidence intervals for Beta
    lower, upper = solver_beta.get_confidence_intervals(percentile=95)
    print(f"\n  Beta 95% CI: [{lower.mean():.4f}, {upper.mean():.4f}]")
    print(f"  Mirror descent within CI: {lower.mean() <= x_md.mean() <= upper.mean()}")
    
    return x_md, x_beta, solver_md, solver_beta


if __name__ == "__main__":
    print("=" * 70)
    print("Beta-Distributed Topology Optimization with Implicit Differentiation")
    print("=" * 70)
    
    # Run main example
    print("\n[Example 1: Basic Beta Solver]")
    x_opt, solver, problem, filter_obj, gui = example_beta_mbb_beam()
    
    # Run uncertainty quantification example
    print("\n\n[Example 2: Uncertainty Quantification]")
    x_opt_uq, solver_uq = example_uncertainty_quantification()
    
    # Run comparison example
    print("\n\n[Example 3: Solver Comparison]")
    x_md, x_beta, solver_md, solver_beta = example_compare_solvers()
    
    print("\n" + "=" * 70)
    print("Examples complete!")
    print("=" * 70)
