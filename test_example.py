#!/usr/bin/env python
"""Test runner for beta_implicit_diff example with non-interactive backend."""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless testing

import numpy as np
from topopt.boundary_conditions import MBBBeamBoundaryConditions
from topopt.problems import ComplianceProblem
from topopt.filters import DensityBasedFilter
from topopt.guis import GUI
from topopt.solvers import BetaSolverWithImplicitDiff


def test_beta_solver_creation():
    """Test that we can create and use the Beta solver."""
    print("=" * 70)
    print("Test 1: Beta Solver Creation")
    print("=" * 70)
    
    nelx, nely = 10, 5
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
    gui = GUI(problem, title="Beta Solver Test")
    
    # Create solver
    solver = BetaSolverWithImplicitDiff(
        problem=problem,
        volfrac=volfrac,
        filter=filter_obj,
        gui=gui,
        maxeval=2,  # Just 2 iterations for quick test
        learning_rate=0.01,
        n_samples=5  # Small sample size
    )
    
    print(f"✓ Created solver: {solver}")
    print(f"  Problem: {nelx}x{nely} mesh with {problem.nel} elements")
    print(f"  Volume fraction: {volfrac}")
    print(f"  Samples per iteration: {solver.n_samples}")
    print()
    return True


def test_confidence_intervals():
    """Test that we can get confidence intervals."""
    print("=" * 70)
    print("Test 2: Confidence Intervals API")
    print("=" * 70)
    
    nelx, nely = 8, 4
    volfrac = 0.3
    penalty = 3.0
    rmin = 1.5
    
    bc = MBBBeamBoundaryConditions(nelx, nely)
    problem = ComplianceProblem(bc, penalty=penalty)
    filter_obj = DensityBasedFilter(nelx, nely, rmin)
    gui = GUI(problem, title="UQ Test")
    
    solver = BetaSolverWithImplicitDiff(
        problem=problem,
        volfrac=volfrac,
        filter=filter_obj,
        gui=gui,
        maxeval=2,
        learning_rate=0.01,
        n_samples=5
    )
    
    # Check that the API methods exist
    assert hasattr(solver, 'get_confidence_intervals'), "Missing get_confidence_intervals method"
    assert hasattr(solver, 'get_design_variance'), "Missing get_design_variance method"
    
    print("✓ Solver has confidence interval methods")
    
    # Test calling these methods
    try:
        lower, upper = solver.get_confidence_intervals(percentile=95)
        print(f"✓ get_confidence_intervals works")
        print(f"  Shape: lower={lower.shape}, upper={upper.shape}")
    except Exception as e:
        print(f"✗ get_confidence_intervals failed: {e}")
        return False
    
    try:
        variance = solver.get_design_variance()
        print(f"✓ get_design_variance works")
        print(f"  Shape: {variance.shape}")
    except Exception as e:
        print(f"✗ get_design_variance failed: {e}")
        return False
    
    print()
    return True


def test_solver_comparison():
    """Test that we can use TopOptSolver for comparison."""
    print("=" * 70)
    print("Test 3: TopOptSolver Comparison")
    print("=" * 70)
    
    from topopt.solvers import TopOptSolver
    
    nelx, nely = 8, 4
    volfrac = 0.4
    penalty = 3.0
    rmin = 1.5
    
    bc = MBBBeamBoundaryConditions(nelx, nely)
    problem = ComplianceProblem(bc, penalty=penalty)
    filter_obj = DensityBasedFilter(nelx, nely, rmin)
    gui = GUI(problem, title="Comparison Test")
    
    # Standard solver
    solver_std = TopOptSolver(
        problem=problem,
        volfrac=volfrac,
        filter=filter_obj,
        gui=gui,
        maxeval=2,
        learning_rate=0.05
    )
    print(f"✓ Created TopOptSolver: {solver_std}")
    
    # Beta solver
    solver_beta = BetaSolverWithImplicitDiff(
        problem=problem,
        volfrac=volfrac,
        filter=filter_obj,
        gui=gui,
        maxeval=2,
        learning_rate=0.01,
        n_samples=5
    )
    print(f"✓ Created BetaSolverWithImplicitDiff: {solver_beta}")
    
    print()
    return True


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Testing beta_implicit_diff example components")
    print("=" * 70 + "\n")
    
    try:
        success = True
        success = test_beta_solver_creation() and success
        success = test_confidence_intervals() and success
        success = test_solver_comparison() and success
        
        if success:
            print("=" * 70)
            print("✓ ALL TESTS PASSED")
            print("=" * 70)
        else:
            print("=" * 70)
            print("✗ SOME TESTS FAILED")
            print("=" * 70)
            exit(1)
    
    except Exception as e:
        print(f"\n✗ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
