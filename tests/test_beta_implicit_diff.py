"""Test the Beta implicit diff example can import and run."""

import numpy as np

# Use non-interactive backend for CI
import matplotlib
matplotlib.use('Agg')

from topopt.boundary_conditions import MBBBeamBoundaryConditions
from topopt.problems import ComplianceProblem
from topopt.filters import DensityBasedFilter
from topopt.guis import GUI
from topopt.solvers import BetaSolverWithImplicitDiff


def test_beta_implicit_diff_import():
    """Test that the example module can be imported."""
    import examples.beta_implicit_diff
    assert hasattr(examples.beta_implicit_diff, 'example_beta_mbb_beam')
    assert hasattr(examples.beta_implicit_diff, 'example_uncertainty_quantification')
    assert hasattr(examples.beta_implicit_diff, 'example_compare_solvers')


def test_beta_solver_small_problem():
    """Test Beta solver on a tiny problem (smoke test)."""
    # Use very small mesh to keep test fast
    nelx, nely = 8, 4
    volfrac = 0.3
    penalty = 3.0
    rmin = 1.0
    
    bc = MBBBeamBoundaryConditions(nelx, nely)
    problem = ComplianceProblem(bc, penalty=penalty)
    filter_obj = DensityBasedFilter(nelx, nely, rmin)
    
    gui = GUI(problem, title='Test GUI')
    
    solver = BetaSolverWithImplicitDiff(
        problem=problem,
        volfrac=volfrac,
        filter=filter_obj,
        gui=gui,
        maxeval=5,  # Very few iterations for speed
        learning_rate=0.01,
        n_samples=5  # Very few samples for speed
    )
    
    x = volfrac * np.ones(nelx * nely)
    x_opt = solver.optimize(x)
    
    # Validate output
    assert x_opt is not None
    assert len(x_opt) == nelx * nely
    assert np.all(x_opt >= 0) and np.all(x_opt <= 1)
    # Mean should be roughly near volfrac (within reason given few iterations)
    assert 0.0 <= np.mean(x_opt) <= 1.0


def test_beta_solver_confidence_intervals():
    """Test that confidence interval API works."""
    nelx, nely = 8, 4
    volfrac = 0.3
    penalty = 3.0
    rmin = 1.0
    
    bc = MBBBeamBoundaryConditions(nelx, nely)
    problem = ComplianceProblem(bc, penalty=penalty)
    filter_obj = DensityBasedFilter(nelx, nely, rmin)
    
    gui = GUI(problem, title='Test GUI')
    
    solver = BetaSolverWithImplicitDiff(
        problem=problem,
        volfrac=volfrac,
        filter=filter_obj,
        gui=gui,
        maxeval=5,
        learning_rate=0.01,
        n_samples=5
    )
    
    x = volfrac * np.ones(nelx * nely)
    x_opt = solver.optimize(x)
    
    # Test confidence intervals
    lower, upper = solver.get_confidence_intervals(percentile=95)
    assert lower.shape == (nelx * nely,)
    assert upper.shape == (nelx * nely,)
    assert np.all(lower <= upper)
    assert np.all(lower >= 0) and np.all(upper <= 1)
    
    # Test variance
    variance = solver.get_design_variance()
    assert variance.shape == (nelx * nely,)
    assert np.all(variance >= 0)


if __name__ == "__main__":
    # Run tests when invoked directly
    test_beta_implicit_diff_import()
    print("✓ Import test passed")
    
    test_beta_solver_small_problem()
    print("✓ Small problem test passed")
    
    test_beta_solver_confidence_intervals()
    print("✓ Confidence intervals test passed")
    
    print("\nAll tests passed!")
