"""
Example demonstrating random load topology optimization.

This script shows how to:
1. Set up a problem with uncertain loads
2. Optimize design using BetaSolverRandomLoads
3. Analyze robustness to load variations
4. Compare with deterministic design
"""

import numpy
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from topopt.problems import MBBBeam
    from topopt.filters import DensityFilter
    from topopt.guis import MatplotlibGUI, NullGUI
    from topopt.solvers import (
        TopOptSolver,
        BetaSolverWithImplicitDiff,
        BetaSolverRandomLoads
    )
except ImportError as e:
    print(f"Error importing topopt: {e}")
    print("Make sure you're in the correct directory and topopt is installed")
    sys.exit(1)


def example_deterministic_optimization():
    """
    Example 1: Standard deterministic topology optimization.
    
    This is our baseline - design for nominal loads only.
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Deterministic Topology Optimization (Baseline)")
    print("="*70)
    
    # Create problem
    problem = MBBBeam(nelx=60, nely=30)
    
    # Setup solver
    filter = DensityFilter(problem, rmin=1.5)
    gui = NullGUI()
    
    solver = BetaSolverWithImplicitDiff(
        problem, volfrac=0.3, filter=filter, gui=gui,
        maxeval=100, learning_rate=0.01, n_samples=20
    )
    
    # Optimize
    x_init = numpy.ones(problem.nelx * problem.nely) * 0.3
    print("Optimizing deterministic design...")
    x_det = solver.optimize(x_init)
    
    # Evaluate on nominal load
    print("\nEvaluating deterministic design:")
    dobj_dummy = numpy.zeros_like(x_det)
    c_nominal = problem.compute_objective(x_det, dobj_dummy)
    print(f"  Compliance on nominal load: {c_nominal:.6f}")
    
    return problem, x_det, c_nominal


def example_robust_optimization():
    """
    Example 2: Robust topology optimization under load uncertainty.
    
    Designs to minimize expected compliance over uncertain loads.
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Robust Topology Optimization (Random Loads)")
    print("="*70)
    
    # Create problem
    problem = MBBBeam(nelx=60, nely=30)
    
    # Define load distribution: normal with 15% std dev
    load_dist_params = {
        'type': 'normal',
        'mean': problem.f.copy(),
        'std': 0.15 * numpy.abs(problem.f)  # 15% uncertainty
    }
    
    print("Load distribution:")
    print(f"  Type: Normal (Gaussian)")
    print(f"  Mean: nominal load")
    print(f"  Std Dev: 15% of mean")
    
    # Setup solver
    filter = DensityFilter(problem, rmin=1.5)
    gui = NullGUI()
    
    solver = BetaSolverRandomLoads(
        problem, volfrac=0.3, filter=filter, gui=gui,
        load_dist_params=load_dist_params,
        maxeval=100, learning_rate=0.01,
        n_design_samples=20, n_load_samples=10
    )
    
    # Optimize
    x_init = numpy.ones(problem.nelx * problem.nely) * 0.3
    print("\nOptimizing robust design (over 20 design samples × 10 load samples)...")
    x_robust = solver.optimize(x_init)
    
    # Get robustness statistics
    print("\nEvaluating robustness of design (1000 random load samples):")
    stats = solver.get_robust_statistics(n_eval_samples=1000)
    
    print(f"  Expected compliance: {stats['mean']:.6f}")
    print(f"  Std deviation: {stats['std']:.6f}")
    print(f"  Range: [{stats['min']:.6f}, {stats['max']:.6f}]")
    print(f"  95% confidence: [{stats['percentile_5']:.6f}, {stats['percentile_95']:.6f}]")
    print(f"  Coefficient of variation: {stats['std']/stats['mean']:.4f}")
    
    return problem, x_robust, stats


def example_comparison():
    """
    Example 3: Compare deterministic vs robust designs.
    
    Evaluates how well each design performs under load variations.
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Comparison - Deterministic vs Robust Design")
    print("="*70)
    
    problem, x_det, c_nom = example_deterministic_optimization()
    _, x_robust, stats_robust = example_robust_optimization()
    
    print("\n" + "-"*70)
    print("COMPARISON SUMMARY")
    print("-"*70)
    
    # Evaluate deterministic design on uncertain loads
    print("\nDeterministic Design Performance:")
    print(f"  On nominal load: {c_nom:.6f}")
    
    load_dist_params = {
        'type': 'normal',
        'mean': problem.f.copy(),
        'std': 0.15 * numpy.abs(problem.f)
    }
    
    # Sample and evaluate
    from topopt.solvers import _sample_load_distribution
    load_samples = _sample_load_distribution(load_dist_params, n_samples=1000)
    
    nominal_load = problem.f.copy()
    compliances_det = []
    for f in load_samples:
        problem.f = f
        dobj_dummy = numpy.zeros_like(x_det)
        c = problem.compute_objective(x_det, dobj_dummy)
        compliances_det.append(c)
    problem.f = nominal_load
    
    compliances_det = numpy.array(compliances_det)
    
    print(f"  Under uncertain loads (1000 samples):")
    print(f"    Mean: {compliances_det.mean():.6f}")
    print(f"    Std: {compliances_det.std():.6f}")
    print(f"    95% CI: [{numpy.percentile(compliances_det, 5):.6f}, {numpy.percentile(compliances_det, 95):.6f}]")
    print(f"    Worst case: {compliances_det.max():.6f}")
    
    print("\nRobust Design Performance:")
    print(f"  Under uncertain loads (1000 samples):")
    print(f"    Mean: {stats_robust['mean']:.6f}")
    print(f"    Std: {stats_robust['std']:.6f}")
    print(f"    95% CI: [{stats_robust['percentile_5']:.6f}, {stats_robust['percentile_95']:.6f}]")
    print(f"    Worst case: {stats_robust['max']:.6f}")
    
    # Compute improvement
    improvement = (compliances_det.mean() - stats_robust['mean']) / compliances_det.mean() * 100
    worst_case_imp = (compliances_det.max() - stats_robust['max']) / compliances_det.max() * 100
    std_reduction = (compliances_det.std() - stats_robust['std']) / compliances_det.std() * 100
    
    print("\nImprovements with Robust Design:")
    print(f"  Average compliance: {improvement:.2f}% worse (trade-off for robustness)")
    print(f"  Worst-case compliance: {worst_case_imp:.2f}% better")
    print(f"  Variability reduction: {std_reduction:.2f}%")


def example_different_load_distributions():
    """
    Example 4: Try different load distributions.
    
    Shows how to specify and optimize for different uncertainty models.
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Different Load Distributions")
    print("="*70)
    
    problem = MBBBeam(nelx=40, nely=20)
    filter = DensityFilter(problem, rmin=1.5)
    gui = NullGUI()
    
    # Try different distributions
    distributions = [
        {
            'name': 'Normal (±15%)',
            'params': {
                'type': 'normal',
                'mean': problem.f.copy(),
                'std': 0.15 * numpy.abs(problem.f)
            }
        },
        {
            'name': 'Uniform (±10%)',
            'params': {
                'type': 'uniform',
                'mean': problem.f.copy(),
                'scale': 0.10 * numpy.abs(problem.f)
            }
        },
    ]
    
    results = {}
    
    for dist_info in distributions:
        name = dist_info['name']
        params = dist_info['params']
        
        print(f"\nOptimizing for: {name}")
        
        solver = BetaSolverRandomLoads(
            problem, volfrac=0.3, filter=filter, gui=gui,
            load_dist_params=params,
            maxeval=50, learning_rate=0.01,
            n_design_samples=15, n_load_samples=8
        )
        
        x_init = numpy.ones(problem.nelx * problem.nely) * 0.3
        x_opt = solver.optimize(x_init)
        
        stats = solver.get_robust_statistics(n_eval_samples=500)
        results[name] = stats
        
        print(f"  Mean compliance: {stats['mean']:.6f}")
        print(f"  Std deviation: {stats['std']:.6f}")
        print(f"  Range: [{stats['percentile_5']:.6f}, {stats['percentile_95']:.6f}]")
    
    print("\n" + "-"*70)
    print("DISTRIBUTION COMPARISON")
    print("-"*70)
    for name, stats in results.items():
        print(f"{name}:")
        print(f"  Mean: {stats['mean']:.6f}")
        print(f"  CV: {stats['std']/stats['mean']:.4f}")


if __name__ == '__main__':
    print("\n" + "="*70)
    print("TOPOLOGY OPTIMIZATION WITH RANDOM LOADS")
    print("Demonstrating joint optimization of design and load robustness")
    print("="*70)
    
    try:
        # Run examples
        example_comparison()
        example_different_load_distributions()
        
        print("\n" + "="*70)
        print("All examples completed successfully!")
        print("="*70)
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
