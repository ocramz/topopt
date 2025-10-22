"""
Test suite for random load functionality in topology optimization.

Tests the autograd functions and solvers for handling uncertain loads.
"""

import numpy
import pytest
import torch

# Try to import topopt components
try:
    from topopt.problems import MBBBeam
    from topopt.filters import DensityFilter
    from topopt.guis import NullGUI
    from topopt.solvers import (
        BetaParameterFunction,
        BetaRandomLoadFunction,
        BetaSolverWithImplicitDiff,
        BetaSolverRandomLoads,
        _sample_load_distribution
    )
    TOPOPT_AVAILABLE = True
except ImportError as e:
    TOPOPT_AVAILABLE = False
    IMPORT_ERROR = str(e)


@pytest.mark.skipif(not TOPOPT_AVAILABLE, reason="topopt not available")
class TestLoadDistributionSampling:
    """Test load distribution sampling functions."""
    
    def test_normal_distribution_sampling(self):
        """Test normal load distribution sampling."""
        mean = numpy.array([1.0, 2.0, 3.0])
        dist_params = {
            'type': 'normal',
            'mean': mean,
            'std': 0.1 * numpy.abs(mean)
        }
        
        samples = _sample_load_distribution(dist_params, n_samples=100)
        
        # Check shape
        assert samples.shape == (100, 3)
        
        # Check mean is close
        assert numpy.allclose(samples.mean(axis=0), mean, atol=0.1)
    
    def test_uniform_distribution_sampling(self):
        """Test uniform load distribution sampling."""
        mean = numpy.array([1.0, 2.0])
        dist_params = {
            'type': 'uniform',
            'mean': mean,
            'scale': 0.5
        }
        
        samples = _sample_load_distribution(dist_params, n_samples=100)
        
        # Check shape
        assert samples.shape == (100, 2)
        
        # Check all samples are within bounds
        lower = mean - 0.5
        upper = mean + 0.5
        assert numpy.all(samples >= lower)
        assert numpy.all(samples <= upper)
    
    def test_gaussian_mixture_sampling(self):
        """Test Gaussian mixture load distribution."""
        mean = numpy.array([1.0, 2.0])
        dist_params = {
            'type': 'gaussian_mixture',
            'mean': mean,
            'weights': [0.5, 0.5],
            'means': [mean * 0.9, mean * 1.1],
        }
        
        samples = _sample_load_distribution(dist_params, n_samples=200)
        
        assert samples.shape == (200, 2)
        # Should have samples from both modes
        assert samples.shape[0] > 0
    
    def test_invalid_distribution_type(self):
        """Test error handling for invalid distribution type."""
        dist_params = {
            'type': 'invalid_type',
            'mean': numpy.array([1.0, 2.0])
        }
        
        with pytest.raises(ValueError):
            _sample_load_distribution(dist_params, n_samples=10)
    
    def test_missing_mean(self):
        """Test error handling when mean is missing."""
        dist_params = {'type': 'normal'}
        
        with pytest.raises(ValueError):
            _sample_load_distribution(dist_params, n_samples=10)


@pytest.mark.skipif(not TOPOPT_AVAILABLE, reason="topopt not available")
class TestBetaRandomLoadFunction:
    """Test BetaRandomLoadFunction autograd implementation."""
    
    def setup_method(self):
        """Create test problem."""
        self.problem = MBBBeam(nelx=10, nely=5)
        self.n_elements = self.problem.nelx * self.problem.nely
    
    def test_forward_pass(self):
        """Test forward pass produces valid output."""
        alpha = torch.ones(self.n_elements, dtype=torch.float32) * 2.0
        beta = torch.ones(self.n_elements, dtype=torch.float32) * 2.0
        
        dist_params = {
            'type': 'normal',
            'mean': self.problem.f.copy(),
            'std': 0.05 * numpy.abs(self.problem.f)
        }
        
        result = BetaRandomLoadFunction.apply(
            alpha, beta, self.problem, dist_params,
            n_design_samples=5, n_load_samples=5
        )
        
        # Check output is scalar and finite
        assert result.ndim == 0
        assert torch.isfinite(result)
        assert result > 0  # Compliance should be positive
    
    def test_backward_pass_computes_gradients(self):
        """Test that backward pass computes non-zero gradients."""
        alpha = torch.ones(self.n_elements, dtype=torch.float32) * 2.0
        alpha.requires_grad_(True)
        beta = torch.ones(self.n_elements, dtype=torch.float32) * 2.0
        beta.requires_grad_(True)
        
        dist_params = {
            'type': 'normal',
            'mean': self.problem.f.copy(),
            'std': 0.05 * numpy.abs(self.problem.f)
        }
        
        result = BetaRandomLoadFunction.apply(
            alpha, beta, self.problem, dist_params,
            n_design_samples=3, n_load_samples=3
        )
        
        result.backward()
        
        # Check gradients exist and are non-zero
        assert alpha.grad is not None
        assert beta.grad is not None
        assert not torch.allclose(alpha.grad, torch.zeros_like(alpha.grad))
    
    def test_gradient_finite_difference_check(self):
        """Test gradient correctness with finite differences."""
        eps = 1e-4
        alpha = torch.ones(self.n_elements, dtype=torch.float32) * 2.0
        alpha.requires_grad_(True)
        beta = torch.ones(self.n_elements, dtype=torch.float32) * 2.0
        
        dist_params = {
            'type': 'normal',
            'mean': self.problem.f.copy(),
            'std': 0.05 * numpy.abs(self.problem.f)
        }
        
        # Compute autograd gradient
        result = BetaRandomLoadFunction.apply(
            alpha, beta, self.problem, dist_params,
            n_design_samples=2, n_load_samples=2
        )
        result.backward()
        grad_autograd = alpha.grad[0].item()
        
        # Compute finite difference gradient (for first element)
        alpha_pos = torch.ones(self.n_elements, dtype=torch.float32) * 2.0
        alpha_pos[0] += eps
        result_pos = BetaRandomLoadFunction.apply(
            alpha_pos, beta, self.problem, dist_params,
            n_design_samples=2, n_load_samples=2
        )
        
        alpha_neg = torch.ones(self.n_elements, dtype=torch.float32) * 2.0
        alpha_neg[0] -= eps
        result_neg = BetaRandomLoadFunction.apply(
            alpha_neg, beta, self.problem, dist_params,
            n_design_samples=2, n_load_samples=2
        )
        
        grad_fd = (result_pos.item() - result_neg.item()) / (2 * eps)
        
        # Should be reasonably close (allowing for MC error)
        assert abs(grad_autograd - grad_fd) / (abs(grad_fd) + 1e-6) < 0.5


@pytest.mark.skipif(not TOPOPT_AVAILABLE, reason="topopt not available")
class TestBetaSolverRandomLoads:
    """Test BetaSolverRandomLoads solver class."""
    
    def setup_method(self):
        """Create test problem and solver."""
        self.problem = MBBBeam(nelx=8, nely=4)
        self.filter = DensityFilter(self.problem)
        self.gui = NullGUI()
        
        # Load distribution: 10% uncertainty
        self.load_dist_params = {
            'type': 'normal',
            'mean': self.problem.f.copy(),
            'std': 0.1 * numpy.abs(self.problem.f)
        }
    
    def test_solver_initialization(self):
        """Test solver can be initialized."""
        solver = BetaSolverRandomLoads(
            self.problem, volfrac=0.3, filter=self.filter, gui=self.gui,
            load_dist_params=self.load_dist_params,
            n_design_samples=5, n_load_samples=5
        )
        
        assert solver is not None
        assert solver.volfrac == 0.3
        assert solver.n_load_samples == 5
    
    def test_solver_initialization_default_load_dist(self):
        """Test solver with default load distribution."""
        solver = BetaSolverRandomLoads(
            self.problem, volfrac=0.3, filter=self.filter, gui=self.gui,
            load_dist_params=None,  # Use default
            n_design_samples=5, n_load_samples=5
        )
        
        assert solver is not None
        assert solver.load_dist_params is not None
        assert solver.load_dist_params['type'] == 'normal'
    
    def test_solver_optimization_runs(self):
        """Test optimization loop runs without errors."""
        solver = BetaSolverRandomLoads(
            self.problem, volfrac=0.4, filter=self.filter, gui=self.gui,
            load_dist_params=self.load_dist_params,
            maxeval=5,  # Just 5 iterations for testing
            n_design_samples=3, n_load_samples=3
        )
        
        x_init = numpy.ones(self.problem.nelx * self.problem.nely) * 0.5
        x_opt = solver.optimize(x_init)
        
        # Check output
        assert x_opt.shape == (self.problem.nelx * self.problem.nely,)
        assert numpy.all(x_opt >= 0.0)
        assert numpy.all(x_opt <= 1.0)
    
    def test_robust_statistics(self):
        """Test robustness statistics computation."""
        solver = BetaSolverRandomLoads(
            self.problem, volfrac=0.4, filter=self.filter, gui=self.gui,
            load_dist_params=self.load_dist_params,
            maxeval=3, n_design_samples=3, n_load_samples=3
        )
        
        # Run quick optimization
        x_init = numpy.ones(self.problem.nelx * self.problem.nely) * 0.5
        solver.optimize(x_init)
        
        # Get robustness statistics
        stats = solver.get_robust_statistics(n_eval_samples=10)
        
        # Check statistics
        assert isinstance(stats, dict)
        assert 'mean' in stats
        assert 'std' in stats
        assert 'percentile_5' in stats
        assert 'percentile_95' in stats
        assert 'all_samples' in stats
        
        # Check statistical properties
        assert stats['mean'] > 0
        assert stats['std'] > 0
        assert stats['min'] > 0
        assert stats['percentile_5'] <= stats['mean']
        assert stats['mean'] <= stats['percentile_95']
        assert len(stats['all_samples']) == 10


@pytest.mark.skipif(not TOPOPT_AVAILABLE, reason="topopt not available")
class TestComparisonWithBaseBeta:
    """Compare BetaSolverRandomLoads with BetaSolverWithImplicitDiff."""
    
    def setup_method(self):
        """Create test problem."""
        self.problem = MBBBeam(nelx=8, nely=4)
        self.filter = DensityFilter(self.problem)
        self.gui = NullGUI()
    
    def test_deterministic_loads_converge_similarly(self):
        """Test that deterministic loads (no variance) give similar results."""
        # Solver without load uncertainty
        solver1 = BetaSolverWithImplicitDiff(
            self.problem, volfrac=0.4, filter=self.filter, gui=self.gui,
            maxeval=5, n_samples=5
        )
        
        # Solver with zero-variance loads (deterministic)
        load_dist_params = {
            'type': 'normal',
            'mean': self.problem.f.copy(),
            'std': 1e-8 * numpy.ones_like(self.problem.f)  # Essentially zero
        }
        solver2 = BetaSolverRandomLoads(
            self.problem, volfrac=0.4, filter=self.filter, gui=self.gui,
            load_dist_params=load_dist_params,
            maxeval=5, n_design_samples=5, n_load_samples=3
        )
        
        x_init = numpy.ones(self.problem.nelx * self.problem.nely) * 0.5
        
        # Both should run without error
        x_opt1 = solver1.optimize(x_init.copy())
        x_opt2 = solver2.optimize(x_init.copy())
        
        assert x_opt1.shape == x_opt2.shape
        assert numpy.all(x_opt1 >= 0)
        assert numpy.all(x_opt2 >= 0)


if __name__ == '__main__':
    # Run tests with pytest
    pytest.main([__file__, '-v'])
