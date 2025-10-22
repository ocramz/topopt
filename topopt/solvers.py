"""
Solvers to solve topology optimization problems.

Todo:
    * Make TopOptSolver an abstract class
    * Rename the current TopOptSolver to MMASolver(TopOptSolver)
    * Create a TopOptSolver using originality criterion
"""
from __future__ import division

import numpy
import torch
import torch.autograd
import torch.nn as nn
import torch.nn.functional as F

from topopt.problems import Problem
from topopt.filters import Filter
from topopt.guis import GUI


class ComplianceFunction(torch.autograd.Function):
    """
    Custom autograd function for compliance computation.
    
    Wraps the FEM analysis so PyTorch can compute gradients through it.
    The forward pass solves the FE problem, backward pass uses adjoint method.
    """
    
    @staticmethod
    def forward(ctx, x_phys, problem):
        """
        Forward pass: compute compliance using FEM.
        
        Parameters
        ----------
        x_phys : torch.Tensor
            Design variables (densities)
        problem : Problem
            The topology optimization problem instance
            
        Returns
        -------
        torch.Tensor
            Scalar compliance value
        """
        x_np = x_phys.detach().cpu().numpy()
        dobj = numpy.zeros_like(x_np)
        
        # Compute objective and sensitivity using problem's FEM solver
        # This modifies dobj IN-PLACE with the sensitivities
        obj = problem.compute_objective(x_np, dobj)
        
        # IMPORTANT: dobj is now updated with the actual gradients!
        # Make a copy to ensure we capture the modified values
        dobj_copy = dobj.copy()
        
        # Save for backward pass
        ctx.save_for_backward(x_phys)
        ctx.problem = problem
        ctx.dobj = torch.from_numpy(dobj_copy).float()
        
        return torch.tensor(obj, dtype=x_phys.dtype, device=x_phys.device)
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: return gradient using stored sensitivities.
        
        Since we already computed dobj in forward (adjoint method),
        we just scale by grad_output.
        """
        x_phys, = ctx.saved_tensors
        dobj = ctx.dobj.to(x_phys.device)
        
        # Gradient w.r.t. x_phys
        grad_x = dobj * grad_output
        
        return grad_x, None  # None for problem argument


class VolumeConstraint(torch.autograd.Function):
    """
    Custom autograd function for volume constraint.
    
    Computes g(x) = sum(x)/n_elements - volfrac such that g <= 0.
    """
    
    @staticmethod
    def forward(ctx, x_phys, volfrac):
        """Compute volume constraint: g = sum(x)/n_elements - volfrac."""
        n_elements = x_phys.numel()
        constraint_value = x_phys.sum() / n_elements - volfrac
        
        ctx.save_for_backward(x_phys)
        ctx.n_elements = n_elements
        
        return constraint_value
    
    @staticmethod
    def backward(ctx, grad_output):
        """Gradient of constraint is uniform: dg/dx = 1/n_elements."""
        x_phys, = ctx.saved_tensors
        n_elements = ctx.n_elements
        
        grad_x = torch.ones_like(x_phys) / n_elements * grad_output
        
        return grad_x, None  # None for volfrac argument


class BetaParameterFunction(torch.autograd.Function):
    """
    Implicit differentiation through Beta parameter optimization.
    
    For each element, we have a Beta(α_e, β_e) distribution over [0,1].
    The optimal density ρ_e is sampled from this distribution.
    
    Gradients w.r.t. α, β are computed via implicit function theorem:
    dC/dα = (∂C/∂ρ) · (∂E[ρ]/∂α) where E[ρ] = α/(α+β)
    """
    
    @staticmethod
    def forward(ctx, alpha, beta, problem, n_samples=100):
        """
        Forward: compute expected compliance over Beta samples.
        
        Parameters
        ----------
        alpha : torch.Tensor
            Shape (n_elements,), must be > 1
        beta : torch.Tensor
            Shape (n_elements,), must be > 1
        problem : Problem
            FEM problem instance
        n_samples : int
            Number of samples for expectation
            
        Returns
        -------
        torch.Tensor
            Expected compliance value
        """
        alpha_np = alpha.detach().cpu().numpy()
        beta_np = beta.detach().cpu().numpy()
        
        # Sample from Beta distributions: ρ_e ~ Beta(α_e, β_e)
        rho_samples = numpy.random.beta(alpha_np, beta_np, 
                                       size=(n_samples, len(alpha_np)))
        
        # Compute compliance for each sample and average sensitivities
        compliances = []
        dobj_avg = numpy.zeros_like(alpha_np)
        
        for sample in rho_samples:
            dobj_sample = numpy.zeros_like(sample)
            c = problem.compute_objective(sample, dobj_sample)
            compliances.append(c)
            dobj_avg += dobj_sample
        
        # Expected compliance: E[C(ρ)]
        obj = numpy.mean(compliances)
        dobj_avg /= n_samples
        
        # Save for implicit differentiation
        ctx.save_for_backward(alpha, beta)
        ctx.dobj_avg = torch.from_numpy(dobj_avg).float()
        ctx.n_samples = n_samples
        
        return torch.tensor(obj, dtype=alpha.dtype)
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Implicit differentiation using gradient of Beta moments.
        
        By implicit function theorem:
        - E[ρ] = α/(α+β)   (mean of Beta)
        - dE[ρ]/dα = β/(α+β)²
        - dE[ρ]/dβ = -α/(α+β)²
        
        dC/dα = (∂C/∂ρ) · (∂E[ρ]/∂α)
        """
        alpha, beta = ctx.saved_tensors
        dobj = ctx.dobj_avg
        
        # Moments of Beta distribution
        ab_sum = alpha + beta
        
        # Gradient of mean E[ρ] w.r.t. parameters
        # dE[ρ]/dα = β/(α+β)²
        d_mean_d_alpha = beta / (ab_sum ** 2)
        
        # dE[ρ]/dβ = -α/(α+β)²
        d_mean_d_beta = -alpha / (ab_sum ** 2)
        
        # Chain rule through expected compliance
        grad_alpha = dobj * d_mean_d_alpha * grad_output
        grad_beta = dobj * d_mean_d_beta * grad_output
        
        return grad_alpha, grad_beta, None, None


<<<<<<< HEAD
def _sample_load_distribution(dist_params, n_samples):
    """
    Sample from specified load distribution.
    
    Parameters
    ----------
    dist_params : dict
        Distribution specification with keys:
        - 'type': 'normal', 'uniform', or 'gaussian_mixture'
        - 'mean': base load vector
        - Additional parameters depend on type
        
    n_samples : int
        Number of samples to draw
        
    Returns
    -------
    numpy.ndarray
        Shape (n_samples, n_dof) - load vectors for each sample
    """
    dist_type = dist_params.get('type', 'normal')
    mean = dist_params.get('mean')
    
    if mean is None:
        raise ValueError("Load distribution must specify 'mean'")
    
    mean_array = numpy.asarray(mean).flatten()
    n_dof = len(mean_array)
    
    if dist_type == 'normal':
        cov = dist_params.get('cov', None)
        if cov is None:
            # Default: 10% standard deviation
            std = dist_params.get('std', 0.1 * numpy.abs(mean_array))
            cov = numpy.diag(std ** 2)
        
        samples = numpy.random.multivariate_normal(mean_array, cov, n_samples)
        
    elif dist_type == 'uniform':
        scale = dist_params.get('scale', 0.1 * numpy.abs(mean_array))
        scale_array = numpy.asarray(scale).flatten()
        samples = mean_array[None, :] + numpy.random.uniform(
            -scale_array[None, :], scale_array[None, :], size=(n_samples, n_dof)
        )
    
    elif dist_type == 'gaussian_mixture':
        weights = dist_params.get('weights', [0.5, 0.5])
        means_list = dist_params.get('means', [mean_array * 0.9, mean_array * 1.1])
        covs = dist_params.get('covs', 
                               [numpy.eye(n_dof) * (0.05 * numpy.abs(mean_array)) ** 2 
                                for _ in means_list])
        
        samples = []
        for _ in range(n_samples):
            idx = numpy.random.choice(len(means_list), p=weights)
            sample = numpy.random.multivariate_normal(means_list[idx], covs[idx])
            samples.append(sample)
        samples = numpy.array(samples)
    
    else:
        raise ValueError(f"Unknown distribution type: {dist_type}")
    
    return samples


class BetaRandomLoadFunction(torch.autograd.Function):
    """
    Implicit differentiation through Beta parameters under random loads.
    
    Optimizes: min E_ρ,f[C(ρ, f)] 
    where: ρ ~ Beta(α, β), f ~ LoadDistribution
    
    This handles BOTH design uncertainty AND load uncertainty jointly.
    """
    
    @staticmethod
    def forward(ctx, alpha, beta, problem, load_dist_params,
                n_design_samples=50, n_load_samples=20):
        """
        Forward: compute expected compliance over design AND load uncertainty.
        
        Parameters
        ----------
        alpha : torch.Tensor
            Shape (n_elements,), design Beta parameter
        beta : torch.Tensor
            Shape (n_elements,), design Beta parameter
        problem : Problem
            FEM problem with nominal load
        load_dist_params : dict
            Load distribution specification
        n_design_samples : int
            Monte Carlo samples over designs
        n_load_samples : int
            Monte Carlo samples over loads
            
        Returns
        -------
        torch.Tensor
            Expected compliance E_ρ,f[C(ρ,f)]
        """
        alpha_np = alpha.detach().cpu().numpy()
        beta_np = beta.detach().cpu().numpy()
        
        # Sample designs: ρ ~ Beta(α, β)
        rho_samples = numpy.random.beta(alpha_np, beta_np,
                                        size=(n_design_samples, len(alpha_np)))
        
        # Sample loads from specified distribution
        load_samples = _sample_load_distribution(load_dist_params, n_load_samples)
        
        # Store nominal load
        nominal_load = problem.f.copy() if hasattr(problem, 'f') else None
        
        # Compute compliance for each (ρ, f) pair and average sensitivities
        compliances = []
        sensitivities_avg = numpy.zeros_like(alpha_np)
        
        for rho in rho_samples:
            for f in load_samples:
                # Temporarily set load
                if hasattr(problem, 'f'):
                    problem.f = f
                
                dobj_sample = numpy.zeros_like(rho)
                c = problem.compute_objective(rho, dobj_sample)
                compliances.append(c)
                sensitivities_avg += dobj_sample
        
        # Restore nominal load
        if nominal_load is not None and hasattr(problem, 'f'):
            problem.f = nominal_load
        
        # Average over all (design, load) pairs
        obj = numpy.mean(compliances)
        sensitivities_avg /= (n_design_samples * n_load_samples)
        
        # Save for backward pass
        ctx.save_for_backward(alpha, beta)
        ctx.sensitivities_avg = torch.from_numpy(sensitivities_avg).float()
        ctx.n_design_samples = n_design_samples
        ctx.n_load_samples = n_load_samples
        
        return torch.tensor(obj, dtype=alpha.dtype)
    
    @staticmethod
    def backward(ctx, grad_output):
        """Implicit differentiation through nested expectation."""
        alpha, beta = ctx.saved_tensors
        sensitivities = ctx.sensitivities_avg
        
        # Beta moment derivatives (same as BetaParameterFunction)
        ab_sum = alpha + beta
        d_mean_d_alpha = beta / (ab_sum ** 2)
        d_mean_d_beta = -alpha / (ab_sum ** 2)
        
        grad_alpha = sensitivities * d_mean_d_alpha * grad_output
        grad_beta = sensitivities * d_mean_d_beta * grad_output
        
        return grad_alpha, grad_beta, None, None, None, None


=======
class TopOptSolver:
>>>>>>> 0983bf624982fb46d2010aa1a9dd36d7b998483f
    """Solver using mirror descent on the simplex with PyTorch autograd."""

    def __init__(self, problem: Problem, volfrac: float, filter: Filter,
                 gui: GUI, maxeval=2000, ftol_rel=1e-3, learning_rate=0.05):
        """
        Create a solver using PyTorch and mirror descent.

        Parameters
        ----------
        problem: :obj:`topopt.problems.Problem`
            The topology optimization problem to solve.
        volfrac: float
            The maximum fraction of the volume to use.
        filter: :obj:`topopt.filters.Filter`
            A filter for the solutions to reduce artefacts.
        gui: :obj:`topopt.guis.GUI`
            The graphical user interface to visualize intermediate results.
        maxeval: int
            The maximum number of evaluations to perform.
        ftol_rel: float
            A floating point tolerance for relative change.
        learning_rate: float
            Step size for mirror descent updates.
        """
        self.problem = problem
        self.filter = filter
        self.gui = gui
        self.volfrac = volfrac
        
        n = problem.nelx * problem.nely
        self.xPhys = numpy.ones(n)
        self._maxeval = maxeval
        self._ftol_rel = ftol_rel
        self.learning_rate = learning_rate
        
        # Dual variable (Lagrange multiplier for volume constraint)
        self.lambda_vol = 0.0
        self.dual_step_size = 0.01
        
        # Setup filter
        self.passive = problem.bc.passive_elements
        if self.passive.size > 0:
            self.xPhys[self.passive] = 0
        self.active = problem.bc.active_elements
        if self.active.size > 0:
            self.xPhys[self.active] = 1

    def __str__(self):
        """Create a string representation of the solver."""
        return self.__class__.__name__

    def __format__(self, format_spec):
        """Create a formated representation of the solver."""
        return "{} with {}".format(str(self.problem), str(self))

    def __repr__(self):
        """Create a representation of the solver."""
        return ("{}(problem={!r}, volfrac={:g}, filter={!r}, ".format(
            self.__class__.__name__, self.problem, self.volfrac, self.filter)
            + "gui={!r}, maxeval={:d}, ftol={:g}, learning_rate={:g})".format(
                self.gui, self.maxeval, self.ftol_rel, self.learning_rate))

    @property
    def ftol_rel(self):
        """:obj:`float`: Relative tolerance for convergence."""
        return self._ftol_rel

    @ftol_rel.setter
    def ftol_rel(self, ftol_rel):
        self._ftol_rel = ftol_rel

    @property
    def maxeval(self):
        """:obj:`int`: Maximum number of objective evaluations (iterations)."""
        return self._maxeval

    @maxeval.setter
    def maxeval(self, maxeval):
        self._maxeval = maxeval

    def optimize(self, x: numpy.ndarray) -> numpy.ndarray:
        """
        Optimize the problem using mirror descent with augmented Lagrangian.

        Parameters
        ----------
        x:
            The initial value for the design variables.

        Returns
        -------
        numpy.ndarray
            The optimal value of x found.
        """
        self.xPhys = x.copy()
        
        # Convert to torch tensor
        x_torch = torch.from_numpy(x.copy()).float()
        x_torch.requires_grad_(True)
        
        prev_obj = float('inf')
        penalty_param = 1.0
        
        for iteration in range(self.maxeval):
            # Filter design variables
            self.filter_variables(x_torch.detach().cpu().numpy())
            
            # Compute objective using custom autograd function
            obj_value = ComplianceFunction.apply(x_torch, self.problem)
            
            # Compute volume constraint
            vol_constraint = VolumeConstraint.apply(x_torch, self.volfrac)
            
            # Augmented Lagrangian: L = obj + lambda*g + (penalty/2)*g^2
            lagrangian = obj_value + self.lambda_vol * vol_constraint + \
                         (penalty_param / 2.0) * vol_constraint ** 2
            
            # Backward pass
            if x_torch.grad is not None:
                x_torch.grad.zero_()
            lagrangian.backward()
            
            grad = x_torch.grad.clone().detach()
            
            # Mirror descent step in log-space (natural gradient on simplex)
            log_x = torch.log(torch.clamp(x_torch.detach(), min=1e-10))
            log_x = log_x - self.learning_rate * grad
            
            # Project back to simplex via softmax
            x_torch = self._softmax(log_x)
            x_torch.requires_grad_(True)
            
            # Update Lagrange multiplier (dual ascent)
            vol_val = vol_constraint.item()
            self.lambda_vol += self.dual_step_size * vol_val
            
            # Increase penalty if constraint not satisfied
            if abs(vol_val) > 1e-4:
                penalty_param *= 1.1
            
            # Update GUI
            self.gui.update(self.xPhys)
            
            # Check convergence
            if abs(prev_obj - obj_value.item()) / (abs(prev_obj) + 1e-10) < self.ftol_rel:
                if iteration > 100:  # Require minimum iterations
                    break
            
            prev_obj = obj_value.item()
            
            if iteration % 100 == 0:
                print(f"Iteration {iteration}: obj={obj_value.item():.6f}, "
                      f"vol_constraint={vol_val:.6f}, lambda={self.lambda_vol:.6f}")
        
        return x_torch.detach().cpu().numpy()
    
    def _softmax(self, log_x: torch.Tensor) -> torch.Tensor:
        """
        Softmax projection to project onto probability simplex.
        
        Parameters
        ----------
        log_x : torch.Tensor
            Log-space representation
            
        Returns
        -------
        torch.Tensor
            Projected values on [0, 1]
        """
        log_x = log_x - torch.max(log_x)  # Numerical stability
        exp_x = torch.exp(log_x)
        return exp_x / torch.sum(exp_x)

    def filter_variables(self, x: numpy.ndarray) -> numpy.ndarray:
        """
        Filter the variables and impose values on passive/active variables.

        Parameters
        ----------
        x:
            The variables to be filtered.

        Returns
        -------
        numpy.ndarray
            The filtered "physical" variables.
        """
        self.filter.filter_variables(x, self.xPhys)
        if self.passive.size > 0:
            self.xPhys[self.passive] = 0
        if self.active.size > 0:
            self.xPhys[self.active] = 1
        return self.xPhys


class BetaSolverWithImplicitDiff(TopOptSolver):
    """
    Topology optimization using Beta-distributed design variables.
    
    Each element density ρ_e ~ Beta(α_e, β_e) where α_e, β_e are optimized
    via implicit differentiation through the FEM analysis.
    
    This approach provides:
    - Exact FEM gradients via implicit differentiation
    - Uncertainty quantification through Beta parameters
    - Principled probabilistic design exploration
    """
    
    def __init__(self, problem: Problem, volfrac: float, filter: Filter,
                 gui: GUI, maxeval=2000, ftol_rel=1e-3, learning_rate=0.01,
                 n_samples=50):
        """
        Create solver with Beta-distributed design variables.
        
        Parameters
        ----------
        problem : Problem
            FEM problem
        volfrac : float
            Volume constraint
        filter : Filter
            Density filter
        gui : GUI
            Visualization
        maxeval : int
            Max iterations
        ftol_rel : float
            Convergence tolerance
        learning_rate : float
            Adam optimizer learning rate
        n_samples : int
            Samples per iteration for Monte Carlo expectation
        """
        super().__init__(problem, volfrac, filter, gui, maxeval, ftol_rel, 
                        learning_rate)
        self.n_samples = n_samples
        
        # Initialize Beta parameters
        n_elements = problem.nelx * problem.nely
        
        # Start with Beta(2, 2) - roughly uniform on [0,1]
        # Use softplus transformation to enforce α, β > 1
        self.alpha_logit = nn.Parameter(torch.ones(n_elements) * 0.0)
        self.beta_logit = nn.Parameter(torch.ones(n_elements) * 0.0)
        
        # Optimizer for Beta parameters
        self.optimizer = torch.optim.Adam(
            [self.alpha_logit, self.beta_logit], 
            lr=learning_rate
        )
        
        # Constraint on volume
        self.lambda_vol = 0.0
    
    def _get_alpha_beta(self):
        """
        Get α and β from logit parameters, ensuring α, β > 1.
        
        Uses softplus + 1 to ensure positivity and minimum value of 1.
        """
        alpha = F.softplus(self.alpha_logit) + 1.0
        beta = F.softplus(self.beta_logit) + 1.0
        return alpha, beta
    
    def optimize(self, x: numpy.ndarray) -> numpy.ndarray:
        """
        Optimize Beta parameters using implicit differentiation.
        
        Parameters
        ----------
        x : numpy.ndarray
            Initial design (not used, overridden by Beta initialization)
            
        Returns
        -------
        numpy.ndarray
            Optimal design densities (mean of learned Beta)
        """
        prev_obj = float('inf')
        
        for iteration in range(self.maxeval):
            self.optimizer.zero_grad()
            
            # Get α, β ensuring they are > 1
            alpha, beta = self._get_alpha_beta()
            
            # Compute compliance with implicit differentiation
            obj = BetaParameterFunction.apply(alpha, beta, 
                                             self.problem, self.n_samples)
            
            # Volume constraint in expectation
            # E[ρ] = α/(α+β)
            mean_rho = alpha / (alpha + beta)
            vol_constraint = mean_rho.mean() - self.volfrac
            
            # Augmented Lagrangian
            penalty = 100.0
            lagrangian = obj + self.lambda_vol * vol_constraint + \
                        (penalty / 2.0) * vol_constraint ** 2
            
            # Backward pass
            lagrangian.backward()
            self.optimizer.step()
            
            # Dual update for Lagrange multiplier
            self.lambda_vol += 0.01 * vol_constraint.item()
            
            # Print progress
            if iteration % 50 == 0:
                alpha_val, beta_val = self._get_alpha_beta()
                print(f"Iter {iteration}: obj={obj.item():.6f}, "
                      f"vol={vol_constraint.item():.6f}, "
                      f"α_mean={alpha_val.mean().item():.3f}, "
                      f"β_mean={beta_val.mean().item():.3f}")
            
            # Convergence check
            if abs(vol_constraint.item()) < 1e-4 and iteration > 100:
                if abs(prev_obj - obj.item()) / (abs(prev_obj) + 1e-10) < self.ftol_rel:
                    break
            
            prev_obj = obj.item()
        
        # Extract final design (use mean of Beta as point estimate)
        alpha_final, beta_final = self._get_alpha_beta()
        final_rho = (alpha_final / (alpha_final + beta_final)).detach().cpu().numpy()
        
        return final_rho
    
    def get_confidence_intervals(self, percentile=95):
        """
        Return credible intervals for each element's density.
        
        Since each ρ_e ~ Beta(α_e, β_e), compute quantile-based intervals.
        
        Parameters
        ----------
        percentile : float
            Confidence level (default 95%)
            
        Returns
        -------
        tuple of numpy.ndarray
            (lower, upper) bounds for each element density
        """
        try:
            from scipy import stats
        except ImportError:
            raise ImportError("scipy required for confidence intervals")
        
        alpha_final, beta_final = self._get_alpha_beta()
        
        alpha_np = alpha_final.detach().cpu().numpy()
        beta_np = beta_final.detach().cpu().numpy()
        
        p_lower = (100 - percentile) / 2 / 100
        p_upper = 1 - p_lower
        
        lower = numpy.array([stats.beta.ppf(p_lower, a, b) 
                            for a, b in zip(alpha_np, beta_np)])
        upper = numpy.array([stats.beta.ppf(p_upper, a, b) 
                            for a, b in zip(alpha_np, beta_np)])
        
        return lower, upper
    
    def get_design_variance(self):
        """
        Compute variance of densities for each element.
        
        For Beta(α, β):
        Var[ρ] = (α·β) / ((α+β)² · (α+β+1))
        
        Returns
        -------
        numpy.ndarray
            Variance for each element
        """
        alpha, beta = self._get_alpha_beta()
        
        ab_sum = alpha + beta
        variance = (alpha * beta) / ((ab_sum ** 2) * (ab_sum + 1))
        
        return variance.detach().cpu().numpy()


class BetaSolverRandomLoads(BetaSolverWithImplicitDiff):
    """
    Topology optimization under BOTH design AND load uncertainty.
    
    Minimizes: E_ρ,f[C(ρ, f)]
    subject to: E_ρ[∑ρ_e] ≤ V_frac
    
    where: ρ_e ~ Beta(α_e, β_e), f ~ specified distribution
    
    This handles joint optimization of design and robustness to load variability.
    """
    
    def __init__(self, problem: Problem, volfrac: float, filter: Filter,
                 gui: GUI, load_dist_params=None, maxeval=2000, 
                 ftol_rel=1e-3, learning_rate=0.01, 
                 n_design_samples=50, n_load_samples=20):
        """
        Initialize solver for random loads.
        
        Parameters
        ----------
        problem : Problem
            FEM problem with nominal load
        volfrac : float
            Volume constraint
        filter : Filter
            Density filter
        gui : GUI
            Visualization
        load_dist_params : dict, optional
            Load distribution specification:
            {
                'type': 'normal',  # or 'uniform', 'gaussian_mixture'
                'mean': problem.f,  # base load
                'cov': covariance_matrix,  # or 'scale'/'std' depending on type
                ...
            }
            If None, defaults to normal distribution with 10% std dev
        maxeval : int
            Maximum iterations
        ftol_rel : float
            Convergence tolerance
        learning_rate : float
            Adam learning rate
        n_design_samples : int
            Monte Carlo samples for design variables
        n_load_samples : int
            Monte Carlo samples for loads
        """
        super().__init__(problem, volfrac, filter, gui, maxeval, ftol_rel,
                        learning_rate, n_samples=n_design_samples)
        
        # Load distribution
        if load_dist_params is None:
            # Default: normal distribution with 10% std dev
            load_dist_params = {
                'type': 'normal',
                'mean': problem.f.copy() if hasattr(problem, 'f') else numpy.ones(problem.ndof),
                'std': 0.1 * numpy.ones(problem.f.shape if hasattr(problem, 'f') else (problem.ndof,))
            }
        
        self.load_dist_params = load_dist_params
        self.n_load_samples = n_load_samples
        self.nominal_load = problem.f.copy() if hasattr(problem, 'f') else None
    
    def optimize(self, x: numpy.ndarray) -> numpy.ndarray:
        """
        Optimize with nested Monte Carlo over designs and loads.
        
        Parameters
        ----------
        x : numpy.ndarray
            Initial design (not used, overridden by Beta initialization)
            
        Returns
        -------
        numpy.ndarray
            Optimal design densities robust to load uncertainty
        """
        prev_obj = float('inf')
        
        for iteration in range(self.maxeval):
            self.optimizer.zero_grad()
            
            # Get α, β ensuring they are > 1
            alpha, beta = self._get_alpha_beta()
            
            # Compute compliance with nested Monte Carlo + implicit differentiation
            obj = BetaRandomLoadFunction.apply(
                alpha, beta, self.problem, self.load_dist_params,
                self.n_samples, self.n_load_samples
            )
            
            # Volume constraint (expectation over designs only)
            mean_rho = alpha / (alpha + beta)
            vol_constraint = mean_rho.mean() - self.volfrac
            
            # Augmented Lagrangian
            penalty = 100.0
            lagrangian = obj + self.lambda_vol * vol_constraint + \
                        (penalty / 2.0) * vol_constraint ** 2
            
            # Backward pass
            lagrangian.backward()
            self.optimizer.step()
            
            # Dual update for Lagrange multiplier
            self.lambda_vol += 0.01 * vol_constraint.item()
            
            # Print progress
            if iteration % 50 == 0:
                alpha_val, beta_val = self._get_alpha_beta()
                print(f"Iter {iteration}: E[C(ρ,f)]={obj.item():.6f}, "
                      f"vol={vol_constraint.item():.6f}, "
                      f"α∈[{alpha_val.min().item():.2f}, {alpha_val.max().item():.2f}]")
            
            # Convergence check
            if abs(vol_constraint.item()) < 1e-4 and iteration > 100:
                if abs(prev_obj - obj.item()) / (abs(prev_obj) + 1e-10) < self.ftol_rel:
                    break
            
            prev_obj = obj.item()
        
        # Restore nominal load
        if self.nominal_load is not None and hasattr(self.problem, 'f'):
            self.problem.f = self.nominal_load
        
        # Extract final design
        alpha_final, beta_final = self._get_alpha_beta()
        final_rho = (alpha_final / (alpha_final + beta_final)).detach().cpu().numpy()
        
        return final_rho
    
    def get_robust_statistics(self, n_eval_samples=1000):
        """
        Evaluate design robustness: compute distribution of compliance under random loads.
        
        Parameters
        ----------
        n_eval_samples : int
            Number of load samples to evaluate
            
        Returns
        -------
        dict with robustness statistics:
            - 'mean': average compliance
            - 'std': standard deviation
            - 'min', 'max': range
            - 'percentile_5', 'percentile_95': confidence bounds
            - 'all_samples': array of all compliance values
        """
        # Get optimal design
        alpha_final, beta_final = self._get_alpha_beta()
        rho_opt = (alpha_final / (alpha_final + beta_final)).detach().cpu().numpy()
        
        # Sample loads and evaluate
        load_samples = _sample_load_distribution(self.load_dist_params, n_eval_samples)
        
        # Store nominal load
        nominal_load = self.problem.f.copy() if hasattr(self.problem, 'f') else None
        
        compliances = []
        try:
            for f in load_samples:
                if hasattr(self.problem, 'f'):
                    self.problem.f = f
                
                dobj_dummy = numpy.zeros_like(rho_opt)
                c = self.problem.compute_objective(rho_opt, dobj_dummy)
                compliances.append(c)
        finally:
            # Restore nominal load
            if nominal_load is not None and hasattr(self.problem, 'f'):
                self.problem.f = nominal_load
        
        compliances = numpy.array(compliances)
        
        return {
            'mean': compliances.mean(),
            'std': compliances.std(),
            'min': compliances.min(),
            'max': compliances.max(),
            'percentile_5': numpy.percentile(compliances, 5),
            'percentile_95': numpy.percentile(compliances, 95),
            'all_samples': compliances
        }


# TODO: Seperate optimizer from TopOptSolver
# class MMASolver(TopOptSolver):
#     pass
#
#
# TODO: Port over OC to TopOptSolver
# class OCSolver(TopOptSolver):
#     def oc(self, x, volfrac, dc, dv, g):
#         """ Optimality criterion """
#         l1 = 0
#         l2 = 1e9
#         move = 0.2
#         # reshape to perform vector operations
#         xnew = np.zeros(nelx * nely)
#         while (l2 - l1) / (l1 + l2) > 1e-3:
#             lmid = 0.5 * (l2 + l1)
#             xnew[:] =  np.maximum(0.0, np.maximum(x - move, np.minimum(1.0,
#                 np.minimum(x + move, x * np.sqrt(-dc / dv / lmid)))))
#             gt = g + np.sum((dv * (xnew - x)))
#             if gt > 0:
#                 l1 = lmid
#             else:
#                 l2 = lmid
#         return (xnew, gt)
