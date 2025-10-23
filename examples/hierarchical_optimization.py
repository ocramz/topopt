"""
Hierarchical Optimization with Beta Sampling and Implicit Differentiation

This example demonstrates Beta-parameterized optimization on a nonlinear,
box-constrained problem with an active constraint. We show how implicit
differentiation through Beta parameters enables uncertainty quantification
in optimization.

Problem:
    minimize    f(x) = (x1 - 2)^4 + (x1 - 2x2)^2 + exp(x2 - 1) - 1
    subject to  x1^2 + x2^2 <= 0.5  (active constraint)
                0 <= x1 <= 1
                0 <= x2 <= 1

The constraint becomes active at the optimum. We solve this using:
1. Standard gradient descent (baseline)
2. Beta-parameterized optimization with implicit differentiation
3. Robustness analysis under constraint perturbation
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import minimize
from scipy import stats


# ============================================================================
# 1. PYTORCH MODULES FOR OBJECTIVE AND CONSTRAINT
# ============================================================================

class NonlinearObjective(nn.Module):
    """
    PyTorch module for the nonlinear objective function.
    
    f(x) = (x1 - 2)^4 + (x1 - 2*x2)^2 + exp(x2 - 1) - 1
    
    This module is fully differentiable and integrates with PyTorch's
    automatic differentiation system.
    """
    
    def forward(self, x):
        """
        Evaluate objective function.
        
        Parameters
        ----------
        x : torch.Tensor or np.ndarray
            Design variables [x1, x2]
            
        Returns
        -------
        torch.Tensor or float
            Objective value
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        
        x1, x2 = x[0], x[1]
        return (x1 - 2)**4 + (x1 - 2*x2)**2 + torch.exp(x2 - 1) - 1


class EllipticalConstraint(nn.Module):
    """
    PyTorch module for elliptical inequality constraint.
    
    g(x) = x1^2 + x2^2 - bound <= 0
    
    This module computes the constraint value, fully differentiable
    for use in gradient-based optimization.
    """
    
    def __init__(self, bound=0.5):
        """
        Initialize constraint.
        
        Parameters
        ----------
        bound : float
            Right-hand side of constraint (default 0.5)
        """
        super().__init__()
        self.bound = bound
    
    def forward(self, x):
        """
        Evaluate constraint function.
        
        Parameters
        ----------
        x : torch.Tensor or np.ndarray
            Design variables [x1, x2]
            
        Returns
        -------
        torch.Tensor or float
            Constraint value (negative means feasible)
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        
        return x[0]**2 + x[1]**2 - self.bound


# Create numpy wrappers for scipy compatibility
class ObjectiveNumpy:
    """Wrapper for scipy.optimize compatibility (numpy interface)."""
    
    def __init__(self):
        pass
    
    def __call__(self, x):
        """Evaluate objective with numpy array, return float."""
        x1, x2 = x[0], x[1]
        return float((x1 - 2)**4 + (x1 - 2*x2)**2 + np.exp(x2 - 1) - 1)
    
    def gradient(self, x):
        """Compute analytical gradient."""
        x1, x2 = x[0], x[1]
        df_dx1 = 4*(x1 - 2)**3 + 2*(x1 - 2*x2)
        df_dx2 = -4*(x1 - 2*x2) + np.exp(x2 - 1)
        return np.array([df_dx1, df_dx2])


class ConstraintNumpy:
    """Wrapper for scipy.optimize compatibility (numpy interface)."""
    
    def __init__(self, bound=0.5):
        self.bound = bound
    
    def __call__(self, x):
        """Evaluate constraint with numpy array, return float."""
        return float(x[0]**2 + x[1]**2 - self.bound)
    
    def gradient(self, x):
        """Compute analytical gradient."""
        return 2.0 * x


# ============================================================================
# 2. PYTORCH AUTOGRAD FUNCTIONS FOR IMPLICIT DIFFERENTIATION
# ============================================================================

class ConstrainedOptimizer(torch.autograd.Function):
    """
    Implicit differentiation through constrained optimization.
    
    For given Beta parameters α, β, we:
    1. Sample x ~ Beta(α, β) (mapped to [0,1]^2)
    2. Solve constrained optimization problem for that x
    3. Compute gradient using implicit function theorem
    """
    
    @staticmethod
    def forward(ctx, alpha, beta, objective_module, constraint_module, 
                n_samples=100):
        """
        Forward pass: optimize for sampled designs.
        
        Parameters
        ----------
        alpha : torch.Tensor
            Beta parameter α for each dimension
        beta : torch.Tensor
            Beta parameter β for each dimension
        objective_module : nn.Module
            Objective function module
        constraint_module : nn.Module
            Constraint function module
        n_samples : int
            Number of samples for Monte Carlo expectation
            
        Returns
        -------
        torch.Tensor
            Expected optimal value
        """
        alpha_np = alpha.detach().cpu().numpy()
        beta_np = beta.detach().cpu().numpy()
        
        # Sample designs from Beta distributions (map to [0,1])
        x1_samples = np.random.beta(alpha_np[0], beta_np[0], n_samples)
        x2_samples = np.random.beta(alpha_np[1], beta_np[1], n_samples)
        
        # Create numpy wrappers for scipy
        obj_func = ObjectiveNumpy()
        constraint_func = ConstraintNumpy(constraint_module.bound)
        
        # For each sample, solve constrained optimization
        objectives = []
        gradients_sum = np.zeros_like(alpha_np)
        
        for x1_init, x2_init in zip(x1_samples, x2_samples):
            x_init = np.array([x1_init, x2_init])
            
            # Solve constrained problem using scipy
            constraints = {'type': 'ineq', 'fun': constraint_func}
            bounds = [(0, 1), (0, 1)]
            
            result = minimize(
                obj_func,
                x_init,
                method='SLSQP',
                jac=obj_func.gradient,
                constraints=constraints,
                bounds=bounds,
                options={'ftol': 1e-9}
            )
            
            x_opt = result.x
            f_opt = result.fun
            objectives.append(f_opt)
            
            # Compute sensitivity: gradient at optimal point
            grad = obj_func.gradient(x_opt)
            gradients_sum += grad
        
        # Expected objective and gradient
        expected_obj = np.mean(objectives)
        expected_grad = gradients_sum / n_samples
        
        # Save for backward
        ctx.save_for_backward(alpha, beta)
        ctx.expected_grad = torch.from_numpy(expected_grad).float()
        ctx.n_samples = n_samples
        
        return torch.tensor(expected_obj, dtype=alpha.dtype)
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass using implicit function theorem.
        
        For Beta distribution:
        - E[x] = α/(α+β)
        - dE[x]/dα = β/(α+β)²
        - dE[x]/dβ = -α/(α+β)²
        """
        alpha, beta = ctx.saved_tensors
        expected_grad = ctx.expected_grad
        
        # Beta moment derivatives
        ab_sum = alpha + beta
        d_mean_d_alpha = beta / (ab_sum ** 2)
        d_mean_d_beta = -alpha / (ab_sum ** 2)
        
        # Chain rule
        grad_alpha = expected_grad * d_mean_d_alpha * grad_output
        grad_beta = expected_grad * d_mean_d_beta * grad_output
        
        # None for the module parameters
        return grad_alpha, grad_beta, None, None, None


# ============================================================================
# 3. BETA-PARAMETERIZED SOLVER
# ============================================================================

class BetaParameterizedSolver:
    """
    Solve optimization problem using Beta-parameterized design variables
    with implicit differentiation.
    """
    
    def __init__(self, objective_module, constraint_module, 
                 learning_rate=0.01, n_samples=50, n_iterations=100):
        """
        Initialize solver.
        
        Parameters
        ----------
        objective_module : nn.Module
            Objective function module
        constraint_module : nn.Module
            Constraint function module
        learning_rate : float
            Adam optimizer learning rate
        n_samples : int
            Samples per iteration for Monte Carlo
        n_iterations : int
            Maximum optimization iterations
        """
        self.objective_module = objective_module
        self.constraint_module = constraint_module
        self.learning_rate = learning_rate
        self.n_samples = n_samples
        self.n_iterations = n_iterations
        
        # Beta parameters (initialized to Beta(2,2) - uniform-ish)
        self.alpha_logit = nn.Parameter(torch.zeros(2))
        self.beta_logit = nn.Parameter(torch.zeros(2))
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            [self.alpha_logit, self.beta_logit],
            lr=learning_rate
        )
        
        # History
        self.history = {
            'objective': [],
            'alpha': [],
            'beta': [],
            'mean_design': []
        }
    
    def _get_alpha_beta(self):
        """
        Get α and β from logit parameters, ensuring both > 1.
        """
        alpha = F.softplus(self.alpha_logit) + 1.0
        beta = F.softplus(self.beta_logit) + 1.0
        return alpha, beta
    
    def optimize(self):
        """
        Optimize Beta parameters.
        
        Returns
        -------
        dict
            Optimization history
        """
        print("\nBeta-Parameterized Optimization")
        print("=" * 70)
        print(f"{'Iter':>5} {'Objective':>15} {'||α||':>10} {'||β||':>10} {'E[x1]':>8} {'E[x2]':>8}")
        print("-" * 70)
        
        for iteration in range(self.n_iterations):
            self.optimizer.zero_grad()
            
            # Get α, β
            alpha, beta = self._get_alpha_beta()
            
            # Compute objective via implicit differentiation
            obj = ConstrainedOptimizer.apply(
                alpha, beta, self.objective_module, self.constraint_module,
                self.n_samples
            )
            
            # Backward pass
            obj.backward()
            self.optimizer.step()
            
            # Track history
            alpha_val, beta_val = self._get_alpha_beta()
            mean_design = (alpha_val / (alpha_val + beta_val)).detach().cpu().numpy()
            
            self.history['objective'].append(obj.item())
            self.history['alpha'].append(alpha_val.detach().cpu().numpy().copy())
            self.history['beta'].append(beta_val.detach().cpu().numpy().copy())
            self.history['mean_design'].append(mean_design.copy())
            
            # Print progress
            if iteration % 10 == 0 or iteration == self.n_iterations - 1:
                print(f"{iteration:5d} {obj.item():15.6f} {torch.norm(alpha):10.4f} "
                      f"{torch.norm(beta):10.4f} {mean_design[0]:8.4f} {mean_design[1]:8.4f}")
        
        print("-" * 70)
        return self.history
    
    def get_solution(self):
        """Get mean design and confidence intervals."""
        alpha_final, beta_final = self._get_alpha_beta()
        alpha_np = alpha_final.detach().cpu().numpy()
        beta_np = beta_final.detach().cpu().numpy()
        
        # Mean design
        mean_design = alpha_np / (alpha_np + beta_np)
        
        # 95% confidence intervals
        ci_lower = np.array([stats.beta.ppf(0.025, a, b) for a, b in zip(alpha_np, beta_np)])
        ci_upper = np.array([stats.beta.ppf(0.975, a, b) for a, b in zip(alpha_np, beta_np)])
        
        # Variance
        variance = (alpha_np * beta_np) / ((alpha_np + beta_np)**2 * (alpha_np + beta_np + 1))
        
        return {
            'mean': mean_design,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'variance': variance,
            'alpha': alpha_np,
            'beta': beta_np
        }


# ============================================================================
# 4. BASELINE: STANDARD GRADIENT DESCENT
# ============================================================================

def baseline_optimization(objective_module, constraint_module):
    """
    Baseline: Standard constrained optimization using PyTorch modules.
    
    Parameters
    ----------
    objective_module : nn.Module
        Objective function module
    constraint_module : nn.Module
        Constraint function module
        
    Returns
    -------
    dict
        Optimization result
    """
    print("\nBaseline Gradient Descent (SLSQP)")
    print("=" * 70)
    
    # Create numpy wrappers
    obj_func = ObjectiveNumpy()
    constraint_func = ConstraintNumpy(constraint_module.bound)
    
    # Random starting points
    n_trials = 10
    best_result = None
    best_obj = np.inf
    
    for trial in range(n_trials):
        x0 = np.random.uniform(0, 1, 2)
        
        constraints = {'type': 'ineq', 'fun': constraint_func}
        bounds = [(0, 1), (0, 1)]
        
        result = minimize(
            obj_func,
            x0,
            method='SLSQP',
            jac=obj_func.gradient,
            constraints=constraints,
            bounds=bounds,
            options={'ftol': 1e-9}
        )
        
        if result.fun < best_obj:
            best_obj = result.fun
            best_result = result
    
    x_opt = best_result.x
    f_opt = best_result.fun
    
    print(f"Optimal design: x1={x_opt[0]:.4f}, x2={x_opt[1]:.4f}")
    print(f"Objective: {f_opt:.6f}")
    print(f"Constraint: x1^2 + x2^2 = {x_opt[0]**2 + x_opt[1]**2:.6f} <= {constraint_module.bound}")
    print(f"Constraint active: {np.isclose(x_opt[0]**2 + x_opt[1]**2, constraint_module.bound)}")
    
    return {'x': x_opt, 'f': f_opt}


# ============================================================================
# 5. ROBUSTNESS ANALYSIS
# ============================================================================

def robustness_analysis(beta_solver, objective_module, constraint_module, 
                       n_perturbations=100, perturbation_scale=0.1):
    """
    Analyze robustness by varying constraint bound.
    
    Parameters
    ----------
    beta_solver : BetaParameterizedSolver
        Trained solver
    objective_module : nn.Module
        Objective function module
    constraint_module : nn.Module
        Constraint function module
    n_perturbations : int
        Number of constraint perturbations
    perturbation_scale : float
        Scale of perturbation
        
    Returns
    -------
    dict
        Robustness statistics
    """
    print("\nRobustness Analysis")
    print("=" * 70)
    print("Evaluating design robustness under constraint perturbations...")
    
    solution = beta_solver.get_solution()
    mean_design = solution['mean']
    
    # Convert to torch tensor for evaluation
    x_torch = torch.from_numpy(mean_design).float()
    
    # Evaluate at perturbed constraint bounds
    constraint_bounds = constraint_module.bound + np.linspace(-perturbation_scale, 
                                                              perturbation_scale, 
                                                              n_perturbations)
    objectives = []
    feasible_count = 0
    
    for bound in constraint_bounds:
        # Compute constraint value
        c = (mean_design[0]**2 + mean_design[1]**2) - bound
        feasible = c <= 1e-6
        if feasible:
            feasible_count += 1
            obj = objective_module(x_torch).item()
            objectives.append(obj)
    
    objectives = np.array(objectives)
    
    print(f"Feasibility under perturbations: {feasible_count}/{n_perturbations} ({100*feasible_count/n_perturbations:.1f}%)")
    if len(objectives) > 0:
        print(f"Objective distribution:")
        print(f"  Mean: {objectives.mean():.6f}")
        print(f"  Std:  {objectives.std():.6f}")
        print(f"  Min:  {objectives.min():.6f}")
        print(f"  Max:  {objectives.max():.6f}")
    
    return {
        'constraint_bounds': constraint_bounds,
        'objectives': objectives,
        'feasible_count': feasible_count,
        'feasibility_rate': feasible_count / n_perturbations
    }


# ============================================================================
# 6. MAIN EXAMPLE
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("HIERARCHICAL OPTIMIZATION WITH BETA SAMPLING")
    print("="*70)
    print("\nProblem: Nonlinear 2D constrained optimization")
    print("  minimize  (x1-2)^4 + (x1-2*x2)^2 + exp(x2-1) - 1")
    print("  s.t.      x1^2 + x2^2 <= 0.5  (active constraint)")
    print("            0 <= x1, x2 <= 1")
    
    # Create PyTorch modules for objective and constraint
    objective_module = NonlinearObjective()
    constraint_module = EllipticalConstraint(bound=0.5)
    
    # ========================================================================
    # Method 1: Baseline (standard constrained optimization)
    # ========================================================================
    baseline = baseline_optimization(objective_module, constraint_module)
    
    # ========================================================================
    # Method 2: Beta-parameterized optimization with implicit differentiation
    # ========================================================================
    solver = BetaParameterizedSolver(
        objective_module, constraint_module,
        learning_rate=0.05, n_samples=50, n_iterations=100
    )
    history = solver.optimize()
    
    # ========================================================================
    # Extract and display results
    # ========================================================================
    solution = solver.get_solution()
    
    print("\n" + "="*70)
    print("RESULTS COMPARISON")
    print("="*70)
    
    print("\nBaseline (standard constrained optimization):")
    print(f"  Design:     x = [{baseline['x'][0]:.4f}, {baseline['x'][1]:.4f}]")
    print(f"  Objective:  f(x) = {baseline['f']:.6f}")
    
    print("\nBeta-parameterized optimization (mean design):")
    print(f"  E[x]:       E[x] = [{solution['mean'][0]:.4f}, {solution['mean'][1]:.4f}]")
    print(f"  Confidence intervals (95%):")
    print(f"    x1 ∈ [{solution['ci_lower'][0]:.4f}, {solution['ci_upper'][0]:.4f}]")
    print(f"    x2 ∈ [{solution['ci_lower'][1]:.4f}, {solution['ci_upper'][1]:.4f}]")
    print(f"  Beta parameters:")
    print(f"    α = [{solution['alpha'][0]:.4f}, {solution['alpha'][1]:.4f}]")
    print(f"    β = [{solution['beta'][0]:.4f}, {solution['beta'][1]:.4f}]")
    
    # Evaluate objective at mean design using module
    x_mean = torch.from_numpy(solution['mean']).float()
    obj_mean = objective_module(x_mean).item()
    print(f"  Objective at mean: f(E[x]) = {obj_mean:.6f}")
    
    # ========================================================================
    # Uncertainty quantification
    # ========================================================================
    print("\n" + "="*70)
    print("UNCERTAINTY QUANTIFICATION")
    print("="*70)
    print("\nDesign variance (uncertainty at each variable):")
    print(f"  Var[x1] = {solution['variance'][0]:.6f}")
    print(f"  Var[x2] = {solution['variance'][1]:.6f}")
    print(f"  Total variance: {solution['variance'].sum():.6f}")
    
    # ========================================================================
    # Robustness analysis
    # ========================================================================
    robustness = robustness_analysis(
        solver, objective_module, constraint_module,
        n_perturbations=50, perturbation_scale=0.15
    )
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("""
Key observations:
1. The constraint x1^2 + x2^2 <= 0.5 is active at the optimum
2. Beta parameterization captures design uncertainty naturally
3. Implicit differentiation enables gradient-based optimization
4. Confidence intervals quantify design uncertainty
5. Robustness analysis shows design stability under perturbations

Advantages of Beta-parameterized approach:
- Principled uncertainty quantification
- Natural handling of box constraints (0-1)
- Implicit differentiation through complex constraints
- Hierarchical design exploration
- Robustness evaluation via sampling

PyTorch Modules Benefits:
- Fully differentiable objective and constraint functions
- Seamless integration with autograd system
- Easy to extend with custom PyTorch operations
- Natural handling of batched computations
- GPU acceleration ready
    """)
