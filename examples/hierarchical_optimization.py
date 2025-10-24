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
        """
        Evaluate constraint with numpy array.
        
        For scipy 'ineq' type: constraint(x) >= 0 means feasible
        We want: x1^2 + x2^2 <= bound
        So we return: bound - (x1^2 + x2^2)
        """
        return float(self.bound - (x[0]**2 + x[1]**2))
    
    def gradient(self, x):
        """
        Compute analytical gradient.
        
        d/dx (bound - x1^2 - x2^2) = [-2*x1, -2*x2]
        """
        return -2.0 * x


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
        x_optima = []
        gradients_at_opt = []
        
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
            x_optima.append(x_opt)
            
            # Gradient of objective at optimal point
            grad = obj_func.gradient(x_opt)
            gradients_at_opt.append(grad)
        
        # Expected objective and average gradient at optimal points
        expected_obj = np.mean(objectives)
        avg_grad_at_opt = np.mean(gradients_at_opt, axis=0)
        
        # Debug: print first few optimal designs and objectives
        if alpha_np[0] > 4.0:  # Only print later iterations
            print(f"\n  [Forward] alpha={alpha_np}, beta={beta_np}")
            print(f"  [Forward] First 3 samples: x_opt={x_optima[:3]}, f_opt={objectives[:3]}")
            print(f"  [Forward] Expected obj={expected_obj:.6f}, avg_grad={avg_grad_at_opt}")
        
        # Now compute implicit differentiation
        # For Beta distribution: E[x] = α/(α+β)
        # dE[x]/dα = β/(α+β)²
        # dE[x]/dβ = -α/(α+β)²
        
        # The gradient w.r.t. α and β comes from:
        # dL/dα = (df/dx)|_{x*} · (dE[x]/dα)
        ab_sum = alpha_np + beta_np
        d_mean_d_alpha = beta_np / (ab_sum ** 2)
        d_mean_d_beta = -alpha_np / (ab_sum ** 2)
        
        # These gradients tell us how the expected value changes
        grad_alpha_implicit = avg_grad_at_opt * d_mean_d_alpha
        grad_beta_implicit = avg_grad_at_opt * d_mean_d_beta
        
        # Save for backward
        ctx.save_for_backward(alpha, beta)
        ctx.grad_alpha_implicit = torch.from_numpy(grad_alpha_implicit).float()
        ctx.grad_beta_implicit = torch.from_numpy(grad_beta_implicit).float()
        ctx.x_optima = x_optima
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
        grad_alpha_implicit = ctx.grad_alpha_implicit
        grad_beta_implicit = ctx.grad_beta_implicit
        
        # Chain through with grad_output (keeping shape [2])
        grad_alpha = grad_alpha_implicit * grad_output
        grad_beta = grad_beta_implicit * grad_output
        
        # None for the module parameters
        return grad_alpha, grad_beta, None, None, None


# ============================================================================
# 3. BETA-PARAMETERIZED SOLVER
# ============================================================================

class BetaParameterizedSolver:
    """
    Solve optimization problem using Beta-parameterized design variables
    with implicit differentiation.
    
    OUTPUT: Optimal Beta parameters α*, β* such that when you sample
    x ~ Beta(α*, β*) and solve the constrained problem, you get a good solution.
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
        
        # Beta parameters: these are what we optimize
        # Initialized to Beta(2,2) for each dimension
        self.alpha_logit = nn.Parameter(torch.tensor([0.0, 0.0]))
        self.beta_logit = nn.Parameter(torch.tensor([0.0, 0.0]))
        
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
        }
    
    def _get_alpha_beta(self):
        """
        Get α and β from logit parameters, ensuring both > 1.
        
        Returns
        -------
        torch.Tensor, torch.Tensor
            Alpha and beta parameters, each with shape [2]
        """
        alpha = F.softplus(self.alpha_logit) + 1.0
        beta = F.softplus(self.beta_logit) + 1.0
        return alpha, beta
    
    def optimize(self):
        """
        Optimize Beta parameters.
        
        The goal is to find α*, β* such that:
          E_ρ[f*(ρ)] is minimized
        where f*(ρ) = min_x f(x) s.t. g(x) <= 0, x initialized at ρ
        
        Returns
        -------
        dict
            Optimization history
        """
        print("\nBeta-Parameterized Optimization")
        print("=" * 70)
        print("Finding optimal Beta parameters α* and β*")
        print(f"{'Iter':>5} {'Expected Obj':>15} {'α[0]':>8} {'β[0]':>8} {'α[1]':>8} {'β[1]':>8}")
        print("-" * 70)
        
        for iteration in range(self.n_iterations):
            self.optimizer.zero_grad()
            
            # Get current α, β (these are the parameters being optimized)
            alpha, beta = self._get_alpha_beta()
            
            # Compute expected objective via implicit differentiation
            # This samples x ~ Beta(α, β), solves the constrained problem,
            # and returns E[f*(x)]
            obj = ConstrainedOptimizer.apply(
                alpha, beta, self.objective_module, self.constraint_module,
                self.n_samples
            )
            
            # Backward pass - computes gradients of expected objective w.r.t. α and β
            obj.backward()
            
            self.optimizer.step()
            
            # Track history
            alpha_val, beta_val = self._get_alpha_beta()
            
            self.history['objective'].append(obj.item())
            self.history['alpha'].append(alpha_val.detach().cpu().numpy().copy())
            self.history['beta'].append(beta_val.detach().cpu().numpy().copy())
            
            # Print progress
            if iteration % 10 == 0 or iteration == self.n_iterations - 1:
                print(f"{iteration:5d} {obj.item():15.6f} {alpha_val[0].item():8.4f} "
                      f"{beta_val[0].item():8.4f} {alpha_val[1].item():8.4f} {beta_val[1].item():8.4f}")
        
        print("-" * 70)
        print("Optimization complete!\n")
        return self.history
    
    def get_optimal_parameters(self):
        """
        Get optimal Beta parameters.
        
        This is THE main output of the solver.
        Users will sample from Beta(α*, β*) to get designs.
        
        Returns
        -------
        dict with keys:
            'alpha' : np.ndarray, shape (2,)
                Optimal alpha parameters for each design variable
            'beta' : np.ndarray, shape (2,)
                Optimal beta parameters for each design variable
        """
        alpha_final, beta_final = self._get_alpha_beta()
        alpha_np = alpha_final.detach().cpu().numpy()
        beta_np = beta_final.detach().cpu().numpy()
        
        return {
            'alpha': alpha_np,
            'beta': beta_np,
        }
    
    def sample_designs(self, n_designs=100):
        """
        Sample designs from the optimized Beta distribution.
        
        This is how you USE the trained solver.
        
        Parameters
        ----------
        n_designs : int
            Number of design samples
            
        Returns
        -------
        np.ndarray, shape (n_designs, 2)
            Design samples from Beta(α*, β*)
        """
        params = self.get_optimal_parameters()
        alpha = params['alpha']
        beta = params['beta']
        
        # Sample from the Beta distribution for each design variable
        designs = np.column_stack([
            np.random.beta(alpha[0], beta[0], n_designs),
            np.random.beta(alpha[1], beta[1], n_designs)
        ])
        
        return designs
    
    def evaluate_designs(self, designs, constraint_module):
        """
        For sampled designs, solve the constrained problem and evaluate.
        
        This shows how to USE the learned distribution in practice.
        
        Parameters
        ----------
        designs : np.ndarray, shape (n_designs, 2)
            Initial design samples
        constraint_module : nn.Module
            Constraint function
            
        Returns
        -------
        dict with keys:
            'designs_init': Initial sampled designs
            'designs_optimized': Optimized designs (after solving constrained problem)
            'objectives': Objective values at optimized designs
            'feasible': Boolean array indicating feasibility
        """
        obj_func = ObjectiveNumpy()
        constraint_func = ConstraintNumpy(constraint_module.bound)
        
        designs_optimized = []
        objectives = []
        feasible = []
        
        for x_init in designs:
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
            
            designs_optimized.append(result.x)
            objectives.append(result.fun)
            # Check feasibility: x1^2 + x2^2 <= 0.5
            c_val = result.x[0]**2 + result.x[1]**2 - constraint_module.bound
            feasible.append(c_val <= 1e-6)
        
        return {
            'designs_init': designs,
            'designs_optimized': np.array(designs_optimized),
            'objectives': np.array(objectives),
            'feasible': np.array(feasible),
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
    print("\n" + "="*70)
    print("TRAINING THE BETA SOLVER")
    print("="*70)
    
    solver = BetaParameterizedSolver(
        objective_module, constraint_module,
        learning_rate=0.05, n_samples=50, n_iterations=100
    )
    history = solver.optimize()
    
    # ========================================================================
    # MAIN OUTPUT: Optimal Beta Parameters
    # ========================================================================
    optimal_params = solver.get_optimal_parameters()
    
    print("="*70)
    print("OPTIMAL BETA PARAMETERS (Main Output)")
    print("="*70)
    print("\nThese are the parameters α* and β* of the learned distribution.")
    print("When you need a design, sample x ~ Beta(α*, β*) and solve the")
    print("constrained problem starting from that sample.\n")
    
    print(f"α* = {optimal_params['alpha']}")
    print(f"β* = {optimal_params['beta']}")
    print(f"\nInterpretation:")
    print(f"  x1 ~ Beta({optimal_params['alpha'][0]:.4f}, {optimal_params['beta'][0]:.4f})")
    print(f"  x2 ~ Beta({optimal_params['alpha'][1]:.4f}, {optimal_params['beta'][1]:.4f})")
    
    # ========================================================================
    # HOW TO USE THE LEARNED DISTRIBUTION
    # ========================================================================
    print("\n" + "="*70)
    print("USING THE LEARNED DISTRIBUTION")
    print("="*70)
    
    # Sample designs from the learned distribution
    n_samples_user = 20
    sampled_designs = solver.sample_designs(n_designs=n_samples_user)
    print(f"\nSampled {n_samples_user} designs from Beta(α*, β*):")
    print(f"  Shape: {sampled_designs.shape}")
    print(f"  Mean: [{sampled_designs.mean(axis=0)[0]:.4f}, {sampled_designs.mean(axis=0)[1]:.4f}]")
    print(f"  Std:  [{sampled_designs.std(axis=0)[0]:.4f}, {sampled_designs.std(axis=0)[1]:.4f}]")
    
    # Solve constrained problem for each sampled design
    print(f"\nSolving constrained problem for each sampled design...")
    results = solver.evaluate_designs(sampled_designs, constraint_module)
    
    print(f"\nResults:")
    print(f"  Total samples: {len(results['feasible'])}")
    print(f"  Feasible: {results['feasible'].sum()}/{len(results['feasible'])}")
    print(f"  Objective values:")
    print(f"    Mean: {results['objectives'].mean():.6f}")
    print(f"    Std:  {results['objectives'].std():.6f}")
    print(f"    Min:  {results['objectives'].min():.6f}")
    print(f"    Max:  {results['objectives'].max():.6f}")
    
    print(f"\nOptimized designs:")
    print(f"  Mean: [{results['designs_optimized'].mean(axis=0)[0]:.4f}, "
          f"{results['designs_optimized'].mean(axis=0)[1]:.4f}]")
    
    # ========================================================================
    # COMPARISON
    # ========================================================================
    print("\n" + "="*70)
    print("COMPARISON WITH BASELINE")
    print("="*70)
    
    print("\nBaseline (single constrained optimization):")
    print(f"  Design: x* = [{baseline['x'][0]:.4f}, {baseline['x'][1]:.4f}]")
    print(f"  Objective: f(x*) = {baseline['f']:.6f}")
    
    print("\nBeta-parameterized (distribution of solutions):")
    print(f"  Optimal parameters: α* = {optimal_params['alpha']}, β* = {optimal_params['beta']}")
    print(f"  Expected objective (over 20 samples): {results['objectives'].mean():.6f}")
    print(f"  Design variety (std of objectives): {results['objectives'].std():.6f}")
    
    print("\nKey difference:")
    print("  - Baseline: Single point x*")
    print("  - Beta method: Distribution Beta(α*, β*)")
    print("              Users sample from this to get designs")
    print("              Each sample, when optimized, yields a good solution")
