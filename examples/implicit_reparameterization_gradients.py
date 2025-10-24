"""
Implicit Reparameterization Gradients

This script demonstrates how to compute gradients of expectations E_q[f(z)]
w.r.t. distribution parameters φ when the inverse CDF is intractable.

The key insight: Instead of computing ∇_φ z = ∇_φ F^{-1}(ε), we use
implicit differentiation on F(z|φ) = ε to get:

    ∇_φ z = -(∇_z F)^{-1} ∇_φ F

This avoids inverting the CDF and instead only requires differentiating it.
"""

import numpy as np
import torch
import torch.nn as nn
from scipy import special, stats
from scipy.optimize import fminbound
import matplotlib.pyplot as plt

from betaincder import betaincderp, betaincderq


# ============================================================================
# EXAMPLE 1: UNIVARIATE NORMAL DISTRIBUTION
# ============================================================================

class NormalReparameterizationGradients:
    """
    Compute reparameterization gradients for Normal distribution.
    
    Both explicit and implicit formulations are equivalent here, serving as
    a sanity check for the implicit differentiation approach.
    """
    
    def __init__(self, mu_init=0.0, sigma_init=1.0):
        """Initialize Normal distribution parameters."""
        self.mu = torch.tensor(mu_init, requires_grad=True)
        self.sigma = torch.tensor(sigma_init, requires_grad=True)
    
    def sample(self, n_samples=100):
        """Sample from N(μ, σ²)."""
        eps = torch.randn(n_samples)
        z = self.mu + self.sigma * eps
        return z, eps
    
    def explicit_reparameterization(self, z, eps):
        """
        Explicit reparameterization:
        z = μ + σ·ε
        
        So: ∂z/∂μ = 1, ∂z/∂σ = ε
        """
        dz_dmu = torch.ones_like(z)
        dz_dsigma = eps
        return dz_dmu, dz_dsigma
    
    def implicit_reparameterization(self, z, eps):
        """
        Implicit reparameterization using standardization function:
        S(z|μ,σ) = (z - μ)/σ = ε  (standardized normal)
        
        Total derivative: ∇^TD [S(z|μ,σ) = ε]
        Expand: (∂S/∂z)·(∂z/∂φ) + (∂S/∂φ) = 0
        
        So: ∂z/∂φ = -(∂S/∂z)^{-1} · (∂S/∂φ)
        
        For Normal:
        - S = (z - μ)/σ
        - ∂S/∂z = 1/σ
        - ∂S/∂μ = -1/σ
        - ∂S/∂σ = -(z-μ)/σ²
        """
        # ∂S/∂z
        dS_dz = 1.0 / self.sigma
        
        # ∂S/∂μ = -1/σ
        dS_dmu = -1.0 / self.sigma
        
        # ∂S/∂σ = -(z-μ)/σ²
        dS_dsigma = -(z - self.mu) / (self.sigma ** 2)
        
        # Apply implicit formula: ∂z/∂φ = -(∂S/∂z)^{-1} · (∂S/∂φ)
        dz_dmu_implicit = -dS_dmu / dS_dz  # = -(-1/σ) / (1/σ) = 1
        dz_dsigma_implicit = -dS_dsigma / dS_dz  # = -(-(z-μ)/σ²) / (1/σ) = (z-μ)/σ
        
        return dz_dmu_implicit, dz_dsigma_implicit
    
    def verify_equivalence(self, n_samples=10000):
        """
        Verify that explicit and implicit formulations give identical results.
        """
        z, eps = self.sample(n_samples)
        
        dz_dmu_exp, dz_dsigma_exp = self.explicit_reparameterization(z, eps)
        dz_dmu_imp, dz_dsigma_imp = self.implicit_reparameterization(z, eps)
        
        error_mu = torch.abs(dz_dmu_exp - dz_dmu_imp).max().item()
        error_sigma = torch.abs(dz_dsigma_exp - dz_dsigma_imp).max().item()
        
        print("\n" + "="*70)
        print("EXAMPLE 1: Normal Distribution")
        print("="*70)
        print("\nEquivalence Check (should be ~0):")
        print(f"  Max error in ∂z/∂μ: {error_mu:.2e}")
        print(f"  Max error in ∂z/∂σ: {error_sigma:.2e}")
        
        # Compute gradient of expectation E_q[f(z)] where f(z) = z²
        def objective(z):
            return (z ** 2).mean()
        
        # Test with explicit
        z, eps = self.sample(n_samples)
        dz_dmu_exp, dz_dsigma_exp = self.explicit_reparameterization(z, eps)
        
        df_dz = 2 * z
        
        grad_mu_exp = (df_dz * dz_dmu_exp).mean()
        grad_sigma_exp = (df_dz * dz_dsigma_exp).mean()
        
        # Test with implicit
        dz_dmu_imp, dz_dsigma_imp = self.implicit_reparameterization(z, eps)
        grad_mu_imp = (df_dz * dz_dmu_imp).mean()
        grad_sigma_imp = (df_dz * dz_dsigma_imp).mean()
        
        print("\nGradient of E[z²] (explicit vs implicit):")
        print(f"  ∇_μ E[f]: {grad_mu_exp.item():.6f} vs {grad_mu_imp.item():.6f}")
        print(f"  ∇_σ E[f]: {grad_sigma_exp.item():.6f} vs {grad_sigma_imp.item():.6f}")
        
        # Verify against numerical gradient (finite difference)
        eps_fd = 1e-4
        
        mu_orig = self.mu.item()
        self.mu.data = torch.tensor(mu_orig + eps_fd)
        f_plus = objective(self.sample(n_samples)[0])
        self.mu.data = torch.tensor(mu_orig - eps_fd)
        f_minus = objective(self.sample(n_samples)[0])
        grad_mu_fd = (f_plus - f_minus) / (2 * eps_fd)
        
        sigma_orig = self.sigma.item()
        self.sigma.data = torch.tensor(sigma_orig + eps_fd)
        f_plus = objective(self.sample(n_samples)[0])
        self.sigma.data = torch.tensor(sigma_orig - eps_fd)
        f_minus = objective(self.sample(n_samples)[0])
        grad_sigma_fd = (f_plus - f_minus) / (2 * eps_fd)
        
        self.mu.data = torch.tensor(mu_orig)
        self.sigma.data = torch.tensor(sigma_orig)
        
        print("\nFinite Difference Verification:")
        print(f"  ∇_μ E[f] (FD): {grad_mu_fd.item():.6f}")
        print(f"  ∇_σ E[f] (FD): {grad_sigma_fd.item():.6f}")


# ============================================================================
# EXAMPLE 2: BETA DISTRIBUTION
# ============================================================================

class BetaReparameterizationGradients:
    """
    Compute implicit reparameterization gradients for Beta distribution.
    
    For Beta(α, β) on [0,1], the CDF is the regularized incomplete Beta function:
    F(z|α,β) = I_z(α, β) = regularized incomplete beta
    
    The inverse CDF is intractable, but we can compute gradients of F using
    the betaincder package.
    """
    
    def __init__(self, alpha_init=2.0, beta_init=2.0):
        """Initialize Beta distribution parameters."""
        self.alpha = torch.tensor(alpha_init, requires_grad=True, dtype=torch.float32)
        self.beta = torch.tensor(beta_init, requires_grad=True, dtype=torch.float32)
    
    def sample_scipy(self, n_samples=100):
        """Sample from Beta(α, β) using scipy."""
        alpha_np = self.alpha.detach().cpu().numpy()
        beta_np = self.beta.detach().cpu().numpy()
        z = np.random.beta(alpha_np, beta_np, n_samples)
        return torch.from_numpy(z).float()
    
    def beta_cdf(self, z_np, alpha_np, beta_np):
        """Compute regularized incomplete Beta function (CDF)."""
        return special.betainc(alpha_np, beta_np, z_np)
    
    def beta_cdf_grad_alpha(self, z_np, alpha_np, beta_np):
        """Compute ∂F/∂α using betaincder."""
        return betaincderp(z_np, alpha_np, beta_np)
    
    def beta_cdf_grad_beta(self, z_np, alpha_np, beta_np):
        """Compute ∂F/∂β using betaincder."""
        return betaincderq(z_np, alpha_np, beta_np)
    
    def pdf_beta(self, z_np, alpha_np, beta_np):
        """Compute probability density function q(z|α,β)."""
        return special.beta(alpha_np, beta_np) ** (-1) * (
            z_np ** (alpha_np - 1) * (1 - z_np) ** (beta_np - 1)
        )
    
    def implicit_reparameterization(self, z_np, alpha_np, beta_np):
        """
        Implicit reparameterization for Beta distribution.
        
        Standardization function: S(z|α,β) = F(z|α,β) ~ Uniform(0,1)
        
        Implicit formula: ∂z/∂φ = -(∂S/∂z)^{-1} · (∂S/∂φ)
                                 = -(∂F/∂z)^{-1} · (∂F/∂φ)
                                 = -q(z)^{-1} · (∂F/∂φ)
        
        Where q(z) is the PDF of Beta(α,β).
        """
        # betaincder works with scalars, so we vectorize
        dF_dalpha = np.array([betaincderp(float(z_i), float(alpha_np), float(beta_np)) 
                             for z_i in z_np])
        dF_dbeta = np.array([betaincderq(float(z_i), float(alpha_np), float(beta_np)) 
                            for z_i in z_np])
        
        # Compute PDF at z
        q_z = self.pdf_beta(z_np, alpha_np, beta_np)
        
        # ∂z/∂α = -(∂F/∂z)^{-1} · (∂F/∂α) = -q(z)^{-1} · (∂F/∂α)
        dz_dalpha = -dF_dalpha / q_z
        
        # ∂z/∂β = -(∂F/∂z)^{-1} · (∂F/∂β) = -q(z)^{-1} · (∂F/∂β)
        dz_dbeta = -dF_dbeta / q_z
        
        return dz_dalpha, dz_dbeta
    
    def compute_gradient_of_expectation(self, f_func, n_samples=10000):
        """
        Compute ∇_φ E_q[f(z)] where φ = (α, β).
        
        Using implicit reparameterization:
        E_q[f(z)] = E_ε[f(z(ε))]  where ε ~ Uniform(0,1)
        ∇_φ E[f] = E[∇_z f(z) · ∇_φ z]
        """
        # Sample from Beta
        z = self.sample_scipy(n_samples).numpy()
        
        # Extract parameters
        alpha_np = self.alpha.item()
        beta_np = self.beta.item()
        
        # Compute implicit reparameterization gradients
        dz_dalpha, dz_dbeta = self.implicit_reparameterization(z, alpha_np, beta_np)
        
        # Compute function and its gradient
        z_torch = torch.from_numpy(z).float()
        z_torch.requires_grad_(True)
        f_z = f_func(z_torch)
        df_dz = torch.autograd.grad(f_z.sum(), z_torch, create_graph=False)[0].detach().numpy()
        
        # Chain rule: ∇_φ E[f] = E[∇_z f · ∇_φ z]
        grad_alpha = (df_dz * dz_dalpha).mean()
        grad_beta = (df_dz * dz_dbeta).mean()
        
        return grad_alpha, grad_beta, f_z.detach().mean().item()
    
    def optimize(self, f_func, n_iterations=50, learning_rate=0.1, n_samples=5000):
        """
        Optimize expectation E_q[f(z)] w.r.t. Beta parameters.
        """
        print("\n" + "="*70)
        print("EXAMPLE 2: Beta Distribution")
        print("="*70)
        
        history = {'alpha': [], 'beta': [], 'objective': [], 'grad_alpha': [], 'grad_beta': []}
        
        print(f"\nOptimizing E_Beta[f(z)] where f(z) = {f_func.__name__}")
        print(f"{'Iter':>5} {'α':>8} {'β':>8} {'E[f]':>12} {'∇α':>12} {'∇β':>12}")
        print("-" * 70)
        
        for it in range(n_iterations):
            # Compute gradients using implicit reparameterization
            grad_alpha, grad_beta, obj = self.compute_gradient_of_expectation(f_func, n_samples)
            
            # Store history
            history['alpha'].append(self.alpha.item())
            history['beta'].append(self.beta.item())
            history['objective'].append(obj)
            history['grad_alpha'].append(grad_alpha)
            history['grad_beta'].append(grad_beta)
            
            # Gradient descent (note: negative for minimization)
            with torch.no_grad():
                self.alpha -= learning_rate * grad_alpha
                self.beta -= learning_rate * grad_beta
                
                # Ensure parameters stay > 0
                self.alpha.clamp_(min=0.1)
                self.beta.clamp_(min=0.1)
            
            if it % 10 == 0 or it == n_iterations - 1:
                print(f"{it:5d} {self.alpha.item():8.4f} {self.beta.item():8.4f} "
                      f"{obj:12.6f} {grad_alpha:12.6e} {grad_beta:12.6e}")
        
        return history


# ============================================================================
# EXAMPLE 3: VISUALIZATION AND COMPARISON
# ============================================================================

def visualize_beta_optimization(f_func, history, output_filename='implicit_reparameterization_gradients.png'):
    """
    Create visualizations of Beta distribution optimization.
    
    Parameters
    ----------
    f_func : callable
        Objective function f(z)
    history : dict
        Optimization history with keys: 'alpha', 'beta', 'objective', 'grad_alpha', 'grad_beta'
    output_filename : str
        Name of output PNG file
    """
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Evolution of parameters
    ax = axes[0, 0]
    ax.plot(history['alpha'], label='α', linewidth=2)
    ax.plot(history['beta'], label='β', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Parameter Value')
    ax.set_title('Evolution of Beta Parameters')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Objective value
    ax = axes[0, 1]
    ax.plot(history['objective'], linewidth=2, color='green')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('E[f(z)]')
    ax.set_title(f'Objective: E[{f_func.__name__}]')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Gradients
    ax = axes[1, 0]
    ax.semilogy(np.abs(history['grad_alpha']), label='|∇α|', linewidth=2)
    ax.semilogy(np.abs(history['grad_beta']), label='|∇β|', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('|Gradient|')
    ax.set_title('Gradient Magnitudes (log scale)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Final distribution
    ax = axes[1, 1]
    alpha_final = history['alpha'][-1]
    beta_final = history['beta'][-1]
    z_plot = np.linspace(0.001, 0.999, 1000)
    pdf_final = stats.beta.pdf(z_plot, alpha_final, beta_final)
    ax.fill_between(z_plot, pdf_final, alpha=0.3, color='blue')
    ax.plot(z_plot, pdf_final, linewidth=2, color='blue', 
            label=f'Beta({alpha_final:.2f}, {beta_final:.2f})')
    ax.set_xlabel('z')
    ax.set_ylabel('Density')
    ax.set_title('Final Learned Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    print(f"\n✅ Visualization saved: {output_filename}")
    plt.close()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("IMPLICIT REPARAMETERIZATION GRADIENTS")
    print("="*70)
    
    # Example 1: Normal distribution (sanity check)
    print("\n" + "="*70)
    normal_solver = NormalReparameterizationGradients(mu_init=0.0, sigma_init=1.0)
    normal_solver.verify_equivalence(n_samples=10000)
    
    # Example 2: Beta distribution 
    
    print("\n" + "="*70)
    beta_solver = BetaReparameterizationGradients(alpha_init=1.5, beta_init=1.5)

    n_iterations = 100
    
    # def cost_fun(z):
    #     return (z - 0.25) ** 2
    # cost_fun.__name__ = "(z - 0.25)²"

    def cost_fun(z):
        return torch.sin(0.25 * z)
    cost_fun.__name__ = "sin(0.25·z)"
    
    history = beta_solver.optimize(cost_fun, n_iterations= n_iterations, learning_rate=0.1)
    
    print("\n" + "="*70)
    print(f"RESULTS (Objective: {cost_fun.__name__})")
    print("="*70)
    print(f"\nInitial Beta parameters: α=1.5, β=1.5")
    print(f"Final Beta parameters:   α={history['alpha'][-1]:.4f}, β={history['beta'][-1]:.4f}")
    print(f"\nInitial E[{cost_fun.__name__}]: {history['objective'][0]:.6f}")
    print(f"Final E[{cost_fun.__name__}]:   {history['objective'][-1]:.6f}")
    print(f"Improvement:         {history['objective'][0] - history['objective'][-1]:.6f}")
    
    # Visualize optimization
    visualize_beta_optimization(cost_fun, history, 'implicit_reparameterization.png')
    
    print("\n" + "="*70)
    print("✅ Implicit Reparameterization Gradients Demo Complete")
    print("="*70)
