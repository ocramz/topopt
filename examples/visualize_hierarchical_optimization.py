"""
Visualization of Hierarchical Optimization Problem

This script creates comprehensive matplotlib visualizations of:
1. Objective function surface
2. Constraint surface
3. Combined surface with constraint boundary
4. Contour plots with constraint region
5. Optimization trajectories

Run from command line:
    python examples/visualize_hierarchical_optimization.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Circle
import torch
from hierarchical_optimization import NonlinearObjective, EllipticalConstraint


# ============================================================================
# VISUALIZATION SETUP
# ============================================================================

def create_grid(x_min=0, x_max=1, y_min=0, y_max=1, resolution=100):
    """
    Create a regular grid for evaluation.
    
    Parameters
    ----------
    x_min, x_max : float
        Range for x1
    y_min, y_max : float
        Range for x2
    resolution : int
        Number of points in each dimension
        
    Returns
    -------
    tuple
        (X, Y) grid arrays
    """
    x = np.linspace(x_min, x_max, resolution)
    y = np.linspace(y_min, y_max, resolution)
    X, Y = np.meshgrid(x, y)
    return X, Y


def evaluate_objective(X, Y, objective_module):
    """
    Evaluate objective function on grid.
    
    Parameters
    ----------
    X, Y : np.ndarray
        Grid arrays
    objective_module : nn.Module
        Objective function
        
    Returns
    -------
    np.ndarray
        Function values on grid
    """
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x_point = torch.tensor([X[i, j], Y[i, j]]).float()
            Z[i, j] = objective_module(x_point).item()
    return Z


def evaluate_constraint(X, Y, constraint_module):
    """
    Evaluate constraint function on grid.
    
    Parameters
    ----------
    X, Y : np.ndarray
        Grid arrays
    constraint_module : nn.Module
        Constraint function
        
    Returns
    -------
    np.ndarray
        Constraint values on grid
    """
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x_point = torch.tensor([X[i, j], Y[i, j]]).float()
            Z[i, j] = constraint_module(x_point).item()
    return Z



# ============================================================================
# FIGURE 2: CONTOUR PLOTS WITH ANALYSIS
# ============================================================================

def plot_contour_analysis(objective_module, constraint_module, 
                         output_file='contour_analysis.png'):
    """
    Create detailed contour plot analysis.
    
    Parameters
    ----------
    objective_module : nn.Module
        Objective function
    constraint_module : nn.Module
        Constraint function
    output_file : str
        Output filename
    """
    print("Generating contour analysis plots...")
    
    X, Y = create_grid(resolution=150)
    Z_obj = evaluate_objective(X, Y, objective_module)
    Z_const = evaluate_constraint(X, Y, constraint_module)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot 1: Objective contours
    ax = axes[0, 0]
    contour = ax.contourf(X, Y, Z_obj, levels=25, cmap='viridis')
    contour_lines = ax.contour(X, Y, Z_obj, levels=10, colors='black', 
                               alpha=0.3, linewidths=0.5)
    ax.clabel(contour_lines, inline=True, fontsize=8, fmt='%.2f')
    # Draw constraint
    theta = np.linspace(0, 2*np.pi, 200)
    x_circle = np.sqrt(0.5) * np.cos(theta)
    y_circle = np.sqrt(0.5) * np.sin(theta)
    ax.plot(x_circle, y_circle, 'r-', linewidth=3, label='Constraint boundary')
    ax.set_xlabel('x₁', fontsize=11, fontweight='bold')
    ax.set_ylabel('x₂', fontsize=11, fontweight='bold')
    ax.set_title('Objective Contours with Constraint', fontsize=12, fontweight='bold')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_aspect('equal')
    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label('f(x)', fontsize=10)
    ax.legend(fontsize=10)
    
    # Plot 2: Constraint contours (level sets)
    ax = axes[0, 1]
    contour = ax.contourf(X, Y, Z_const, levels=[-1, 0, 1], colors=['green', 'red'],
                          alpha=0.3)
    contour_lines = ax.contour(X, Y, Z_const, levels=[0], colors='black', 
                               linewidths=2.5)
    ax.clabel(contour_lines, inline=True, fontsize=10, fmt='%.2f')
    ax.fill_between(np.linspace(0, 1, 100), 0, 1, alpha=0.1, color='green',
                    label='Feasible region (g(x) ≤ 0)')
    ax.set_xlabel('x₁', fontsize=11, fontweight='bold')
    ax.set_ylabel('x₂', fontsize=11, fontweight='bold')
    ax.set_title('Constraint Region', fontsize=12, fontweight='bold')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_aspect('equal')
    ax.legend(fontsize=10, loc='upper left')
    
    # Plot 3: Objective magnitude (log scale for visibility)
    ax = axes[1, 0]
    Z_obj_log = np.log1p(np.abs(Z_obj))
    contour = ax.contourf(X, Y, Z_obj_log, levels=20, cmap='hot')
    theta = np.linspace(0, 2*np.pi, 200)
    x_circle = np.sqrt(0.5) * np.cos(theta)
    y_circle = np.sqrt(0.5) * np.sin(theta)
    ax.plot(x_circle, y_circle, 'c-', linewidth=3, label='Constraint boundary')
    ax.set_xlabel('x₁', fontsize=11, fontweight='bold')
    ax.set_ylabel('x₂', fontsize=11, fontweight='bold')
    ax.set_title('Objective Magnitude (Log Scale)', fontsize=12, fontweight='bold')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_aspect('equal')
    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label('log(1 + |f(x)|)', fontsize=10)
    ax.legend(fontsize=10)
    
    # Plot 4: Combined objective and constraint
    ax = axes[1, 1]
    contour = ax.contourf(X, Y, Z_obj, levels=20, cmap='viridis', alpha=0.7)
    # Highlight constraint region
    Z_const_feasible = np.where(Z_const <= 0, 1, 0)
    ax.contourf(X, Y, Z_const_feasible, levels=[0.5, 1.5], colors=['green'], alpha=0.2)
    # Draw constraint boundary
    ax.contour(X, Y, Z_const, levels=[0], colors='red', linewidths=3)
    ax.set_xlabel('x₁', fontsize=11, fontweight='bold')
    ax.set_ylabel('x₂', fontsize=11, fontweight='bold')
    ax.set_title('Objective with Feasible Region (shaded)', fontsize=12, fontweight='bold')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_aspect('equal')
    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label('f(x)', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


# ============================================================================
# FIGURE 3: GRADIENT FIELDS AND VECTOR ANALYSIS
# ============================================================================

def plot_gradient_field(objective_module, constraint_module,
                       output_file='gradient_field.png'):
    """
    Create gradient field visualization.
    
    Parameters
    ----------
    objective_module : nn.Module
        Objective function
    constraint_module : nn.Module
        Constraint function
    output_file : str
        Output filename
    """
    print("Generating gradient field plots...")
    
    # Create coarser grid for gradient arrows
    X, Y = create_grid(resolution=150)
    Z_obj = evaluate_objective(X, Y, objective_module)
    Z_const = evaluate_constraint(X, Y, constraint_module)
    
    # Compute gradients numerically
    eps = 1e-5
    dfdx1 = np.zeros_like(X)
    dfdx2 = np.zeros_like(X)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x_p = torch.tensor([X[i, j] + eps, Y[i, j]]).float()
            x_m = torch.tensor([X[i, j] - eps, Y[i, j]]).float()
            y_p = torch.tensor([X[i, j], Y[i, j] + eps]).float()
            y_m = torch.tensor([X[i, j], Y[i, j] - eps]).float()
            
            dfdx1[i, j] = (objective_module(x_p).item() - objective_module(x_m).item()) / (2*eps)
            dfdx2[i, j] = (objective_module(y_p).item() - objective_module(y_m).item()) / (2*eps)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Objective with gradient field
    ax = axes[0]
    contour = ax.contourf(X, Y, Z_obj, levels=20, cmap='viridis', alpha=0.6)
    
    # Downsample for clearer vector field
    step = 8
    X_sparse = X[::step, ::step]
    Y_sparse = Y[::step, ::step]
    dfdx1_sparse = dfdx1[::step, ::step]
    dfdx2_sparse = dfdx2[::step, ::step]
    
    # Normalize for visualization
    grad_mag = np.sqrt(dfdx1_sparse**2 + dfdx2_sparse**2) + 1e-6
    u_norm = dfdx1_sparse / grad_mag
    v_norm = dfdx2_sparse / grad_mag
    
    ax.quiver(X_sparse, Y_sparse, u_norm, v_norm, grad_mag, 
             cmap='cool', scale=20, width=0.003, alpha=0.7)
    
    # Draw constraint
    theta = np.linspace(0, 2*np.pi, 200)
    x_circle = np.sqrt(0.5) * np.cos(theta)
    y_circle = np.sqrt(0.5) * np.sin(theta)
    ax.plot(x_circle, y_circle, 'r-', linewidth=3, label='Constraint boundary')
    
    ax.set_xlabel('x₁', fontsize=11, fontweight='bold')
    ax.set_ylabel('x₂', fontsize=11, fontweight='bold')
    ax.set_title('Objective Function with Gradient Field', fontsize=12, fontweight='bold')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_aspect('equal')
    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label('f(x)', fontsize=10)
    ax.legend(fontsize=10)
    
    # Plot 2: Constraint with gradient field
    ax = axes[1]
    contour = ax.contourf(X, Y, Z_const, levels=np.linspace(-0.5, 0.5, 20), 
                         cmap='RdYlGn_r', alpha=0.6)
    
    # Constraint gradient
    dgdx1 = 2 * X
    dgdx2 = 2 * Y
    
    grad_mag_c = np.sqrt(dgdx1**2 + dgdx2**2) + 1e-6
    u_norm_c = dgdx1 / grad_mag_c
    v_norm_c = dgdx2 / grad_mag_c
    
    X_sparse_c = X[::step, ::step]
    Y_sparse_c = Y[::step, ::step]
    u_sparse_c = u_norm_c[::step, ::step]
    v_sparse_c = v_norm_c[::step, ::step]
    
    ax.quiver(X_sparse_c, Y_sparse_c, u_sparse_c, v_sparse_c, 
             np.ones_like(u_sparse_c), cmap='cool', scale=20, width=0.003, alpha=0.7)
    
    # Draw constraint boundary
    ax.plot(x_circle, y_circle, 'k-', linewidth=3, label='Constraint boundary')
    ax.fill(x_circle, y_circle, alpha=0.1, color='green', label='Feasible region')
    
    ax.set_xlabel('x₁', fontsize=11, fontweight='bold')
    ax.set_ylabel('x₂', fontsize=11, fontweight='bold')
    ax.set_title('Constraint Function with Gradient Field', fontsize=12, fontweight='bold')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_aspect('equal')
    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label('g(x)', fontsize=10)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()



# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("VISUALIZING HIERARCHICAL OPTIMIZATION PROBLEM")
    print("="*70)
    
    # Create problem modules
    objective_module = NonlinearObjective()
    constraint_module = EllipticalConstraint(bound=0.5)
    
    print("\nProblem:")
    print("  minimize  f(x) = (x₁-2)⁴ + (x₁-2x₂)² + exp(x₂-1) - 1")
    print("  subject to g(x) = x₁² + x₂² - 0.5 ≤ 0")
    print("            0 ≤ x₁, x₂ ≤ 1")
    
    print("\n" + "-"*70)
    print("Generating visualizations...")
    print("-"*70)
    
    
    plot_contour_analysis(objective_module, constraint_module,
                         output_file='contour_analysis.png')
    
    plot_gradient_field(objective_module, constraint_module,
                       output_file='gradient_field.png')
    
    
    print("\n" + "-"*70)
    print("✓ All visualizations complete!")
    print("-"*70)
    print("\nGenerated files:")
    print("  2. contour_analysis.png - Detailed contour analysis")
    print("  3. gradient_field.png - Gradient vector fields")
    print("\n" + "="*70 + "\n")
