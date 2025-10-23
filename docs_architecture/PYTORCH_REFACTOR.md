# PyTorch Refactoring Summary

## Overview
The `TopOptSolver` has been refactored from using NLopt's MMA solver to a **first-order mirror descent method** with PyTorch autograd for automatic differentiation.

## Key Components

### 1. Custom Autograd Functions

#### `ComplianceFunction(torch.autograd.Function)`
- **Forward pass**: Solves FEM analysis using the problem's `compute_objective()` method
- **Backward pass**: Returns stored sensitivities (exploits self-adjoint property of stiffness matrix)
- **Efficiency**: No need to solve an additional adjoint system since K is symmetric (λ = u)

```python
# Usage in optimize loop:
obj_value = ComplianceFunction.apply(x_torch, self.problem)
```

#### `VolumeConstraint(torch.autograd.Function)`
- **Forward pass**: Computes `g(x) = sum(x)/n_elements - volfrac`
- **Backward pass**: Gradient is uniform `dg/dx = 1/n_elements`
- Enables PyTorch to differentiate through constraint values

### 2. Mirror Descent Optimization

The solver uses **mirror descent on the simplex**, which naturally exploits the probability distribution structure of design variables.

#### Algorithm (in `optimize()` method):

```
for each iteration:
    1. Evaluate objective: obj = ComplianceFunction(x)
    2. Evaluate constraint: g = VolumeConstraint(x, volfrac)
    3. Form augmented Lagrangian: L = obj + λ*g + (penalty/2)*g²
    4. Backward pass: compute ∇L via PyTorch autograd
    5. Mirror descent step in log-space:
       log_x ← log_x - learning_rate * ∇L
    6. Project back to simplex: x = softmax(log_x)
    7. Update dual variable: λ ← λ + dual_step_size * g
    8. Increase penalty if constraint violated
```

#### Key Features:

- **Augmented Lagrangian Method**: Handles volume constraint without explicit projection at each iteration
- **Natural Gradient (KL divergence)**: Uses log-space representation to naturally handle simplex geometry
- **Adaptive Penalty**: Penalty parameter increases if constraint not satisfied
- **Dual Ascent**: Lagrange multiplier updated based on constraint violation

### 3. Design Variables as Probability Distribution

Since design variables are non-negative and typically normalized, they form a **probability simplex**. Mirror descent leverages this:

- **Log-space updates**: `x_new ∝ exp(log_x_old - learning_rate * grad)`
- **Softmax projection**: Automatically maintains `x ∈ [0,1]` and implicit constraint
- **KL geometry**: Natural for probability distributions, better convergence than Euclidean

## Key Changes from NLopt

| Aspect | NLopt (MMA) | PyTorch (Mirror Descent) |
|--------|------------|-------------------------|
| Dependency | External `nlopt` library | PyTorch (widely available) |
| Gradients | User-provided callbacks | Automatic via autograd |
| Constraint Handling | Built into MMA algorithm | Augmented Lagrangian |
| Geometry | Euclidean space | Simplex (log-space) |
| FEM Coupling | Black-box through callbacks | Custom Function for direct integration |

## New Parameters

- **`learning_rate`** (default: 0.05): Step size for mirror descent updates
  - Tune down if oscillating, up if converging slowly
  
- **`dual_step_size`** (default: 0.01): Step size for Lagrange multiplier updates
  - Should be smaller than primal learning rate for stability

- **`penalty_param`** (internal): Increases from 1.0 as `1.1x` per iteration if constraint violated
  - Controls how strongly constraint violations are penalized

## Integration with Problem Classes

The `ComplianceFunction.forward()` calls `problem.compute_objective()`, which:
1. Solves FEM: `K u = f`
2. Computes compliance: `f^T u`
3. Computes sensitivities using adjoint method (exploiting `K^T = K`)

This maintains all existing FEM infrastructure and benefits from the self-adjoint optimization.

## Usage

```python
from topopt.solvers import TopOptSolver
from topopt.problems import ComplianceProblem
from topopt.filters import DensityFilter
from topopt.guis import GUI

# Create problem, filter, GUI as usual
problem = ComplianceProblem(bc, penalty=3.0)
filter = DensityFilter(nelx, nely, rmin)
gui = GUI(nelx, nely)

# Create solver with learning rate
solver = TopOptSolver(
    problem, 
    volfrac=0.4,
    filter=filter, 
    gui=gui,
    maxeval=2000,
    learning_rate=0.05,  # NEW: tune this parameter
    ftol_rel=1e-3
)

# Optimize
x_opt = solver.optimize(x_init)
```

## Performance Considerations

1. **Convergence**: First-order methods typically require more iterations than second-order (MMA), but each iteration is cheaper (no subproblem solves)

2. **Constraint Satisfaction**: Mirror descent naturally drives toward feasible region; adjust `dual_step_size` and `penalty_param` for tighter satisfaction

3. **Memory**: PyTorch tensors in GPU-friendly format (optional: add device support)

## Future Enhancements

1. **Accelerated Methods**: Add Nesterov momentum or adaptive learning rates
2. **GPU Support**: Move tensors to GPU for large problems
3. **Batch Evaluation**: Evaluate multiple constraint/objective combinations in parallel
4. **Multi-Load Problems**: Leverage PyTorch's batch operations for multiple load cases
