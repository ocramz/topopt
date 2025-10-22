# Mirror Descent for Topology Optimization

## Mathematical Foundation

### Problem Formulation

The topology optimization problem is:

```
minimize: c(ρ) = f^T u(ρ)  (compliance)
subject to:
  K(ρ) u(ρ) = f            (equilibrium)
  Σ ρ_e ≤ V_frac           (volume constraint)
  0 ≤ ρ_e ≤ 1              ∀e
```

where:
- `ρ` = density variables (design variables)
- `u` = displacements
- `K` = stiffness matrix (symmetric: K = K^T)
- `f` = force vector

### Lagrangian Formulation

The augmented Lagrangian is:

```
L(ρ, λ, μ) = c(ρ) + λ(g(ρ)) + (μ/2)(g(ρ))^2
```

where:
- `λ` = Lagrange multiplier (dual variable)
- `μ` = penalty parameter
- `g(ρ) = Σ ρ_e / n - V_frac` = constraint function

The algorithm alternates between:
1. **Primal step**: minimize `L` w.r.t. `ρ` using mirror descent
2. **Dual step**: update `λ` based on constraint satisfaction

## Mirror Descent on the Simplex

### Why Mirror Descent?

Since design variables represent a **probability distribution** (non-negative, often summed/averaged):
- Standard gradient descent uses Euclidean geometry (L2 distance)
- Mirror descent uses **KL divergence** (natural for probability distributions)

### Algorithm

**Mirror descent in log-space:**

For each iteration:

```
1. Compute gradient: g_t = ∇_ρ L(ρ_t, λ_t, μ_t)

2. Primal mirror step (log-space):
   log_ρ_{t+1} = log_ρ_t - α * g_t
   
3. Projection via softmax (entropy-based):
   ρ_{t+1} = exp(log_ρ_{t+1}) / Σ_j exp(log_ρ_{t+1}_j)
   
4. Dual update (gradient ascent):
   λ_{t+1} = λ_t + β * g(ρ_{t+1})
   
5. Penalty increase (if needed):
   μ_{t+1} = γ * μ_t  (γ > 1, e.g., γ = 1.1)
```

where:
- `α` = learning_rate (primal step size)
- `β` = dual_step_size (typically much smaller than α)
- `γ` = penalty growth rate

### Key Properties

1. **Automatic bound satisfaction**: Since `softmax` maps R^n → simplex, we automatically get `0 ≤ ρ ≤ 1`

2. **KL convergence**: Mirror descent with KL divergence converges with O(1/t) rate for convex problems

3. **Constraint handling**: Augmented Lagrangian ensures asymptotic feasibility while maintaining convexity structure

## Adjoint Method for Sensitivity

### Self-Adjoint Compliance Gradient

For compliance `c = f^T u` with symmetric stiffness `K`:

```
∂c/∂ρ_e = ∂(f^T u)/∂ρ_e
         = u^T (∂K/∂ρ_e) u    [since λ = u by symmetry]
```

**Key insight**: No need to solve additional linear system!
- Standard FEM: solve `K u = f` (one system)
- Adjoint (non-symmetric): solve `K u = f` AND `K^T λ = ∇_u c`
- **Adjoint (symmetric)**: solve only `K u = f`, use `λ = u` directly

This is what `ComplianceFunction` exploits:
1. Forward: solve FEM, compute sensitivities in `problem.compute_objective()`
2. Backward: return stored sensitivities multiplied by gradient from PyTorch

## Implementation Details

### Custom Autograd Function

```python
class ComplianceFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_phys, problem):
        # Solve FEM and compute gradients
        x_np = x_phys.detach().cpu().numpy()
        dobj = np.zeros_like(x_np)
        obj = problem.compute_objective(x_np, dobj)  # FEM solve here
        
        # Store gradient for backward
        ctx.dobj = torch.from_numpy(dobj).float()
        return torch.tensor(obj)
    
    @staticmethod
    def backward(ctx, grad_output):
        # Return stored gradient scaled by upstream gradient
        dobj = ctx.dobj.to(x_phys.device)
        grad_x = dobj * grad_output
        return grad_x, None
```

### Volume Constraint Gradient

Since `g(ρ) = Σ ρ_e / n - V_frac`:

```
∂g/∂ρ_e = 1/n    ∀e
```

This is constant and simple to compute.

## Convergence Analysis

### Convergence Rate

For the augmented Lagrangian method with mirror descent:

1. **Objective convergence**: O(1/√t) in expectation (first-order)
2. **Constraint satisfaction**: O(1/√t) in constraint violation
3. **Dual convergence**: Linear (exponential) under certain conditions

Compared to:
- **MMA** (NLopt): Superlinear (fast but black-box)
- **Gradient Descent** (Euclidean): O(1/t) slower on simplex geometry

### Parameter Tuning

**Learning rate `α`:**
- Too small: slow convergence
- Too large: oscillation or divergence
- Sweet spot: 0.01–0.1 depending on problem size

**Dual step `β`:**
- Should satisfy: `β << α`
- Typical: `β = α / 10` or smaller
- Controls constraint satisfaction tightness

**Penalty `μ`:**
- Start: `μ_0 = 1.0`
- Growth: `μ_{k+1} = 1.1 * μ_k` when `|g(ρ)| > ε`
- Effect: tighter constraint satisfaction over time

## Performance Comparison

### NLopt MMA
- ✅ Proven, production-quality
- ✅ Superlinear convergence
- ❌ External dependency
- ❌ Black-box, hard to customize
- ❌ No automatic differentiation

### PyTorch Mirror Descent
- ✅ Pure Python/PyTorch (no dependencies)
- ✅ Automatic differentiation
- ✅ Easy to customize/extend
- ✅ GPU-ready
- ✅ Exploits problem structure (self-adjoint K)
- ⚠️ First-order: more iterations needed
- ⚠️ Parameters need tuning

### When to Use Mirror Descent

**Better choice for:**
- Large-scale problems (GPU acceleration)
- Multi-material/multi-physics extensions
- Research & development
- Educational purposes
- Problems with special structure (self-adjoint, hierarchical)

**Better choice for MMA:**
- Production black-box optimization
- When robustness is critical
- Limited tuning time

## Future Extensions

### Accelerated Methods

Add Nesterov momentum:
```
z_{t+1} = ρ_t - α * g_t  # extrapolated point
ρ_{t+1} = (1-β) * z_t + β * ρ_t
```

### Adaptive Learning Rates

Use Adam-style adaptation:
```
m_t = β1 * m_{t-1} + (1-β1) * g_t          # 1st moment
v_t = β2 * v_{t-1} + (1-β2) * g_t^2       # 2nd moment
α_t = α / (√v_t + ε)                      # adaptive rate
```

### Constraint-Aware Acceleration

Adapt dual step based on constraint violation:
```
β_t = β_0 * (1 + |g(ρ_t)|)   # faster dual when infeasible
```

### GPU Acceleration

Move PyTorch tensors to GPU:
```python
x_torch = torch.from_numpy(x).float().to('cuda')
```

## References

- Mirror descent: Beck & Teboulle (2003), "Mirror descent meets fixed share (and feels no regret)"
- Augmented Lagrangian: Nocedal & Wright (2006), "Numerical Optimization"
- Topology optimization: Bendsøe & Sigmund (2003), "Topology Optimization: Theory, Methods, and Applications"
