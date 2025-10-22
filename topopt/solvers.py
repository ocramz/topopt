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
        obj = problem.compute_objective(x_np, dobj)
        
        # Save for backward pass
        ctx.save_for_backward(x_phys)
        ctx.problem = problem
        ctx.dobj = torch.from_numpy(dobj).float()
        
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


class TopOptSolver:
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
