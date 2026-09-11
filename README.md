# Numerical Methods

## Problem
This repo has two Julia solvers for linear systems Ax = b, plus a script that benchmarks them against each other.

conj_grad is an iterative method for symmetric positive definite systems. Rather than taking axis-aligned steps toward the solution, it picks a sequence of mutually orthogonal (A conjugate) search directions, so each step corrects error the previous steps couldn't touch. It converges in at most n steps in exact arithmetic, often far fewer in practice. ComputeLUP and LUPsolve implement a direct solve instead, factoring A into PA = LU with partial pivoting, then solving via forward substitution (Ly = Pb) followed by back substitution (Ux = y).
## Tools used
You'll need Julia (tested on 1.x) — LinearAlgebra is standard library, so there's nothing extra to install. Run julia my_solvers.jl and it walks through N = 10, 100, 1000, generating a random symmetric positive-definite A and random b at each size, solving with both methods, timing each with @time, and printing the residual error ‖Ax - b‖ for both.
## Execution
To use the functions elsewhere, just include("my_solvers.jl") and call LUPsolve(A, b) or conj_grad(A, b) directly. Keep in mind conj_grad assumes A is symmetric positive-definite and isn't guaranteed to converge otherwise, LUPsolve handles general square non-singular matrices, and CG's convergence tolerance is set at 1e-10 on the residual norm.
