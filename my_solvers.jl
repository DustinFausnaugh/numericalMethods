using LinearAlgebra
"""
Conj_Grad 

A function where we take a vector, traverse and there is always a vector 90 degrees to it. 
This allows us to instead of walking down "Stairs" , we push the stair inward and create a ramp which increases computation.

"""
function conj_grad(A, b)
    #Setup
    n = size(A, 1)
    initialPosition = zeros(n) 
    initialResidual = b - A * initialPosition 
    direction = copy(initialResidual) 

    for i in 1:n
        numeratorAlpha = dot(initialResidual, initialResidual)
        Ad = A * direction
        alpha = numeratorAlpha / dot(direction, Ad)

        # Update position
        initialPosition += alpha * direction 

        # Update residual
        newResidual = initialResidual - (alpha * Ad)

        if norm(newResidual) < 1e-10
            break
        end

        # Calculate beta
        beta = dot(newResidual, newResidual) / numeratorAlpha

        # Update direction
        direction = newResidual + (beta * direction)
        initialResidual = newResidual
    end
    return initialPosition

end

#  LUP---
"""
computeLUP

function computeLUP, computes LUP
"""
function computeLUP(A)
    n = size(A, 1)
    U = copy(A)
    L = Matrix{Float64}(I, n, n)
    p = collect(1:n)
    for k = 1:n-1
        val, id = findmax(abs.(U[k:end, k]))
        pivot_row = id + k - 1
        U[k, :], U[pivot_row, :] = U[pivot_row, :], U[k, :]
        p[k], p[pivot_row] = p[pivot_row], p[k]
        if k > 1
            L[k, 1:k-1], L[pivot_row, 1:k-1] = L[pivot_row, 1:k-1], L[k, 1:k-1]
        end
        for i = k+1:n
            m = U[i, k] / U[k, k]
            L[i, k] = m
            U[i, k:end] .-= m .* U[k, k:end]
        end
    end
    return L, U, p
end
"""
LUPsolve

Function LUPsolve does standard LUP decomp, then we have a timing and accuracy section
"""
function LUPsolve(A, b)
    L, U, p = computeLUP(A)
    n = length(b)
    b_perm = b[p]
    y = zeros(n)
    for i = 1:n
        y[i] = b_perm[i] - sum(L[i, 1:i-1] .* y[1:i-1])
    end
    x = zeros(n)
    for i = n:-1:1
        x[i] = (y[i] - sum(U[i, i+1:end] .* x[i+1:end])) / U[i, i]
    end
    return x
end

# --- Testing and Timing Section ---

println("--- Starting Accuracy and Timing Tests ---")

for N in [10, 100, 1000]
    B = rand(N, N)
    A = B' * B + I
    # Use vec() to ensure b is a 1D vector for the CG math
    b = vec(rand(N, 1))
    
    println("\nTesting for N = $N")
    
    #Test LUP
    println("Running LUP...")
    x_lup = @time LUPsolve(A, b)
    error_lup = norm(A * x_lup - b)
    println("LUP Residual Error: ", error_lup)

    #Test Conjugate Gradient
    println("Running Conj_Grad...")
    cg = @time conj_grad(A, b)
    
    
    residualError = norm(A * cg - b)
    println("CG Residual Error  : ", residualError)
end