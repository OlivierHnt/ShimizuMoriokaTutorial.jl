struct EigenpairProblem
    Df
    parameters
    equilibrium
    l_star # index of the normalization condition v[l_star] = 1
end

#-

function validated_solve(pb::EigenpairProblem, v_init::Vector, λ_init::Number)
    pb_approx = EigenpairProblem(pb.Df, mid.(pb.parameters), mid.(pb.equilibrium), pb.l_star)

    # Step 2
    x_bar, newton_success = approximate_zero(pb_approx, [v_init / v_init[pb.l_star] ; λ_init])
    @assert newton_success

    # Step 3
    A = approximate_inverse(pb_approx, x_bar)

    # Step 4
    #-
    Y = Y_bound(pb, interval(x_bar), interval(A))
    R = interval(10sup(Y))
    Z = Z_bound(pb, interval(x_bar), interval(A), R)
    #-
    ie, proof_success = interval_of_existence(Y, Z, R; verbose = true)
    @assert proof_success

    #
    x_star = box.(interval(x_bar), interval(inf(ie)))
    return x_star[1:3], x_star[4] # eigenvector, eigenvalue
end

box(x::Interval, R) = interval(x, R; format = :midpoint)
box(x::Complex{<:Interval}, R) = complex(box(real(x), R), box(imag(x), R))





# Step 1

function F(pb::EigenpairProblem, x)
    v, λ = x[1:3], x[4]
    M = pb.Df(pb.equilibrium, pb.parameters)
    return [M * v - λ * v ; v[pb.l_star] - exact(1)]
end

function DF(pb::EigenpairProblem, x)
    v, λ = x[1:3], x[4]
    M = pb.Df(pb.equilibrium, pb.parameters)
    e = zeros(typeof(λ), 1, 3)
    e[pb.l_star] = one(λ)
    return [M - Diagonal([λ, λ, λ])  -v
            e                         zero(λ)]
end





# Step 2

function approximate_zero(pb_approx::EigenpairProblem, x_init)
    F_DF_newton(x) = F(pb_approx, x), DF(pb_approx, x)
    return newton(F_DF_newton, x_init; verbose = true)
end





# Step 3

approximate_inverse(pb_approx::EigenpairProblem, x_bar) =
    inv(DF(pb_approx, x_bar))





# Step 4

Y_bound(pb::EigenpairProblem, x_bar, A) = norm(A * F(pb, x_bar), 1)

function Z_bound(pb::EigenpairProblem, x_bar, A, R)
    box_x = box.(x_bar, R)
    return opnorm(interval(I) - A * DF(pb, box_x), 1)
end
