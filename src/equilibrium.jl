struct EquilibriumProblem
    f
    Df
    parameters
end

#-

function validated_solve(pb::EquilibriumProblem, c_init::Vector)
    pb_approx = EquilibriumProblem(pb.f, pb.Df, mid.(pb.parameters))

    # Step 2
    c_bar, newton_success = approximate_zero(pb_approx, c_init)
    @assert newton_success

    # Step 3
    A = approximate_inverse(pb_approx, c_bar)

    # Step 4
    #-
    Y = Y_bound(pb, interval(c_bar), interval(A))
    R = interval(10sup(Y))
    Z = Z_bound(pb, interval(c_bar), interval(A), R)
    #-
    ie, proof_success = interval_of_existence(Y, Z, R; verbose = true)
    @assert proof_success

    #
    return interval.(c_bar, inf(ie); format = :midpoint)
end





# Step 1

F(pb::EquilibriumProblem, u) = pb.f(u, pb.parameters)

DF(pb::EquilibriumProblem, u) = pb.Df(u, pb.parameters)





# Step 2

function approximate_zero(pb_approx::EquilibriumProblem, c_init)
    F_DF_newton(c) = F(pb_approx, c), DF(pb_approx, c)
    return newton(F_DF_newton, c_init; verbose = true)
end





# Step 3

approximate_inverse(pb_approx::EquilibriumProblem, c_bar) =
    inv(DF(pb_approx, c_bar))





# Step 4

Y_bound(pb::EquilibriumProblem, c_bar, A) = norm(A * F(pb, c_bar), 1)

function Z_bound(pb::EquilibriumProblem, c_bar, A, R)
    box_c = interval.(c_bar, R; format = :midpoint)
    return opnorm(interval(I) - A * DF(pb, box_c), 1)
end
