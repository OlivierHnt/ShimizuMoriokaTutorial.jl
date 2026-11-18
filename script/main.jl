using ShimizuMoriokaTutorial
using RadiiPolynomial
using LinearAlgebra



# Parameters
parameters = (interval(3)/interval(4), interval(9)/interval(20)) # (a, b)



# Section 2: Equilibria and eigenspaces

c0_star = interval([0, 0, 0])

pb_c1 = EquilibriumProblem(f, Df, parameters)
c1_star = validated_solve(pb_c1, [1.0, 0.0, 1.0])

#

# numerical eigendecompositions, used as initial guesses

λ0_approx, V0_approx = eigen(mid.(Df(c0_star, parameters)))
λ1_approx, V1_approx = eigen(mid.(Df(c1_star, parameters)))

λ0_approx  # eigenvalues of the origin: two real stable + one real unstable (saddle)
λ1_approx  # eigenvalues of c1: one real stable + two complex conjugate unstable (saddle-focus)

#- the two stable eigenpairs of the origin

# the normalization index l_star is that of the largest component of the approximate eigenvector
pb_eig0_1 = EigenpairProblem(Df, parameters, c0_star, argmax(abs.(V0_approx[:,1])))
v0_1, λ0_1 = validated_solve(pb_eig0_1, V0_approx[:,1], λ0_approx[1])
pb_eig0_2 = EigenpairProblem(Df, parameters, c0_star, argmax(abs.(V0_approx[:,2])))
v0_2, λ0_2 = validated_solve(pb_eig0_2, V0_approx[:,2], λ0_approx[2])

#- one unstable eigenpair of c1; the other is its complex conjugate

pb_eig1 = EigenpairProblem(Df, parameters, c1_star, argmax(abs.(V1_approx[:,2])))
v1_1, λ1_1 = validated_solve(pb_eig1, V1_approx[:,2], λ1_approx[2])



# Section 3

#- Stable manifold at the origin

λ0_stable = [λ0_1, λ0_2];
V0_stable = exact(0.5) * [v0_1/norm(v0_1) v0_2/norm(v0_2)];
pb_stable_manifold = ManifoldProblem(f, Df, parameters, c0_star, λ0_stable, V0_stable);
ν_Q = interval(1.1)
Q_bar, r_Q = validated_solve(pb_stable_manifold, 15, ν_Q);

#- Unstable manifold at c1

λ1_unstable = [λ1_1, conj(λ1_1)];
V1_unstable = exact(0.15) * [v1_1/norm(v1_1) conj(v1_1/norm(v1_1))];
pb_unstable_manifold = ManifoldProblem(f, Df, parameters, c1_star, λ1_unstable, V1_unstable);
ν_P = interval(1.1)
P_bar, r_P = validated_solve(pb_unstable_manifold, 22, ν_P);



# Section 4

pb_het = HeteroclinicProblem(f, Df, parameters,
    P_bar, [Derivative(1, 0) .* P_bar Derivative(0, 1) .* P_bar], r_P, ν_P,
    Q_bar, [Derivative(1, 0) .* Q_bar Derivative(0, 1) .* Q_bar], r_Q, ν_Q,
    interval(13));

x_bar, r = validated_solve(pb_het, 0.6, 125, interval(1.05));



# Plot

c0_plot = mid.(c0_star)
c1_plot = mid.(c1_star)
Q_plot = mid.(Q_bar)
P_plot = mid.(P_bar)
u_plot = mid.(x_bar[1:3])
α_plot = mid.(x_bar[4][1])
θ₁_plot = mid.(x_bar[5][1])
θ₂_plot = mid.(x_bar[6][1])

using GLMakie

fig = Figure()
ax = Axis3(fig[1,1]; viewmode = :free, aspect = :data)

#- chaotic attractor

using OrdinaryDiffEq

init_cond = rand(3)

pb = ODEProblem((du, u, p, t) -> du .= f(u, p), init_cond, (0.0, 2000.0), mid.(parameters))
sol = solve(pb)

lines!(ax,
    [Point3f(sol(t)) for t ∈ LinRange(200, 2000, 100000)], color = :grey, linewidth = 6e-2, transparency=true)

#

surface!(ax,
    [Q_plot[1](θ₁, θ₂) for θ₁ ∈ LinRange(-1, 1, 101), θ₂ ∈ LinRange(-1, 1, 101)],
    [Q_plot[2](θ₁, θ₂) for θ₁ ∈ LinRange(-1, 1, 101), θ₂ ∈ LinRange(-1, 1, 101)],
    [Q_plot[3](θ₁, θ₂) for θ₁ ∈ LinRange(-1, 1, 101), θ₂ ∈ LinRange(-1, 1, 101)];
    colormap = [:green, :green])

surface!(ax,
    [real(P_plot[1](θ₁ + im*θ₂, θ₁ - im*θ₂)) for θ₁ ∈ LinRange(-1, 1, 101), θ₂ ∈ LinRange(-1, 1, 101)],
    [real(P_plot[2](θ₁ + im*θ₂, θ₁ - im*θ₂)) for θ₁ ∈ LinRange(-1, 1, 101), θ₂ ∈ LinRange(-1, 1, 101)],
    [real(P_plot[3](θ₁ + im*θ₂, θ₁ - im*θ₂)) for θ₁ ∈ LinRange(-1, 1, 101), θ₂ ∈ LinRange(-1, 1, 101)];
    colormap = [:red, :red])

lines!(ax,
    [Point3f(real(P_plot[1](cis(α), cis(-α))), real(P_plot[2](cis(α), cis(-α))), real(P_plot[3](cis(α), cis(-α))))  for α ∈ LinRange(0, 2pi, 101)];
    color = :yellow, linewidth = 3)
meshscatter!(ax,
    [Point3f(real(P_plot[1](cis(α_plot), cis(-α_plot))), real(P_plot[2](cis(α_plot), cis(-α_plot))), real(P_plot[3](cis(α_plot), cis(-α_plot))))];
    color = :orange, markersize = 2e-2)

meshscatter!(ax,
    [Point3f(Q_plot[1](θ₁_plot, θ₂_plot), Q_plot[2](θ₁_plot, θ₂_plot), Q_plot[3](θ₁_plot, θ₂_plot))];
    color = :blue, markersize = 2e-2)

cmap = range(colorant"red", colorant"blue", 301)
lines!(ax,
    [Point3f(real(u_plot[1](s)), real(u_plot[2](s)), real(u_plot[3](s))) for s ∈ LinRange(-1, 1, 301)];
    color = cmap, linewidth = 3)
