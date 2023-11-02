# https://docs.sciml.ai/NeuralPDE/stable/tutorials/pdesystem/

using NeuralPDE, Lux, Optimization, OptimizationOptimJL
import ModelingToolkit: Interval

@parameters φ z
@variables p(..)
Dφ = Differential(φ)
Dz = Differential(z)

L, R, c0, η = 20e-3, 20e-3, 50e-6, 0.04
ρ = 800
ω = 2π*50
ε = 0.5
ε̇ = 0.005
γ = π
γ̇ = ω
u_1 = R*ω
u_2 = 0.0

h(φ) = c0*(1-ε*cos(φ-γ))
dhdt(φ) = c0*(-ε̇*cos(φ-γ)-ε*γ̇*sin(φ-γ))

# 2D PDE
eq = Dz(ρ*h(φ)^3/η *Dz(p(φ,z))) ~ 6/R*Dφ(h(φ)*(u_1 + u_2)) + 12*dhdt(φ)

# Boundary conditions
bcs = [ p(φ, -L/2) ~ 0.0,
        p(φ, L/2) ~ 0.0,
        Dz(p(φ, 0)) ~ 0.0,
        p(γ, z) ~ 0.0]
# Space and time domains
domains = [ φ ∈ Interval(0.0, 2π),
            z ∈ Interval(-L/2, L/2)]

# Neural network
dim = 2 # number of dimensions
# chain = Lux.Chain(Dense(dim, 128, Lux.σ), Dense(128, 32, Lux.σ), Dense(32, 32, Lux.σ), Dense(32, 1))
chain = Lux.Chain(Dense(dim, 16, Lux.σ), Dense(16, 16, Lux.σ), Dense(16, 1))

# Discretization
dx = [2π/5, L/10]
discretization = PhysicsInformedNN(chain, GridTraining(dx))

@named pde_system = PDESystem(eq, bcs, domains, [φ, z], [p(φ, z)])
prob = discretize(pde_system, discretization)

#Optimizer
opt = OptimizationOptimJL.BFGS()

#Callback function
callback = function (pp, l)
    println("Current loss is: $l")
    return false
end

res = Optimization.solve(prob, opt, callback = callback, maxiters = 100)
phi = discretization.phi

using Plots

xs, ys = [infimum(d.domain):(ddx / 10):supremum(d.domain) for (d,ddx) in zip(domains,dx)]
# analytic_sol_func(x, y) = (sin(pi * x) * sin(pi * y)) / (2pi^2)

u_predict = reshape([first(phi([x, y], res.u)) for x in xs for y in ys],
                    (length(xs), length(ys)))
# u_real = reshape([analytic_sol_func(x, y) for x in xs for y in ys],
#                  (length(xs), length(ys)))
# diff_u = abs.(u_predict .- u_real)

# p1 = plot(xs, ys, u_real, linetype = :contourf, title = "analytic");
# p3 = plot(xs, ys, diff_u, linetype = :contourf, title = "error");
# plot(p2, p3)

plot(xs, ys, u_predict', linetype = :contourf, title = "Pressure distribution short bearing", 
    xticks = (0:π/2:2π, ["0","π/2","π","3π/2","2π"]), 
    yticks = (-L/2:L/4:L/2, ["-L/2","-L/4","0","L/4","L/2"]))
xlabel!("φ")
ylabel!("z")