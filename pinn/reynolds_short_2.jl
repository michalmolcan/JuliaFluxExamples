# https://docs.sciml.ai/NeuralPDE/stable/tutorials/pdesystem/

using NeuralPDE, Lux, Optimization
import ModelingToolkit: Interval

@parameters z
@variables p(..)
Dz = Differential(z)
Dzz = Differential(z)^2

L, R, c0, η = 20e-3, 20e-3, 50e-6, 0.04

ω = 2*π*20/60

ε, ε̇, γ, γ̇, sng, csg = (0.9482568039025231, -7.204510620111302e-6, -1.3135842352142397, 2.5949873579168545e-5, -0.9671029386427265, 0.25438534955575315) #pro ω = 2*π*20/60, prvni casovy okamzik
# ε, ε̇, γ, γ̇, sng, csg = (0.9482566754619792, -7.350427625890751e-6, -1.313583814862708, 2.1684538777483007e-5, -0.9671028317113698, 0.2543857560789323) #2
# ε, ε̇, γ, γ̇, sng, csg = (0.9482510501799758, -2.249393056454047e-6, -1.313640150191874, -0.00011783304464125771, -0.9671171610820338, 0.25433127361893104) #50
u_1 = R*ω
u_2 = 0.0

φ = 0.0
h = c0 * (1 - ε * cos(φ - γ))
dhdφ = c0 * ε * sin(φ - γ)
dhdt = -c0 * (ε̇ * cos(φ - γ) + ε * γ̇ * sin(φ - γ))

ff = 6 * η / R / h^3 * dhdφ * (u_1 + u_2) + 12 * η / h^3 * dhdt

# 2D PDE
eq = Dzz(p(z)) - ff ~ 0.0

# Boundary conditions
bcs = [ 
    p(-L / 2) ~ 0.0,
    p(L / 2) ~ 0.0,
    Dz(p(0)) ~ 0.0,
]
# Space and time domains
domains = [ z ∈ Interval(-L/2, L/2)]

# Neural network
dim = 1 # number of dimensions
# chain = Lux.Chain(Dense(dim, 128, Lux.σ), Dense(128, 32, Lux.σ), Dense(32, 32, Lux.σ), Dense(32, 1))

chain = Lux.Chain(
    Dense(dim, 16, Lux.σ), 
    Dense(16, 16, Lux.σ), 
    Dense(16, 1)
)

# Discretization
dx = L/15
strategy = GridTraining(dx)
discretization = PhysicsInformedNN(chain, strategy)

@named pde_system = PDESystem(eq, bcs, domains, [z], [p(z)])
prob = discretize(pde_system, discretization)

#Optimizer
opt = OptimizationOptimJL.BFGS()

#Callback function
callback = function (pp, l)
    println("Current loss is: $l")
    return false
end

res = Optimization.solve(prob, opt, callback = callback)
phi = discretization.phi

using Plots

xs = [infimum(d.domain):(ddx / 10):supremum(d.domain) for (d,ddx) in zip(domains,dx)][1]
u_predict = [first(phi(x, res.u)) for x in xs]

plot(xs, u_predict, 
    title = "Pressure distribution short bearing", 
    xticks = (-L/2:L/4:L/2, ["-L/2","-L/4","0","L/4","L/2"]))
xlabel!("z")