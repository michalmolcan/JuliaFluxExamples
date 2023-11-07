# This file is used to train a neural network to predict the deflection of a rail

using Flux, Zygote, Random
using Statistics: mean
using Plots, StatsPlots

# Random.seed!(666)
n_train = 64
n_test = 2048
signal_length = 128
weight_range = (20000.0, 75000.0)

signal_parameters = (;
    padding = 0.1,
    noise_level_factor = 0.1, 
    random_shift_factor = 0.1, 
    random_outer_length_factor = 0.2,
    random_truck_cerner_length_factor = 0.2,
    random_axle_spacing_factor = 0.2
)

function plot_data(test_data)
    plot(test_data.features[:,1:5])
    xlabel!("Time [-]")
    ylabel!("Deflection [-]")
    title!("Test signal for car weight $(round(test_data.targets[1])) [-]")
end

function plot_data_heatmap(train_data)
    x = 1:size(train_data.features,1)
    y = 1:size(train_data.features,2)
    heatmap(x,y,train_data.features)
end

function loader(data; batchsize::Int=64)
    x4dim = reshape(data.features, size(data.features,1), 1, :)   # insert trivial channel dim
    y = reshape(data.targets, 1, :)
    Flux.DataLoader((x4dim, y); batchsize, shuffle=true)
end

# obtain data
include("../synthetize_data/synthetize_railroad_car_data.jl")
test_data = create_rail_data(n_samples=n_test,weight_range=weight_range,signal_length=signal_length,signal_parameters=signal_parameters)

# Define the CNN model
lenet = Chain(
    Conv((5,), 1 => 4, relu, pad=2),
    MaxPool((2,)),
    Conv((5,), 4 => 8, relu, pad=2),
    MaxPool((2,)),
    # Conv((5,), 8 => 16, relu, pad=2),
    # MaxPool((2,)),
    Flux.flatten,
    # Dense(448 => 128, relu),
    Dense(256 => 64, relu),
    Dense(64 => 16, relu),
    Dense(16 => 1)
)

# evaluate test data for visualization
ddata = only(loader(test_data; batchsize=length(test_data.targets)))
initial_values = lenet(ddata[1])  # just to force compilation

function loss_and_accuracy(model, data)
    (x, y) = only(loader(data; batchsize=length(data.targets)))  # make one big batch

    ŷ = model(x)
    loss = Flux.mse(ŷ, y)  # did not include softmax in the model
    acc = maximum(abs.(1 .- ŷ ./ y))
    (; loss, acc)  # return a NamedTuple
end

settings = (;
    eta=3e-4,     # learning rate
    lambda=1e-2,  # for weight decay - to reduce complexity of the model [https://towardsdatascience.com/this-thing-called-weight-decay-a7cd4bcfccab]
    batchsize=64,
    epochs=500
)

# Define the optimizer
opt_rule = OptimiserChain(WeightDecay(settings.lambda), Adam(settings.eta))
opt_state = Flux.setup(opt_rule, lenet);

function train_nn(n_train)
    # randomize initial random weights
    rng = Random.seed!(666)
    for p in Flux.params(lenet)
        p .= Flux.glorot_normal(rng,size(p)...)
    end
    
    train_data = create_rail_data(n_samples=n_train,weight_range=weight_range,signal_length=signal_length,signal_parameters=signal_parameters)
    train_log = []
    # Train the model
    for epoch in 1:settings.epochs
        # @time will show a much longer time for the first epoch, due to compilation
        for (x, y) in loader(train_data,batchsize=settings.batchsize)
            grads = Flux.gradient(m -> Flux.mse(m(x), y), lenet)
            Flux.update!(opt_state, lenet, grads[1])
        end

        loss, acc = loss_and_accuracy(lenet, train_data)
        test_loss, test_acc = loss_and_accuracy(lenet, test_data)
        @info "logging:" n_train epoch loss acc test_acc
    end 
    relative_error = ((ddata[2].-lenet(ddata[1]))./ddata[2]*100)'
    (n_train,maximum(abs.(relative_error)))
end

results = train_nn.([(2).^(2:7)...,256:128:1024...])

plot(results)
xlabel!("Number of training samples [-]")
ylabel!("Maximum relative error [%]")

# plot(((lenet(ddata[1]).-ddata[2])./ddata[2]*100)')
# xlabel!("Test measurement no [-]")
# ylabel!("Relative error [%]")
# title!("Relative error for test data")




# bar(((ddata[2].-lenet(ddata[1]))./ddata[2]*100)',label=false)
# xlabel!("Measurement no [-]")
# ylabel!("Relative error [%]")
