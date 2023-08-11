# This file is used to train a neural network to predict the deflection of a rail

using Plots
using Flux, Zygote, ChainPlots

# obtain data
include("synthetize_railroad_car_data.jl")

n_train = 512
weights_train = rand(n_train)*75_000

train_data = (;
    features = reduce(hcat,[create_railroad_car_signal(w,128) for w in weights_train]),
    targets = weights_train
)

n_test = 32
weights_test = rand(n_test)*75_000

test_data = (;
    features = reduce(hcat,[create_railroad_car_signal(w,128) for w in weights_test]),
    targets = weights_test
)

plot(test_data.features[:,1:2])

function loader(data=train_data; batchsize::Int=64)
    x4dim = reshape(data.features, size(train_data.features,1), 1, :)   # insert trivial channel dim
    y = reshape(data.targets, 1, :)
    Flux.DataLoader((x4dim, y); batchsize, shuffle=true) |> gpu
end

x1, y1 = first(loader())
x1
y1

# Define the CNN model
lenet = Chain(
    Conv((5,), 1 => 7, relu, pad=2),
    MaxPool((2,)),
    Conv((5,), 7 => 14, relu, pad=2),
    MaxPool((2,)),
    Conv((5,), 14 => 28, relu, pad=2),
    MaxPool((2,)),
    Flux.flatten,
    Dense(448 => 128, relu),
    Dense(128 => 64, relu),
    Dense(64 => 16, relu),
    Dense(16 => 1)
)

lenet(x1) |> size

# evaluate test data for visualization
ddata = only(loader(test_data; batchsize=length(test_data.targets)))
initial_values = lenet(ddata[1])  # just to force compilation

using Statistics: mean  # standard library

loader(test_data; batchsize=length(test_data.targets))

function loss_and_accuracy(model, data=test_data)
    (x, y) = only(loader(data; batchsize=length(data.targets)))  # make one big batch

    ŷ = model(x)
    loss = Flux.mse(ŷ, y)  # did not include softmax in the model
    acc = maximum(abs.(1 .- ŷ ./ y))
    (; loss, acc)  # return a NamedTuple
end

@show loss_and_accuracy(lenet);  #

# Define the loss function
# loss(x, y) = Flux.mse(lenet(x), y)

settings = (;
    eta=3e-4,     # learning rate
    lambda=1e-2,  # for weight decay - to reduce complexity of the model [https://towardsdatascience.com/this-thing-called-weight-decay-a7cd4bcfccab]
    batchsize=128,
    epochs=100
)
train_log = []

# Define the optimizer
opt_rule = OptimiserChain(WeightDecay(settings.lambda), Adam(settings.eta))
opt_state = Flux.setup(opt_rule, lenet);

# Train the model
for epoch in 1:settings.epochs
    # @time will show a much longer time for the first epoch, due to compilation
    @time for (x, y) in loader(batchsize=settings.batchsize)
        grads = Flux.gradient(m -> Flux.mse(m(x), y), lenet)
        Flux.update!(opt_state, lenet, grads[1])
    end

    # Logging & saving, but not on every epoch
    if epoch % 2 == 1
        loss, acc = loss_and_accuracy(lenet, train_data)
        test_loss, test_acc = loss_and_accuracy(lenet)
        @info "logging:" epoch loss acc test_acc

        nt = (; epoch, loss, acc, test_loss, test_acc)
        push!(train_log, nt)
    end

    plt = plot(ddata[2]', label="True values")
    plot!(plt, lenet(ddata[1])', linestyle=:dash, label="Predicted values")
    plot!(plt, title="Epoch $epoch")
    xlabel!(plt, "Measurement no [-]")
    ylabel!(plt, "Weight [kg]")
    display(plt)

    # sleep(0.2)
end

lenet(ddata[1])
ddata[2]

plot(((lenet(ddata[1]).-ddata[2])./ddata[2]*100)')
xlabel!("Test measurement no [-]")
ylabel!("Relative error [%]")