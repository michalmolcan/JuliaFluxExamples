using Unitful,Random

Random.seed!(1234)

include("rail_cars_and_locomotives_data.jl")

function railroad_car_axles_location(
    outer_length,
    truck_cerner_length,
    axle_spacing
)

    front_signal_padding = (outer_length - truck_cerner_length-axle_spacing)/2
    back_signal_padding = front_signal_padding + truck_cerner_length
    
    [
        front_signal_padding,
        front_signal_padding + axle_spacing,
        back_signal_padding,
        back_signal_padding + axle_spacing
    ]
end

gaussian_impulse(x,a) = sqrt(a/pi)*exp(-a*x^2)
normalised_gaussian_impulse(x,a) = gaussian_impulse.(x,a)./maximum(gaussian_impulse.(x,a))
normalised_gaussian_impulse(x,a,shift) = normalised_gaussian_impulse(x.-shift,a)

rand_variation(rand_level) = (1+(2*rand()-1)*rand_level)

function create_railroad_car_signal(weight, signal_length = 32; 
    padding = 0.2,
    noise_level_factor = 0.1, 
    random_shift_factor = 0.1, 
    random_outer_length_factor = 0.0,
    random_truck_cerner_length_factor = 0.0,
    random_axle_spacing_factor = 0.0)
    
    outer_length = car_dimensions.outer_length*rand_variation(random_outer_length_factor)
    truck_cerner_length = car_dimensions.truck_cerner_length*rand_variation(random_truck_cerner_length_factor)
    axle_spacing = car_dimensions.axle_spacing*rand_variation(random_axle_spacing_factor)


    rand_location_shift = rand(1).*random_shift_factor*outer_length
    axles_locations = railroad_car_axles_location(outer_length,truck_cerner_length,axle_spacing).+rand_location_shift
    axles_locations_front_padded = axles_locations .+ padding*outer_length
    axle_indices = round.(Int64,axles_locations_front_padded./(1+2*padding)/outer_length .* signal_length)

    x = zeros(signal_length)
    a = 16/signal_length

    for a_i in axle_indices
        x .+= weight/4*normalised_gaussian_impulse(0:signal_length-1,a,a_i)
    end
    noise = rand(length(x)).*noise_level_factor*maximum(x)
    x.+ noise
end

function assemble_train(number_cars::Int64, car_types::Vector{String} = String[])
    full_train = Rail_vehicle[]
    push!(full_train,locomotives_catalogue[1])

    available_cars = filter(rail_car -> rail_car isa Rail_car_4_axles, rail_cars_4_axles_catalogue)
    if !isempty(car_types)
        available_cars = filter(rail_car -> rail_car.name in car_types, available_cars)
    end
    for _ in 1:number_cars
        car = rand(available_cars)
        push!(full_train,car)
    end
    full_train
end

get_weight(rail_vehicle::Locomotive) = rail_vehicle.weight
function get_weight(rail_vehicle::Rail_car)
    if rail_vehicle.empty_weight isa Tuple
        rail_wehicle_empty_weigth = minimum(rail_vehicle.empty_weight)+rand()*(maximum(rail_vehicle.empty_weight)-minimum(rail_vehicle.empty_weight))
    else
        rail_wehicle_empty_weigth = rail_vehicle.empty_weight
    end
    round(typeof(1.0u"kg"),rail_wehicle_empty_weigth)
end

function create_full_rail_signal(loads::Vector{Float64}, traint_velocity = 10.0u"km/hr", sampling_frequency = 100u"Hz")
    full_train = assemble_train(length(loads))
    # add load weights
    weights = get_weight.(full_train)
    total_length = uconvert(u"m",sum(rail_vehicle -> rail_vehicle.outer_length, full_train))
    total_time = uconvert(u"s",total_length/traint_velocity)
    n_samples = uconvert(NoUnits, total_time*sampling_frequency)
    total_time,n_samples
end

create_full_rail_signal([50000.0,40000.0,20000.0,1000.0])