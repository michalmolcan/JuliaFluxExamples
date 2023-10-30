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

const car_dimensions = (;
    outer_length = 9.8,
    truck_cerner_length = 5.5,
    axle_spacing = 1.8
)

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

function create_rail_data(;n_samples,weight_range,signal_length,signal_parameters)
    car_weights = (maximum(weight_range)-minimum(weight_range))*rand(n_samples).+minimum(weight_range)

    features = reduce(hcat,[create_railroad_car_signal(w,signal_length;signal_parameters...) for w in car_weights])
    targets = car_weights
    
    (;features,targets)
end

