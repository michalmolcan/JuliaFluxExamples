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

function create_railroad_car_signal(weight, signal_length = 32,noise_level = 0.1, random_shift = 0.1, padding = 0.2)
    rand_location_shift = rand(1).*random_shift*car_dimensions.outer_length
    axles_locations = railroad_car_axles_location(car_dimensions...).+rand_location_shift
    axles_locations_front_padded = axles_locations .+ padding*car_dimensions.outer_length
    axle_indices = round.(Int64,axles_locations_front_padded./(1+2*padding)/car_dimensions.outer_length .* signal_length)

    x = zeros(signal_length)
    a = 16/signal_length

    for a_i in axle_indices
        x .+= weight/4*normalised_gaussian_impulse(0:signal_length-1,a,a_i)
    end
    noise = rand(length(x)).*noise_level*maximum(x)
    x.+ noise
end

