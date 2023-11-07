function deflection_signal(l,weight)
    u = zeros(l)

    g = 9.81
    F = weight/8*g

    start_index = rand(2:3)
    short_gap = 2
    long_gap = 7

    u[start_index] = F
    u[start_index + short_gap] = F
    u[start_index + short_gap + long_gap] = F
    u[start_index + short_gap + long_gap + short_gap] = F
    
    u
end

function create_data(no_measurements)
    ms = rand(no_measurements)*75_000
    signals = deflection_signal.(16,ms)
    signals2 = reduce(hcat,signals)
    # signals3 = reshape(signals2, (size(signals2,1),1,size(signals2,2)))

    data = (;
        targets = ms,
        features = signals2
    )
end