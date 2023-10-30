using Unitful

abstract type Rail_vehicle end

abstract type Rail_car <: Rail_vehicle end

mutable struct Rail_car_4_axles <: Rail_car
    name::String
    empty_weight::Union{NTuple{2,typeof(1.0u"kg")},typeof(1.0u"kg")}
    outer_length::typeof(1.0u"mm")
    outer_wheel_distance::typeof(1.0u"mm")
    wheelbase::typeof(1.0u"mm")
    rotational_axle_distance::typeof(1.0u"mm")
end

mutable struct Rail_car_2_axles <: Rail_car
    name::String
    empty_weight::Union{NTuple{2,typeof(1.0u"kg")},typeof(1.0u"kg")}
    outer_length::typeof(1.0u"mm")
    outer_wheel_distance::typeof(1.0u"mm")
end

Rail_car(name, empty_weight, outer_length, outer_wheel_distance, wheelbase, rotational_axle_distance) = Rail_car_4_axles(name, empty_weight, outer_length, outer_wheel_distance, wheelbase, rotational_axle_distance)
Rail_car(name, empty_weight, outer_length, outer_wheel_distance) = Rail_car_2_axles(name, empty_weight, outer_length, outer_wheel_distance)




abstract type Locomotive <: Rail_vehicle end

struct Locomotive_4_axles <: Locomotive
    name::String
    weight::Union{NTuple{2,typeof(1.0u"kg")},typeof(1.0u"kg")}
    outer_length::typeof(1.0u"mm")
    outer_wheel_distance::typeof(1.0u"mm")
    wheelbase::typeof(1.0u"mm")
    rotational_axle_distance::typeof(1.0u"mm")
end

const rail_cars_catalogue = [
    Rail_car("Eanos II",    (21800.0u"kg",22000.0u"kg"),    15740u"mm", 15740u"mm", 15740u"mm", 15740u"mm"),
    Rail_car("Eas 51,54",   (21100.0u"kg",23600.0u"kg"),    14040u"mm", 10800u"mm", 1800u"mm",  9000u"mm"),
    Rail_car("Eas 11",      (21000.0u"kg",24000.0u"kg"),    14040u"mm", 11000u"mm", 2000u"mm",  9000u"mm"),
    Rail_car("Eas - u",     (21000.0u"kg",23500.0u"kg"),    14040u"mm", 10800u"mm", 1800u"mm",  9000u"mm"),
    Rail_car("Eamnoss",     20850.0u"kg",                   12766u"mm", 9526u"mm",  1800u"mm",  7726u"mm"),
    Rail_car("Es 11,12,14", (12000.0u"kg",13700.0u"kg"),    10000u"mm", 6000u"mm"),
    Rail_car("Kns 13",      (14000.0u"kg",15000.0u"kg"),    13860u"mm", 9000u"mm"),
    Rail_car("Knps",        (12800.0u"kg",14400.0u"kg"),    13860u"mm", 9000u"mm"),
    Rail_car("Res 51",      (22700.0u"kg",24500.0u"kg"),    19900u"mm", 16660u"mm", 1800u"mm",  14860u"mm"),
    Rail_car("Res 54",      (25600.0u"kg",26500.0u"kg"),    20040u"mm", 16400u"mm", 1800u"mm",  14600u"mm"),
    Rail_car("Res 11",      (23100.0u"kg",25500.0u"kg"),    20040u"mm", 16600u"mm", 2000u"mm",  14600u"mm"),
    Rail_car("Res 67",      (23200.0u"kg",25700.0u"kg"),    19900u"mm", 16660u"mm", 1800u"mm",  14860u"mm"),
    Rail_car("Rils 51",     (23000.0u"kg",25000.0u"kg"),    19900u"mm", 16660u"mm", 1800u"mm",  14860u"mm"),
    Rail_car("Rils-y 51",   (24600.0u"kg",25500.0u"kg"),    19900u"mm", 16660u"mm", 1800u"mm",  14860u"mm"),
    Rail_car("Rs",          (21100.0u"kg",23600.0u"kg"),    19900u"mm", 16660u"mm", 1800u"mm",  14860u"mm"),
    Rail_car("Rbns",        (29400.0u"kg",29950.0u"kg"),    26350u"mm", 21850u"mm", 1800u"mm",  20050u"mm"),
    Rail_car("Scmms 10",    (23700.0u"kg",25600.0u"kg"),    15240u"mm", 11400u"mm", 2000u"mm",  9400u"mm"),
    Rail_car("Shimmns",     (21000.0u"kg",22000.0u"kg"),    12040u"mm", 8800u"mm",  1800u"mm",  7000u"mm"),
    Rail_car("Shimmns II",  (22000.0u"kg",23100.0u"kg"),    12400u"mm", 8300u"mm",  1800u"mm",  6500u"mm"),
    Rail_car("Smmps 11",    (21500.0u"kg",24200.0u"kg"),    15240u"mm", 11400u"mm", 2000u"mm",  9400u"mm"),
    Rail_car("Smmps 54",    (21800.0u"kg",24200.0u"kg"),    15240u"mm", 11200u"mm", 1800u"mm",  9400u"mm")
]

const rail_cars_4_axles_catalogue = filter(rail_car -> rail_car isa Rail_car_4_axles, rail_cars_catalogue)

const locomotives_catalogue = [
    Locomotive_4_axles("744.1", 80000.0u"kg",   16400u"mm", 10900u"mm", 2400u"mm",  8500u"mm"),
    Locomotive_4_axles("753.6", 76000.0u"kg",   16660u"mm", 11400u"mm", 2400u"mm",  9000u"mm")
]