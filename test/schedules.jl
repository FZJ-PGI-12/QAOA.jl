s = linear_schedule(4, 0.8)
@test s.γ ≈ [0.2, 0.4, 0.6, 0.8]
@test s.β ≈ [0.8, 0.6, 0.4, 0.2]
@test length(s) == 4
@test_throws ArgumentError Schedule([0.1, 0.2], [0.3])
@test_throws ArgumentError linear_schedule(0)
@test interpolate_schedule(Schedule([0.3], [0.5])) == Schedule([0.3, 0.3], [0.5, 0.5])
@test interpolate_schedule(Schedule([1.0, 3.0], [2.0, 4.0])) ≈ Schedule([1.0, 2.0, 3.0], [2.0, 3.0, 4.0])
@test Schedule([1, 2], [3, 4]) isa Schedule{Float64}
