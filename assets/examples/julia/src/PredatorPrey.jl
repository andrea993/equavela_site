module PredatorPrey

export lotka_volterra, rk4

"""Right-hand side of the Lotka–Volterra equations for prey `u[1]` and predators `u[2]`."""
lotka_volterra(u, p) = [p.α * u[1] - p.β * u[1] * u[2],
                        p.δ * u[1] * u[2] - p.γ * u[2]]

"""Classical Runge–Kutta method: `n` steps of `du/dt = f(u)` from 0 to `T`.
Returns one row per step: the time, then the state."""
function rk4(f, u0, T, n)
    h = T / n
    u = float.(u0)
    out = zeros(n + 1, 1 + length(u))
    out[1, :] = [0; u]
    for i in 1:n
        k1 = f(u)
        k2 = f(u .+ h / 2 .* k1)
        k3 = f(u .+ h / 2 .* k2)
        k4 = f(u .+ h .* k3)
        u = u .+ h / 6 .* (k1 .+ 2k2 .+ 2k3 .+ k4)
        out[i + 1, :] = [i * h; u]
    end
    out
end

end
