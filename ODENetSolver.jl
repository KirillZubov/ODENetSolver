using ForwardDiff, Plots

function generate_data(low, high, N)
    x = range(low, stop = high, length = N) |> collect
    return x
end

function init_params(hl_width)
    P = Array{Any}(undef, 3)
    P[1] = 0.0001*randn(hl_width,1)
    P[2] = zeros(hl_width,1)
    P[3] = 0.0001*randn(1,hl_width)
    return P
end

sigm(x) =  1 ./ (1 .+ exp.(.-x))

function predict(w, b, v, x)
    h = sigm(w*x .+ b)
    return v*h
end

phi(w, b, v,x) = u0 .+ x*predict(w, b, v, x)
dydx(w, b, v, x) = ForwardDiff.derivative(x -> phi(w, b, v, x)[1], x) - f(x)
loss(w, b, v, x) = sum(abs2, dydx(w, b, v, x))

loss∇w(w, b, v, x) = ForwardDiff.gradient(w -> loss(w, b, v, x), w)
loss∇b(w, b, v, x) = ForwardDiff.gradient(b -> loss(w, b, v, x), b)
loss∇v(w, b, v, x) = ForwardDiff.gradient(v -> loss(w, b, v, x), v)

function test(P,x)
    sumloss = numloss = 0
    for i in x
        sumloss += loss(P[1],P[2],P[3],i)
        numloss += 1
    end
    return sumloss/numloss
end

function train(P, x, epochCount, lr)
    for i = 1:epochCount
        for j in x
            P[1] -= lr * loss∇w(P[1], P[2], P[3], j)
            P[2] -= lr * loss∇b(P[1], P[2], P[3], j)
            P[3] -= lr * loss∇v(P[1], P[2], P[3], j)
        end
        if test(P,x) < 10^-8
            break
        end
    end
    return P
end

#ODE solver using neural network approximation
function ODENetSolver(f, N, u0, x0, xn, epochCount, lr)
    x =  generate_data(x0,xn,N)
    P = init_params(weightCount)
    P = train(P, x, epochCount, lr)
    u = [phi(P[1],P[2],P[3],i)[1] for i in x]
return (x,u)
end

N = 20
weightCount = 5
u0 = 0
x0 = 0
xn = 1
epochCount = 1000
lr = 0.1
f(x) = cos(2pi*x)

solve = ODENetSolver(f, N, u0, x0, xn, epochCount, lr)

f2(x) =  sin(2pi*x) / (2*pi)
x = solve[1]
analyticSolve = (x, f2.(x))
plot(solve, label="y")
plot!(analyticSolve, label ="true y", fmt = :png)
xlabel!("Solution for y'= cos(2pi*x)")
