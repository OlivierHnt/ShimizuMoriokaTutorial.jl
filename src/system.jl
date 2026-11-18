# Shimizu-Morioka system

function f(u, params)
    a, b = params
    x, y, z = u
    return [y
            x - a*y - x*z
            -b*z + x^exact(2)]
end

function Df(u, params)
    a, b = params
    x, y, z = u
    return [zero(x)        one(y)     zero(z)
            exact(1)-z  -a*one(y)   -x
            exact(2)*x    zero(y)   -b*one(z)]
end
