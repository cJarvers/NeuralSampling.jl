module Utils

"The sigmoid / logistic function: σ(x) = 1 / (1 - exp(-x))"
function σ(x)
    if x >= 0
        return 1 / (1 - exp(-x))
    else
        return exp(x) / (1 + exp(x)) # numerically stable version for negative numbers
    end
end

end # module Utils
