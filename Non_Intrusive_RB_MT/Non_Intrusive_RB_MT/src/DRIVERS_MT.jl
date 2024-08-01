#MT 1D analytical forward modelling using Wait's recursion.
#The input is a depth profile, a resistivity profile, and the frequencies to run the modelling
function MT_mod(depth,rho,fa)
    wa = fa .* 2 * π
    f = length(wa)
    cond = 1.0 ./ rho
    mu = 4 * π * 1.0e-7
    nlayers = length(depth)
    qn = sqrt.((1im * mu * cond) .* wa')
    dz= abs.(diff(depth))
    cn = zeros(Complex, nlayers, f)
    cn[nlayers, :] .= 1.0 ./ qn[nlayers, :]
    for k in eachindex(wa), j in nlayers-1:-1:1
        cn1 = 1 / qn[j, k]
        cn2 = qn[j, k] * cn[j+1, k] + tanh(qn[j, k] * dz[j])
        cn3 = 1 + qn[j, k] * cn[j+1, k] * tanh(qn[j, k] * dz[j])
        cn[j, k] = cn1 * cn2 / cn3
    end
    z = 1im * wa .* mu .* cn[1, :]
    phi = angle.(z)#atan.(imag.(z)./real.(z))
    imp=abs.(z).^2
    arho = 1 ./ (wa .* mu) .*imp
    return vcat(arho,phi)
end