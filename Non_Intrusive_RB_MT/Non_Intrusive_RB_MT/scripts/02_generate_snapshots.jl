using LinearAlgebra
include("../src/DRIVERS_MT.jl")
#Here, let us take the Matrix M and calculate the snapshots S using our forward operator

#let us assume we have 5 frequencies per decade
f=10 .^range(-3,3,30)

####### Generating Snapshot Matrix ##########
function build_S(M,f)
    S=zeros(2*length(f),length(M[1,:]))
    for i in 1:length(M[1,:])
        S[:,i]=MT_mod(d,10 .^M[:,i],f)
    end
    return S
end

S_random=build_S(M,f)
S_smooth=build_S(M_smooth,f)

#heatmap(S[1:30,:])


