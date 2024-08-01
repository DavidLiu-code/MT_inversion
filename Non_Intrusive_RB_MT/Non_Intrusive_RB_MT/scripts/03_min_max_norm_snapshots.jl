include("../src/DRIVERS_S_normalization.jl")
# Separate apparent resistivity and phase S Matrix
res_rand = S_random[1:30,:]
pha_rand = S_random[31:end,:]
res_smooth = S_smooth[1:30,:]
pha_smooth = S_smooth[31:end,:]

###### normalization ########

res_rand_norm = normalize_mat(res_rand)
pha_rand_norm = normalize_mat(pha_rand)

Snorm_random = vcat(res_rand_norm,pha_rand_norm)

res_smooth_norm = normalize_mat(res_smooth)
pha_smooth_norm = normalize_mat(pha_smooth)

Snorm_smooth = vcat(res_smooth_norm,pha_smooth_norm)