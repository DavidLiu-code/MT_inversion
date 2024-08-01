using LinearAlgebra

Ur,SVr,Vr = svd(Snorm_random)
Us,SVs,Vs = svd(Snorm_smooth)

heatmap(Us,
title="Principal modes random models",
ylabel="Cell",
xlabel="n",
colorbar_title="Log of resistivity")

plot(SVr)