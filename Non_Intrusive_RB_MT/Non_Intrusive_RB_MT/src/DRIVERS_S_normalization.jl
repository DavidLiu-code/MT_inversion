using LinearAlgebra, Statistics 

function normalize_mat(mat)
    mean_mat=mean(mat)
    std_mat=std(mat)
    norm_mat = (mat .- mean_mat) ./ std_mat
    return norm_mat
end