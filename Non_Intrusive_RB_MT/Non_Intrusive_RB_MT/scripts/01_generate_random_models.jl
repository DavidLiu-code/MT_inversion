using Distributions,Turing,Plots,Surrogates
#Let us define the prior distribution, We can begin with completely random models

#Depth vector
n =90; d=collect(range(0,stop=4400,length=n))
rhomin=1
rhomax=2000
n_realizations = 10000
#Prior def
prior_rand=filldist(Uniform(log10(rhomin),log10(rhomax)),n)

######## Generating Matrix M - Fully Random ############
function build_M(n_realizations,prior_rand)
    M=zeros(length(prior_rand.v),n_realizations)
    for i in 1:n_realizations
        M[:,i]=rand(prior_rand)
    end
    return M
end

M=build_M(n_realizations,prior_rand)

####### Generating M smooth - Smoothness constraint #########
#For this option we will use the same prior, but now we will enforce 
#a smoothness constraint and generate
din = unique(sort(vcat(1,n,Int.(ceil.(sort(Surrogates.sample(10, 
                    1, 
                    n, 
                    SobolSample())))))))

prior_surrogate=filldist(Uniform(log10(rhomin),log10(rhomax)),length(din))

M_surrogate=build_M(n_realizations,prior_surrogate)

function krig_M(M_surrogate,n_realizations)
    M_smooth=zeros(n,n_realizations)
    for i in 1:n_realizations
        krig_rho=Kriging(d[din], M_surrogate[:,i],d[1],d[end])
        M_smooth[:,i]=krig_rho.(d)
    end
    return M_smooth
end

M_smooth=krig_M(M_surrogate,n_realizations)