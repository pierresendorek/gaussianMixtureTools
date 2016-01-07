using Distributions

#=
does not draw a sample according to a gaussian mixture
but draws the parameters of a gaussian mixture
=> useful for testing
=#

function randomDrawGaussianMixture(nComp::Int64)
    dMean = MvNormal(zeros(2),10^2*eye(2,2)) # distribution of the means
    dCov=Wishart(2.0, eye(2))
    dWeight = Dirichlet(nComp,1)
    w=rand(dWeight)
    normalArray=Array(FullNormal,nComp)
    
    for i in 1:nComp    
        normalArray[i]=MvNormal(rand(dMean),rand(dCov))
    end
    
   return MixtureModel(normalArray,w)
end


function testRandomDrawGaussianMixture()
    return randomDrawGaussianMixture(5)
end

