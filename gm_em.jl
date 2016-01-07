using Distributions
using Clustering

#= my own Expectation maximization algorithm to fit a cloud of points with a Gaussian Mixture =#

function gm_em(x,K,nIter)

    r=kmeans(x,K)
    z=zeros(Float64,K,nVect)
    weightOfClass=zeros(Float64,K)
    w=zeros(Float64,K)
    logZ=zeros(Float64,K,nVect)

    for iVect in 1:nVect
        z[r.assignments[iVect],iVect]=1
        weightOfClass[r.assignments[iVect]]+=1
    end

    totalWeight = sum(weightOfClass)
    gaussComp=Array(FullNormal,K)

    for k in 1:K
        zT=reshape(z[k,:],nVect)
        w[k]=weightOfClass[k]/totalWeight
        gaussComp[k]=fit_mle(MvNormal,x,zT/totalWeight)
    end

    gm=MixtureModel(gaussComp,w)
    #= vérifier si à ce stade c'est représentatif =#




    for iIter = 1:nIter
        # Expectation
        for iVect in 1:nVect
            for k in 1:K
                logZ[k,iVect]=log(gm.prior.p[k])+logpdf(gm.components[k],x[:,iVect])
            end
            logZ[:,iVect]=logZ[:,iVect]-maximum(logZ[:,iVect])
            z[:,iVect]=exp(logZ[:,iVect])
            z[:,iVect]=z[:,iVect]/sum(z[:,iVect]) # degré d'appartenance à chacune des composantes
        end

        # Maximization
        for k in 1:K
            zT=reshape(z[k,:],nVect)
            w[k]=sum(z[k,:])
            gaussComp[k]=fit_mle(MvNormal,x,zT)
        end
        w=w/sum(w)
        gm=MixtureModel(gaussComp,w)
    end
    return gm
end


#= for testing 
nVect=10^5
x=rand(2,nVect)
K = 50
nIter=10
=#


#=
f = (x,y)->pdf(gm,[x,y])
p=Plots.Contour(f, (-2,2), (-2,2))
save("myfile.pdf",p)
=#
