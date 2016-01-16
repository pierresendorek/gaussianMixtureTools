# todo : gaussianMixtureWrapper containing precomputed values of log

using Distributions
using Convex
using SCS


include("randomDrawGaussianMixture.jl")

# use typing for gaussians
type GaussComp
    logW::Float64
    normal::MvNormal
end


type Grid
    y::Array{Float64,2} # y[i,d] is the d'th dimension of the i'th vector
end

function newGrid(y::Array{Float64,2})
    @assert size(y)[1]>=2
    # creates a Grid in the right format
    D=size(y)[2]
    for d in 1:D
        y[:,d]=sort(y[:,d])        
    end
    return Grid(y)
end


function getBoxBoundaries(multiIndex::Array{Int64,1},grid::Grid)
    @assert length(multiIndex)==size(y)[2]
    D=size(y)[2]
    zL=zeros(Float64,D)
    zU=zeros(Float64,D)   
    for d in 1:D
        i=multiIndex[d]
        if i==0
            zL[d]=-Inf
            zU[d]=grid.y[i+1,d]
        elseif i==size(y)[1]
            zL[d]=grid.y[i,d]
            zU[d]=+Inf
        else
            zL[d]=grid.y[i,d]
            zU[d]=grid.y[i+1,d]
        end
    end
    return (zL,zU)
end


function pointToBoxMultiIndex(x::Array{Float64,1},grid::Grid)
    @assert length(x)==size(y)[2]
    D=length(x)
    multiIndex=Array(Int64,D)
    for d in 1:D
        i=searchsortedlast(grid.y[:,d],x[d])
        multiIndex[d]=i
    end
    return multiIndex
end


function evalGaussComp(x::Vector,gc::GaussComp)
    return exp(gc.logW + logpdf(gc.normal,x))
end



# datatype definition
GaussianMixture=Distributions.MixtureModel{Distributions.Multivariate,Distributions.Continuous,Distributions.MvNormal{PDMats.PDMat,Array{Float64,1}}}

function product(gC1::GaussComp,gC2::GaussComp)
    g1=gC1.normal
    g2=gC2.normal
    invCov1 = inv(cov(g1))
    invCov2 = inv(cov(g2))
    Gamma = inv(invCov1 + invCov2)
    M = Gamma*(invCov1 * mean(g1) +  invCov2 * mean(g2))
    logW = 0.5*(log(det(2*pi*Gamma))
                - log(det(2*pi*cov(g1)))
                - log(det(2*pi*cov(g2)))
                + M'*(Gamma\M)
                - mean(g1)'*invCov1*mean(g1)
                - mean(g2)'*invCov2*mean(g2))
    return GaussComp(logW[1]+gC1.logW+gC2.logW,MvNormal(M,Gamma))
end


function productAndNormalize(gm1::GaussianMixture,gm2::GaussianMixture)
    nComp =gm1.prior.K * gm2.prior.K
    gcArray=Array(GaussComp,nComp)
    iComp=1
    maxLogW = -Inf
    for i1 in 1:gm1.prior.K, i2 in 1:gm2.prior.K
        gc1 = GaussComp(log(gm1.prior.p[i1]),gm1.components[i1])
        gc2 = GaussComp(log(gm2.prior.p[i2]),gm2.components[i2])
        newComp=product(gc1,gc2)
        gcArray[iComp]=newComp
        if(maxLogW<newComp.logW)
            maxLogW=newComp.logW
        end 
        iComp+=1
    end
    # normalize by the max (avoids cancelling the weights)
    for iComp in 1:nComp
        gcArray[iComp].logW -= maxLogW
    end

    # normalize by the sum
    s=0.0
    for iComp in 1:nComp
        s+= exp( gcArray[iComp].logW )
    end

    for iComp in 1:nComp
        gcArray[iComp].logW -= s
    end

    # create a GaussianMixture
    normalArray=Array(FullNormal,nComp)
    w = zeros(Float64,nComp)

    for iComp in 1:nComp
        w[iComp]=exp(gcArray[iComp].logW)
        normalArray[iComp] = gcArray[iComp].normal
    end
    return  MixtureModel(normalArray,w)
end


function conditionalProba(gm::GaussianMixture,idxGiven,x,idxNonNegligibleGaussianComp)
    # only the idxGiven indexes of x are taken into account as xQ !
    d=length(gm.components[1])
    nTotalGauss= length(gm.components)
    nNonNegligibleGauss = length(idxNonNegligibleGaussianComp)
    idxNotGiven=setdiff(collect(1:d),idxGiven)
    Id=eye(d,d)
    Q = Id[idxGiven,:]
    P = Id[idxNotGiven,:]
    xQ=Q*x

    logCoeffGauss = fill(-Inf,nNonNegligibleGauss)
    gaussCompArray=Array(FullNormal,nNonNegligibleGauss)

    count=1
    for iGauss in idxNonNegligibleGaussianComp
        nu = gm.prior.p[iGauss]
        g = gm.components[iGauss]
        mu = mean(g)
        C = cov(g)
        invC=inv(C)
        M = (P*invC*P')\((P*invC*P')*P*mu - P*invC*Q'*Q*(x-mu))
        Sigma = inv(P*invC*P')

        logKappa = -0.5*(- M'*P*invC*P'*M + (P*mu)'*P*invC*P'*(P*mu) + (Q*(x-mu))'*Q*invC*Q'*Q*(x-mu) - 2*(P*mu)'*P*invC*Q'*Q*(x-mu))
        logCoeffGauss[count] = log(nu) -0.5*log(det(2*pi*C)) + logKappa[1] + 0.5*log(det(2*pi*Sigma))
        gaussCompArray[count] = MvNormal(M,Sigma)
        count+=1
    end

    logCoeffGauss=logCoeffGauss-maximum(logCoeffGauss)
    wGauss = exp(logCoeffGauss)
    wGauss = wGauss/sum(wGauss)

    return MixtureModel(gaussCompArray,wGauss)
end



function invSqrtOfGMCovArray(gm::GaussianMixture )
    invSqrtCovArray=Array(Array{Float64,2},length(gm.prior.p))
    for i in 1:length(gm.prior.p)
        DR=eig(gm.components[i].Σ.mat)
        D=DR[1]
        R=DR[2]
        invSqrtLambda = diagm(D)
        invSqrtCovArray[i] = R*invSqrtLambda*R'
    end
    return invSqrtCovArray
end


#=
function findBoxNegligibleComp(gm::GaussianMixture,multiIndex::Array{Int64,1},invSqrtCovArray)
    d=length(gm.components[1].μ)
    # /!\ remove the following and replace by appropriate values
    xU=2*ones(d)
    xL=ones(d)
    # end remove

    K = 1E3

    nComp=length(gm.prior.p)
    logU

    for iComp in 1:nComp
        logKhi=-0.5*logdet(2*pi*gm.components[iComp].Σ.mat) + log(gm.prior.p[iComp])
        mu = gm.components[iComp].μ
        logU[iComp]=findMaxNegQuadraticFormOnBox(invSqrtCovArray[iComp],mu,logKhi,xL,xU)
        logL[iComp]=findMinNegQuadraticFormOnBox(invSqrtCovArray[iComp],mu,logKhi,xL,xU)
    end
    maxLogU
    logU-=maxLogU
    logL-=maxLogU

    U=exp(logU)
    L=exp(logL)

    idxSortU=sortperm(U)
    idxSortL=sortperm(L)

    beta = nComp
    s=0
    #while(s)



end
=#

function findMaxNegQuadraticFormOnBox(sqrtQ,mu,logKhi,xL,xU)
    # assumes we maximize -0.5*sumsquare(sqrtQ*(x-mu)) + logKhi
    x=Variable(length(mu));
    problem = minimize(sumsquares(sqrtQ*(x-mu)), [x<=xU,x>=xL])
    solve!(problem,SCSSolver(verbose=0))
    solution = problem.optval
    return -0.5*solution + logKhi
end


function findLowerBoundNegQuadraticFormOnBox(eigQ0,mu,logKhi,xL,xU)
    # eigQ0 is the highest eigenvalue of sqrtQ'*sqrtQ
    # assumes we want to lower bound -0.5*sumsquare(sqrtQ*(x-mu)) + logKhi
    # this function is much faster than findMinNegQuadraticFormOnBox for high dimensions
    m=(xL+xU)/2
    r=norm((xU-xL)/2)
    return -eigQ0*norm(m - r*(m-mu)/norm(m-mu))^2+logKhi
end


#===========================================
Less useful
===========================================#

function getBoxVertex(i,xL::Array{Float64,1},xU::Array{Float64,1})
    # taxes indexes from 1 to 2^d
    @assert length(xL)==length(xU)
    d=length(xL)
    xNew=zeros(Float64,d)
    s=bin(i-1,d)
    for c=1:d
        if s[c]=='0'
            xNew[c]=xL[c]
        else
            xNew[c]=xU[c]
        end
    end
    return xNew
end

function findMinNegQuadraticFormOnBox(sqrtQ,mu,logKhi,xL,xU)
    # assumes we minimize -0.5*sumsquare(sqrtQ*(x-mu)) + logKhi
    # the min is found on a vertex
    # to optimize this function : maintain a list of the already evaluated points for the gaussian i
    d=length(mu)
    N=2^d
    vMin=Inf
    for i in 1:N
        x=getBoxVertex(i,xL,xU)
        vMin=minimum([-0.5*sumsquares(sqrtQ*(x-mu)),vMin])
    end
    return vMin+logKhi
end


#=================================
Testing
==================================#


function testConditionalProba()
    nComp=10
    gm = randomDrawGaussianMixture(nComp)
    idxGiven = [2]
    x=randn(2)
    idxNonNegligibleGaussianComp = collect(1:nComp)
    cgm = conditionalProba(gm,idxGiven,x,idxNonNegligibleGaussianComp)

    X=randn(10)
    Ycalc=zeros(Float64,10)
    Ydirect=zeros(Float64,10)
    for i in 1:10
        Ycalc[i]=logpdf(cgm,[X[i]])
        Ydirect[i]=logpdf(gm,[X[i],x[2]])
    end

    Ycalc = Ycalc - maximum(Ycalc)
    Ydirect = Ydirect - maximum(Ydirect)

    for i in 1:10
        println(Ycalc[i])
        println(Ydirect[i])
        println("----------")

    end

end



function testGcProduct()
    #gc = GaussComp(-1,MvNormal(zeros(2),eye(2)))

    dMean = MvNormal(zeros(2),10*eye(2,2)) # distribution of the means
    dCov=Wishart(2.0, eye(2))
    # dWeight = Dirichlet(nComp,1)
    logW1 = log(rand())
    logW2 = log(rand())

    normal1 = MvNormal(rand(dMean),rand(dCov))
    normal2 = MvNormal(rand(dMean),rand(dCov))

    gc1=GaussComp(logW1,normal1)
    gc2=GaussComp(logW2,normal2)

    x = randn(2)*0.01

    resProduct = evalGaussComp(x, product(gc1,gc2))
    resDirect = evalGaussComp(x,gc1)*evalGaussComp(x,gc2)

    println(resProduct)
    println(resDirect)

end



