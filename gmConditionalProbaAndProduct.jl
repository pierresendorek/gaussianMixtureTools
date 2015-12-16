using Distributions

include("randomDrawGaussianMixture.jl")

# use typing for gaussians
type GaussComp
    logW::Float64
    normal::MvNormal
end

function evalGaussComp(x::Vector,gc::GaussComp)
    return exp(gc.logW + logpdf(gc.normal,x))
end




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



