# todo : gaussianMixtureWrapper containing precomputed values of log
# todo : test if bounds 1 and 2 ok separately

using Distributions
#using Convex
#using SCS

include("randomDrawGaussianMixture.jl")



# datatype definition
GaussianMixture=Distributions.MixtureModel{Distributions.Multivariate,Distributions.Continuous,Distributions.MvNormal{PDMats.PDMat,Array{Float64,1}}}


# use typing for gaussians
type GaussComp
    logW::Float64
    normal::MvNormal
end


type Grid
    # one Grid is associated to only one Gaussian Mixture
    y::Array{Array{Float64,1},1} # y[d][i] is the d'th dimension of the i'th vector
    multiIndexToNonNegligibleComp::Dict{Array{Int64,1},Set{Int64}} # for each multiIndex
end


type GaussianMixtureAuxiliary
    gm::GaussianMixture
    grid::Grid
    invSqrtCovArray::Array{Array{Float64,2},1}
    eigLambdaSmallArray::Array{Float64,1} # smallest eigenvalue of the inverseCovariance for each component
    eigLambdaBigArray::Array{Float64,1} # biggest eigenvalue of the inverseCovariance for each component
    logDet2piCov::Array{Float64,1}
    idxGiven::Array{Int64,1}
end



function newGrid(points::Array{Float64,2},idxGiven::Array{Int64,1})
    @assert size(points)[1]>=2
    # creates a Grid in the right format
    # points[i,d] d'th dimension of the i'th point
    # todo : use the gm structure to derive these bounds instead (more accurate)
    D=size(points)[2]
    y=Array(Array{Float64,1},D)
    for d in 1:D
        y[d]=Array(Float64,1)
        points[:,d]=sort(points[:,d])
        if in(d,idxGiven)
            y[d]=points[:,d]
        else
            y[d]=[points[1,d],points[end,d]]
        end
        # creates extended boundaries so as to hardly land on an infinite-sized box
        minD=points[1,d]
        maxD=points[end,d]
        deltaD =  maxD-minD

        y[d][1]=minD-1*deltaD
        y[d][end]=maxD+1*deltaD
    end

    return Grid(y,Dict{Array{Int64,1},Array{Int64,1}}())
end

function marginalProbability(idxNonMarginalized::Array{Int64,1},gm::GaussianMixture)
    # yields another gaussian mixture
    D=length(mean(gm.components[1]))
    nComp=length(gm.prior.p)
    Id=eye(D,D)
    #idxMarginalized=setdiff(collect(1:D),idxNonMarginalized)
    #Q = Id[idxMarginalized,:]
    P = Id[idxNonMarginalized,:]
    
    normalArray = Array(FullNormal,nComp)
    w = deepcopy(gm.prior.p)
    for iComp in 1:nComp
        Pmu = P*mean(gm.components[iComp])
        PCovPt = P*cov(gm.components[iComp])*P'
        normalArray[iComp]=MvNormal(Pmu,PCovPt)
    end
    return MixtureModel(normalArray,w)
end

function invSqrtOfGMCovArrayAndEig(gm::GaussianMixture )
    invSqrtCovArray=Array(Array{Float64,2},length(gm.prior.p))
    eigLambdaSmallArray=Array(Float64,length(gm.prior.p))
    eigLambdaBigArray=Array(Float64,length(gm.prior.p))
    for i in 1:length(gm.prior.p)
        DR=eig(cov(gm.components[i]))
        D=DR[1]
        R=DR[2]
        invEig=1./D
        eigLambdaSmallArray[i]=minimum(invEig)
        eigLambdaBigArray[i]=maximum(invEig)
        invSqrtLambda = diagm(sqrt(invEig))
        invSqrtCovArray[i] = R*invSqrtLambda*R'
    end
    return invSqrtCovArray,eigLambdaSmallArray,eigLambdaBigArray
end



function sumSq(x)
    s=0.0
    for v in x
        s+=v^2
    end
    return s
end



function GaussianMixtureAuxiliary(gm::GaussianMixture,nPoint::Int64,idxGiven::Array{Int64,1})
    D=length(mean(gm.components[1]))
    points=zeros(Float64,nPoint,D)
    # take quantiles instead
    for iPoint in 1:nPoint
        points[iPoint,:]=rand(gm)
    end
    grid=newGrid(points,idxGiven)
    logDet2piCov=Array(Float64,length(gm.prior.p))
    for iComp in 1:length(gm.prior.p)
        logDet2piCov[iComp]=logdet(2*pi*cov(gm.components[iComp]))
    end
    invSqrtCovArray,eigLambdaSmallArray,eigLambdaBigArray=invSqrtOfGMCovArrayAndEig(gm)
    return GaussianMixtureAuxiliary(gm,grid,invSqrtCovArray,
                                    eigLambdaSmallArray,
                                    eigLambdaBigArray,
                                    logDet2piCov,
                                    idxGiven)
end




function getBoxBoundaries(multiIndex::Array{Int64,1},grid::Grid)
    # multiIndex to points
    @assert length(multiIndex)==length(grid.y)
    D=length(grid.y)
    zL=zeros(Float64,D)
    zU=zeros(Float64,D)
    for d in 1:D
        i=multiIndex[d]
        if i==0
            zL[d]=-Inf
            zU[d]=grid.y[d][1]
        elseif i==length(grid.y[d])
            zL[d]=grid.y[d][i]
            zU[d]=+Inf
        else
            zL[d]=grid.y[d][i]
            zU[d]=grid.y[d][i+1]
        end
    end
    return (zL,zU)
end



function pointToBoxMultiIndex(x::Array{Float64,1},grid::Grid)
    #@assert length(x)==size(y)[2]
    D=length(x)
    multiIndex=Array(Int64,D)
    for d in 1:D
        i=searchsortedlast(grid.y[d],x[d])
        multiIndex[d]=i
    end
    return multiIndex
end


function evalGaussComp(x::Vector,gc::GaussComp)
    return exp(gc.logW + logpdf(gc.normal,x))
end



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


function fusion(gc1::GaussComp,gc2::GaussComp)
    w=[exp(gc1.logW),exp(gc2.logW)]
    s=sum(w)
    w=w/s
    mu=w[1]*mean(gc1.normal)+w[2]*mean(gc2.normal)
    C=w[1]*(cov(gc1.normal)+ (mean(gc1.normal)-mu)*(mean(gc1.normal)-mu)')+w[2]*(cov(gc2.normal)+ (mean(gc2.normal)-mu)*(mean(gc2.normal)-mu)')
    return GaussComp(log(s),MvNormal(mu,C))
end



function sumSqDiff(gc1::GaussComp,gc2::GaussComp)
    #=
    yields the value of the integral of the square of the difference
    between gm and a modified version of gm where the components
    gc1 and gc2 are fused into one gaussian : gc[3]
    todo : optimize because the usage of the function product may be not necessary
    =#
    gc=Array(GaussComp,3)
    gc[1]=gc1
    gc[2]=gc2
    gc[3]=fusion(gc[1],gc[2])
    gcp=Array(GaussComp,3,3)
    for i in 1:3
        for j in i:3
            gcp[i,j]=product(gc[i],gc[j])
        end
    end
    s=0
    for i in 1:3
        s+=exp(gcp[i,i].logW)
    end
    s+=2*exp(gcp[1,2].logW)
    s-=2*exp(gcp[1,3].logW)
    s-=2*exp(gcp[2,3].logW)
    return s
end



function gmReduction(gm::GaussianMixture,nCompMax::Int64)
    #= approximation of gm by a GaussianMixture with at most nCompMax components =#
    totalLossBound=0
    nComp=length(gm.prior.p)
    if nComp<=nCompMax
        return (gm,totalLossBound)
    end
    gcSet=Set{GaussComp}()
    for iComp in 1:nComp
        push!(gcSet,GaussComp(log(gm.prior.p[iComp]),gm.components[iComp]))
    end

    distance=Dict{Set{GaussComp},Float64}()

    minDist=Inf
    argMinDist=Set{GaussComp}()
    for gc1 in gcSet
        for gc2 in gcSet
            s=Set{GaussComp}([gc1,gc2])

            if gc1!=gc2 && !haskey(distance,s)
                d=sumSqDiff(gc1,gc2)
                distance[s]=d
                if d<minDist
                    minDist=d
                    argMinDist=Set{GaussComp}([gc1,gc2])
                end
            end
        end
    end

    while(length(gcSet)>nCompMax)

        gc = collect(argMinDist)
        gc3=fusion(gc[1],gc[2])

        # erase the corresponding keys from the distance table
        for k in keys(distance)
            if in(gc[1],k) || in(gc[2],k)
                delete!(distance,k)
            end
        end
        # erase from gcSet
        delete!(gcSet,gc[1])
        delete!(gcSet,gc[2])
        totalLoss+=sqrt(minDist)
        for gc1 in gcSet
            s=Set{GaussComp}([gc1,gc3])
            d=sumSqDiff(gc1,gc3)
            distance[s]=d
        end

        # update gcSet after
        push!(gcSet,gc3)

        minDist=Inf
        argMinDist=Set{GaussComp}()
        for k in keys(distance)
            if distance[k]<minDist
                argMinDist=k
                minDist=distance[k]
            end
        end
    end

    nComp=length(gcSet)
    weight=Array(Float64,nComp)
    normalArray=Array(FullNormal,nComp)
    iComp=1
    for gComp in gcSet
        weight[iComp]=exp(gComp.logW)
        normalArray[iComp]=gComp.normal
        iComp+=1
    end

    weight=weight/sum(weight)
    return  (MixtureModel(normalArray,weight),totalLossBound)
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
    w = Array(Float64,nComp)

    for iComp in 1:nComp
        w[iComp]=exp(gcArray[iComp].logW)
        normalArray[iComp] = gcArray[iComp].normal
    end
    return  MixtureModel(normalArray,w)
end




function conditionalProba(gma::GaussianMixtureAuxiliary,x)
    # only the gma.idxGiven indexes of x are taken into account as xQ !
    # todo : use logDet2piCov where possible
    gm=gma.gm
    idxGiven=gma.idxGiven
    idxNonNegligibleGaussianComp=multiIndexToNonNegligibleComp(pointToBoxMultiIndex(x,gma.grid),gma)
    d=length(mean(gm.components[1]))
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
        PinvCPt=P*invC*P'
        M = (PinvCPt)\((PinvCPt)*P*mu - P*invC*Q'*Q*(x-mu))
        Sigma = inv(PinvCPt)

        logKappa = -0.5*(- M'*PinvCPt*M + (P*mu)'*PinvCPt*(P*mu) + (Q*(x-mu))'*Q*invC*Q'*Q*(x-mu) - 2*(P*mu)'*P*invC*Q'*Q*(x-mu))
        logCoeffGauss[count] = log(nu) -0.5*log(det(2*pi*C)) + logKappa[1] + 0.5*log(det(2*pi*Sigma))
        gaussCompArray[count] = MvNormal(M,Sigma)
        count+=1
    end

    logCoeffGauss=logCoeffGauss-maximum(logCoeffGauss)
    wGauss = exp(logCoeffGauss)
    wGauss = wGauss/sum(wGauss)

    return MixtureModel(gaussCompArray,wGauss)
end



function findBoundsNegQuadraticFormOnBox1(zL,zU,iComp,gma)
    # lambdaSmall is the lowest eigenvalue of sqrtQ'*sqrtQ
    # lambdaBig is the biggest
    # assumes we want to lower/higer bound -0.5*sumsquare(sqrtQ*(x-mu)) + logKhi
    # this function is faster than findMaxNegQuadraticFormOnBox
    # and much faster than findMinNegQuadraticFormOnBox especially in high dimensions

    lambdaSmall=gma.eigLambdaSmallArray[iComp]
    lambdaBig=gma.eigLambdaBigArray[iComp]
    logKhi = -0.5*gma.logDet2piCov[iComp] + log(gma.gm.prior.p[iComp])
    m=(zL+zU)/2
    r=norm((zU-zL)/2)
    upperBound=0
    lowerBound=0
    mu=mean(gma.gm.components[iComp])
    dist=norm(mu-m)
    if dist < r
        upperBound=logKhi
    else
        upperBound= -lambdaSmall*sumSq(m - r*(mu-m)/dist - mu) + logKhi
    end
    if dist<=eps(Float64)
        lowerBound = -lambdaBig*r^2
    else
        lowerBound= -lambdaBig*sumSq(m + r*(mu-m)/dist - mu) + logKhi
    end
    return (lowerBound,upperBound)
end



function findBoundsNegQuadraticFormOnBox2(zL,zU,iComp,gma)
    A=diagm(1./(zU-zL))
    invA=diagm(zU-zL)
    r=sqrt(length(zU))
    m=(zU+zL)/2
    mu=mean(gma.gm.components[iComp])
    logKhi=-0.5*gma.logDet2piCov[iComp] + log(gma.gm.prior.p[iComp])
    mb=A*m
    mub=A*mu
    invC=gma.invSqrtCovArray[iComp]^2
    DR = eig(invA*invC*invA)
    dist= norm(mub-mb)
    upperBound=0
    if(dist<r)
        upperBound=logKhi
    else
        lambdaSmall=minimum(DR[1])
        upperBound = -lambdaSmall*sumSq(mb-r*(mub-mb)/dist - mub) + logKhi
    end
    lowerBound=0
    lambdaBig=maximum(DR[1])
    if norm(mb-mub)<eps(Float64)
        lowerBound = -lambdaBig*r^2 + logKhi
    else
        lowerBound = -lambdaBig*sumSq(mb+r*(mub-mb)/dist -mub) + logKhi
    end
    return (lowerBound,upperBound)
end


function findBoundsNegQuadraticFormOnBox(zL,zU,iComp,gma)
    l1,u1=findBoundsNegQuadraticFormOnBox1(zL,zU,iComp,gma)
    l2,u2=findBoundsNegQuadraticFormOnBox2(zL,zU,iComp,gma)
    return (maximum([l1,l2]),minimum([u1,u2]))
end





function multiIndexToNonNegligibleComp(multiIndex::Array{Int64,1}, gma::GaussianMixtureAuxiliary)
    d=length(mean(gm.components[1]))
    @assert d==length(multiIndex)
    maxLen = size(gma.grid.y)[1]
    if reduce(|,[(v==0 | v==maxLen) for v in multiIndex])
        return Set{Int64}(1:length(gma.gm.prior.p))
    end

    if haskey(gma.grid.multiIndexToNonNegligibleComp,multiIndex)
        return gma.grid.multiIndexToNonNegligibleComp[multiIndex]
    else
        nnIdx=findBoxNonNegligibleComp(multiIndex, gma)
        # todo update the Dict !
        gma.grid.multiIndexToNonNegligibleComp[multiIndex]=nnIdx
        return nnIdx
    end
end



function findBoxNonNegligibleComp(multiIndex::Array{Int64,1}, gma::GaussianMixtureAuxiliary)
    xL,xU=getBoxBoundaries(multiIndex,gma.grid)
    gm=gma.gm
    nComp=length(gm.prior.p)
    invSqrtCovArray = gma.invSqrtCovArray
    logU=Array(Float64,nComp)
    logL=Array(Float64,nComp)
    for iComp in 1:nComp
        logU[iComp],logL[iComp]=findBoundsNegQuadraticFormOnBox(xL,xU,iComp,gma)
    end
    maxLogU = maximum(logU) # can't take logL because it can be -Inf
    logU-=maxLogU
    logL-=maxLogU

    U=exp(logU)
    L=exp(logL)
    #idxSortU=sortperm(logU)
    idxSortL=sortperm(logL)
    K=1E3
    alpha = nComp
    # nNegligibleCompTarget=nComp*0.9
    sU=0.0 #U[idxSortU[alpha]]
    sL=L[idxSortL[alpha]]

    nonNegligibleIndexSet=Set{Int64}(idxSortL[alpha])
    for i in 1:nComp
        if !in(i,nonNegligibleIndexSet)
            sU+=U[i]
        end
    end
    while(K*sU>sL && alpha>1)
        alpha-=1
        sL+=L[idxSortL[alpha]]
        push!(nonNegligibleIndexSet,idxSortL[alpha])
        sU-=U[idxSortL[alpha]]
    end
    return nonNegligibleIndexSet
end




#===========================================
Less useful
===========================================#
#=
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


function findMaxNegQuadraticFormOnBox(sqrtQ,mu,logKhi,xL,xU)
    # assumes we maximize -0.5*sumsquare(sqrtQ*(x-mu)) + logKhi
    # /!\ the optimizer displays warnings
    x=Variable(length(mu));
    problem = minimize(sumsquares(sqrtQ*(x-mu)), [x<=xU,x>=xL])
    solve!(problem,SCSSolver(verbose=0))
    solution = problem.optval
    return -0.5*solution + logKhi
end



function findUpperBoundNegQuadraticFormOnBox(eigLambdaSmall,mu,logKhi,xL,xU)
    # eigLambdaEnd is the lowest eigenvalue of sqrtQ'*sqrtQ
    # assumes we want to lower bound -0.5*sumsquare(sqrtQ*(x-mu)) + logKhi
    # this function is faster than findMaxNegQuadraticFormOnBox
    m=(xL+xU)/2
    r=norm((xU-xL)/2)
    if norm(mu-m) < r
        return logKhi
    else
        return -eigLambdaSmall*sumSq(m - r*(mu-m)/norm(m-mu) - mu) +logKhi
    end
end



function findLowerBoundNegQuadraticFormOnBox(eigLambda0,mu,logKhi,xL,xU)
    # function findLowerBoundNegQuadraticFormOnBox(zL,zU,iComp,gma)
    eigLambda0= gma.
    # eigLambda0 is the highest eigenvalue of sqrtQ'*sqrtQ
    # assumes we want to lower bound -0.5*sumsquare(sqrtQ*(x-mu)) + logKhi
    # this function is much faster than findMinNegQuadraticFormOnBox for high dimensions
    if sumSq(xL)==Inf || sumSq(xU)==Inf
        return -Inf
    end
    if m==mu # very unlikely case
        return  -sumSq((xU-xL)/2)*eigLambda0 +logKhi
    else
        m=(xL+xU)/2
        r=norm((xU-xL)/2)
        return -eigLambda0*sumSq(m - r*(m-mu)/norm(m-mu) - mu)+logKhi
    end
end



function sumSqDiff(idxToFuse1::Int64,idxToFuse2::Int64,gm::GaussianMixture)
    #=
    yields the value of the integral of the square of the difference
    between gm and a modified version of gm where the components
    idxToFuse1 and idxToFuse2 are fused into one gaussian
    todo : optimize because the usage of the function product may be not necessary
    =#
    return sumSqDiff(GaussComp(log(gm.prior.p[idxToFuse1]), gm.components[idxToFuse1]),
                     GaussComp(log(gm.prior.p[idxToFuse2]), gm.components[idxToFuse2]))
end
=#

#=================================
Testing
==================================#

gm=randomDrawGaussianMixture(50)
idxGiven=[1]
gma=GaussianMixtureAuxiliary(gm,50,idxGiven)

#findBoxNonNegligibleComp([2,2], gma)




function testConditionalProba(gma::GaussianMixtureAuxiliary)
    nComp=10
    gm = randomDrawGaussianMixture(nComp)
    idxGiven = [2]
    x=randn(2)
    nPoint=5
    gma= GaussianMixtureAuxiliary(gm,nPoint::Int64,idxGiven)

    #idxNonNegligibleGaussianComp = findBoxNonNegligibleComp(pointToBoxMultiIndex(x,gma.grid),gma)
    #multiIndex
    #idxNonNegligibleGaussianComp=findBoxNonNegligibleComp( ,gma)
    idxNonNegligibleGaussianComp=Set{Int64}(collect(1:nComp))
    cgm = conditionalProba(gma,x)

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





