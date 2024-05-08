
#
# Linear Algebra of order 2 Tensors
#

# svd of an order-2 tensor
# function LinearAlgebra.svd(T::CuDenseTensor{ElT,2,IndsT}; kwargs...) where {ElT,IndsT}
#   maxdim = get(kwargs, :maxdim, minimum(dims(T)))
#   mindim = get(kwargs, :mindim, 1)
#   cutoff = get(kwargs, :cutoff, 0.0)
#   absoluteCutoff = get(kwargs, :absoluteCutoff, false)
#   doRelCutoff = get(kwargs, :doRelCutoff, true)
#   fastSVD = get(kwargs, :fastSVD, false)
#   # Safer to use `Array`, which ensures
#   # no views/aliases are made, since
#   # we are using in-place `CUSOLVER.svd!` below.
#   aT = Array(T)
#   @timeit "CUSOLVER svd" begin
#     MU, MS, MV = CUSOLVER.svd!(aT)
#   end
#   if !(MV isa CuMatrix)
#     # Materialize any array wrappers,
#     # for now, since `Adjoint` wrappers
#     # seem to cause issues throughout
#     # CUDA.jl, for example with slicing,
#     # reshaping and then copying, etc.
#     # TODO: Fix this in a more robust way.
#     MV = copy(MV)
#   end
#   # for consistency with cpu version,
#   # ITensors.jl/NDTensors/src/linearalgebra.jl/svd
#   # need conj!(MV)
#   conj!(MV)
#   P = MS .^ 2
#   truncerr, docut, P = truncate!(
#     P;
#     mindim=mindim,
#     maxdim=maxdim,
#     cutoff=cutoff,
#     absoluteCutoff=absoluteCutoff,
#     doRelCutoff=doRelCutoff,
#   )
#   spec = Spectrum(P, truncerr)
#   dS = length(P)
#   if dS < length(MS)
#     MU = MU[:, 1:dS]
#     MS = MS[1:dS]
#     MV = MV[:, 1:dS]
#   end

#   # Make the new indices to go onto U and V
#   u = eltype(IndsT)(dS)
#   v = eltype(IndsT)(dS)
#   Uinds = IndsT((ind(T, 1), u))
#   Sinds = IndsT((u, v))
#   Vinds = IndsT((ind(T, 2), v))
#   U = tensor(Dense(vec(MU)), Uinds)
#   Sdata = CUDA.zeros(ElT, dS * dS)
#   dsi = diagind(reshape(Sdata, dS, dS), 0)
#   Sdata[dsi] = MS
#   S = tensor(Dense(Sdata), Sinds)
#   V = tensor(Dense(vec(MV)), Vinds)
#   return U, S, V, spec
# end

# function LinearAlgebra.eigen(
#   T::Hermitian{ElT,<:CuDenseTensor{ElT,2,IndsT}}; kwargs...
# ) where {ElT<:Union{Real,Complex},IndsT}
#   ispossemidef::Bool = get(kwargs, :ispossemidef, false)
#   maxdim::Int = get(kwargs, :maxdim, minimum(dims(T)))
#   mindim::Int = get(kwargs, :mindim, 1)
#   cutoff::Float64 = get(kwargs, :cutoff, 0.0)
#   absoluteCutoff::Bool = get(kwargs, :absoluteCutoff, false)
#   doRelCutoff::Bool = get(kwargs, :doRelCutoff, true)
#   @timeit "CUSOLVER eigen" begin
#     local DM, UM
#     if ElT <: Complex
#       DM, UM = CUSOLVER.heevd!('V', 'U', matrix(parent(T)))
#     else
#       DM, UM = CUSOLVER.syevd!('V', 'U', matrix(parent(T)))
#     end
#   end
#   DM_ = reverse(DM)
#   @timeit "truncate" begin
#     truncerr, docut, DM = truncate!(
#       DM_;
#       maxdim=maxdim,
#       cutoff=cutoff,
#       absoluteCutoff=absoluteCutoff,
#       doRelCutoff=doRelCutoff,
#     )
#   end
#   spec = Spectrum(DM, truncerr)
#   dD = length(DM)
#   dV = reverse(UM; dims=2)
#   if dD < size(dV, 2)
#     dV = CuMatrix(dV[:, 1:dD])
#   end
#   # Make the new indices to go onto U and V
#   l = eltype(IndsT)(dD)
#   r = eltype(IndsT)(dD)
#   Vinds = IndsT((dag(ind(T, 2)), dag(r)))
#   Dinds = IndsT((l, dag(r)))
#   U = tensor(Dense(vec(dV)), Vinds)
#   D = tensor(Diag(real.(DM)), Dinds)
#   return D, U, spec
# end

# function LinearAlgebra.qr(T::CuDenseTensor{ElT,2,IndsT}; kwargs...) where {ElT,IndsT}
#   QM, RM = qr(matrix(T))
#   # Make the new indices to go onto Q and R
#   q, r = inds(T)
#   q = dim(q) < dim(r) ? sim(q) : sim(r)
#   Qinds = IndsT((ind(T, 1), q))
#   Rinds = IndsT((q, ind(T, 2)))
#   QM = CuMatrix(QM)
#   Q = tensor(Dense(vec(QM)), Qinds)
#   R = tensor(Dense(vec(RM)), Rinds)
#   return Q, R
# end

# function polar(T::CuDenseTensor{ElT,2,IndsT}) where {ElT,IndsT}
#   QM, RM = polar(matrix(T))
#   dim = size(QM, 2)
#   # Make the new indices to go onto Q and R
#   q = eltype(IndsT)(dim)
#   # TODO: use push/pushfirst instead of a constructor
#   # call here
#   Qinds = IndsT((ind(T, 1), q))
#   Rinds = IndsT((q, ind(T, 2)))
#   Q = tensor(Dense(vec(QM)), Qinds)
#   R = tensor(Dense(vec(RM)), Rinds)
#   return Q, R
# end
