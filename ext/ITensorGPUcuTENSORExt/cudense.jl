function Base.:+(B::CuDenseTensor, A::CuDenseTensor)
    opC = cuTENSOR.CUTENSOR_OP_IDENTITY
    opA = cuTENSOR.CUTENSOR_OP_IDENTITY
    opAC = cuTENSOR.CUTENSOR_OP_ADD
    Ais = inds(A)
    Bis = inds(B)
    ind_dict = Vector{Index}()
    for (idx, i) in enumerate(inds(A))
      push!(ind_dict, i)
    end
    Adata = data(store(A))
    Bdata = data(store(B))
    reshapeBdata = reshape(Bdata, dims(Bis)...)
    reshapeAdata = reshape(Adata, dims(Ais)...)
    ctainds = zeros(Int, length(Ais))
    ctbinds = zeros(Int, length(Bis))
    for (ii, ia) in enumerate(Ais)
      ctainds[ii] = findfirst(x -> x == ia, ind_dict)
    end
    for (ii, ib) in enumerate(Bis)
      ctbinds[ii] = findfirst(x -> x == ib, ind_dict)
    end
    ctcinds = copy(ctbinds)
    C = CUDA.zeros(eltype(Bdata), dims(Bis)...)
    cuTENSOR.elementwiseBinary!(
      one(eltype(Adata)),
      reshapeAdata,
      ctainds,
      opA,
      one(eltype(Bdata)),
      reshapeBdata,
      ctbinds,
      opC,
      C,
      ctcinds,
      opAC,
    )
    copyto!(data(store(B)), vec(C))
    return B
  end
  
  function Base.:+(B::CuDense, Bis::IndexSet, A::CuDense, Ais::IndexSet)
    opA = identity
    opC = identity
    opAC = cuTENSOR.CUTENSOR_OP_ADD
    ind_dict = Vector{Index}()
    for (idx, i) in enumerate(Ais)
      push!(ind_dict, i)
    end
    Adata = data(A)
    Bdata = data(B)
    reshapeBdata = reshape(Bdata, dims(Bis)...)
    reshapeAdata = reshape(Adata, dims(Ais)...)
    ctainds = zeros(Int, length(Ais))
    ctbinds = zeros(Int, length(Bis))
    for (ii, ia) in enumerate(Ais)
      ctainds[ii] = findfirst(x -> x == ia, ind_dict)
    end
    for (ii, ib) in enumerate(Bis)
      ctbinds[ii] = findfirst(x -> x == ib, ind_dict)
    end
    ctcinds = copy(ctbinds)
    C = CUDA.zeros(eltype(Bdata), dims(Bis)...)
    Cis = Bis
    C = cuTENSOR.elementwiseBinary!(
      1, reshapeAdata, ctainds, opA, 1, reshapeBdata, ctbinds, opC, C, ctcinds, opAC
    )
    copyto!(data(B), vec(C))
  end
  
  function Base.:-(B::CuDenseTensor, A::CuDenseTensor)
    opC = cuTENSOR.CUTENSOR_OP_IDENTITY
    opA = cuTENSOR.CUTENSOR_OP_IDENTITY
    opAC = cuTENSOR.CUTENSOR_OP_ADD
    Ais = inds(A)
    Bis = inds(B)
    ind_dict = Vector{Index}()
    for (idx, i) in enumerate(inds(A))
      push!(ind_dict, i)
    end
    Adata = data(store(A))
    Bdata = data(store(B))
    reshapeBdata = reshape(Bdata, dims(Bis)...)
    reshapeAdata = reshape(Adata, dims(Ais)...)
    ctainds = zeros(Int, length(Ais))
    ctbinds = zeros(Int, length(Bis))
    for (ii, ia) in enumerate(Ais)
      ctainds[ii] = findfirst(x -> x == ia, ind_dict)
    end
    for (ii, ib) in enumerate(Bis)
      ctbinds[ii] = findfirst(x -> x == ib, ind_dict)
    end
    ctcinds = copy(ctbinds)
    C = CUDA.zeros(eltype(Bdata), dims(Bis))
    cuTENSOR.elementwiseBinary!(
      -one(eltype(Adata)),
      reshapeAdata,
      ctainds,
      opA,
      one(eltype(Bdata)),
      reshapeBdata,
      ctbinds,
      opC,
      C,
      ctcinds,
      opAC,
    )
    copyto!(data(store(B)), vec(C))
    return B
  end
  
  function Base.:-(A::CuDense, Ais::IndexSet, B::CuDense, Bis::IndexSet)
    opA = cuTENSOR.CUTENSOR_OP_IDENTITY
    opC = cuTENSOR.CUTENSOR_OP_IDENTITY
    opAC = cuTENSOR.CUTENSOR_OP_ADD
    ind_dict = Vector{Index}()
    for (idx, i) in enumerate(Ais)
      push!(ind_dict, i)
    end
    Adata = data(A)
    Bdata = data(B)
    reshapeBdata = reshape(Bdata, dims(Bis)...)
    reshapeAdata = reshape(Adata, dims(Ais)...)
    ctainds = zeros(Int, length(Ais))
    ctbinds = zeros(Int, length(Bis))
    for (ii, ia) in enumerate(Ais)
      ctainds[ii] = findfirst(x -> x == ia, ind_dict)
    end
    for (ii, ib) in enumerate(Bis)
      ctbinds[ii] = findfirst(x -> x == ib, ind_dict)
    end
    ctcinds = copy(ctbinds)
    C = CUDA.zeros(eltype(Bdata), dims(Bis)...)
    Cis = Bis
    C = cuTENSOR.elementwiseBinary!(
      one(eltype(Adata)),
      reshapeAdata,
      ctainds,
      opA,
      -one(eltype(Bdata)),
      reshapeBdata,
      ctbinds,
      opC,
      C,
      ctcinds,
      opAC,
    )
    copyto!(data(B), vec(C))
    return C
  end