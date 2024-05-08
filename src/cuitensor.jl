import CUDA: CUDA, CuArray
using ITensors.NDTensors: NDTensors, NeverAlias, AliasStyle, AllowAlias
using NDTensors: cu
using ITensors: ITensors, ITensor, itensor

function cuITensor(eltype::Type{<:Number}, inds::IndexSet)
  return NDTensors.cu(ITensor(eltype, dim(inds), inds))
end

cuITensor(::Type{T}, inds::Index...) where {T<:Number} = cuITensor(T, IndexSet(inds...))

cuITensor(is::IndexSet) = cuITensor(Float64, is)
cuITensor(inds::Index...) = cuITensor(IndexSet(inds...))

cuITensor() = NDTensors.cu(ITensor())

function cuITensor(x::Number, inds::IndexSet)
  return NDTensors.cu(ITensor(x, inds))
end

cuITensor(x::Number, inds::Index...) = cuITensor(x, IndexSet(inds...))

cuITensor(data::Array, inds...) = cu(ITensor(data, inds...))

cuITensor(data::CuArray, inds...) = ITensor(data, inds...)

cuITensor(A::ITensor) = cu(A)

function randomCuITensor(elt::Type, inds::Indices)
  T = cuITensor(elt, inds)
  randn!(T)
  return T
end

function randomCuITensor(elt::Type{<:Number}, inds::Index...)
  return randomCuITensor(elt, IndexSet(inds...))
end
randomCuITensor(inds::IndexSet) = randomCuITensor(Float64, inds)
randomCuITensor(inds::Index...) = randomCuITensor(Float64, IndexSet(inds...))

CuArray(A::ITensor, args...) = array(A, args...)
CuMatrix(A::ITensor, args...) = array(A, args...)
