using LinearAlgebra: BlasFloat

const CuDense{ElT,VecT} = Dense{ElT,VecT} where {VecT<:CuVector}
const CuDenseTensor{ElT,N,StoreT,IndsT} = Tensor{ElT,N,StoreT,IndsT} where {StoreT<:CuDense}

function Base.complex(::Type{Dense{ElT,VT}}) where {ElT,VT<:CuArray}
  return Dense{complex(ElT),CuVector{complex(ElT)}}
end

CuArray(x::CuDense{ElT}) where {ElT} = CuVector{ElT}(data(x))
function CuArray{ElT,N}(x::CuDenseTensor{ElT,N}) where {ElT,N}
  return CuArray{ElT,N}(reshape(data(store(x)), dims(inds(x))...))
end
CuArray(x::CuDenseTensor{ElT,N}) where {ElT,N} = CuArray{ElT,N}(x)

*(D::Dense{T,AT}, x::S) where {T,AT<:CuArray,S<:Number} = Dense(x .* data(D))

Base.getindex(D::CuDense{<:Number}) = collect(data(D))[]
Base.getindex(D::CuDenseTensor{<:Number,0}) = store(D)[]
LinearAlgebra.norm(T::CuDenseTensor) = norm(data(store(T)))

function Base.copyto!(R::CuDenseTensor{<:Number,N}, T::CuDenseTensor{<:Number,N}) where {N}
  RA = array(R)
  TA = array(T)
  RA .= TA
  return R
end
