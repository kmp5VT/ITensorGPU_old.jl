using NDTensors

using Adapt
using CUDA
using CUDA.CUBLAS
using CUDA.CUSOLVER
using Functors
using ITensors
using LinearAlgebra
using Random
using SimpleTraits
using StaticArrays
using Strided
using TimerOutputs

using NDTensors: setdata, setstorage, cpu, IsWrappedArray, parenttype

import Adapt: adapt_structure
import Base: *, permutedims!
import CUDA: CuArray, CuMatrix, CuVector
import CUDA.Mem: pin
import ITensors:
  randn!,
  compute_contraction_labels,
  eigen,
  tensor,
  scale!,
  unioninds,
  array,
  matrix,
  vector,
  polar,
  tensors,
  truncate!,
  leftlim,
  rightlim,
  permute,
  BroadcastStyle,
  Indices
import NDTensors:
  Atrans,
  Btrans,
  CombinerTensor,
  ContractionProperties,
  Combiner,
  Ctrans,
  Diag,
  DiagTensor,
  Dense,
  DenseTensor,
  NonuniformDiag,
  NonuniformDiagTensor,
  Tensor,
  UniformDiag,
  UniformDiagTensor,
  _contract!!,
  _contract!,
  _contract_scalar!,
  _contract_scalar_noperm!,
  can_contract,
  compute_contraction_properties!,
  contract!!,
  contract!,
  contract,
  contraction_output,
  contraction_output_type,
  data,
  getperm,
  ind,
  is_trivial_permutation,
  outer!,
  outer!!,
  permutedims!!,
  set_eltype,
  set_ndims,
  similartype,
  zero_contraction_output

#using cuTENSOR
#import cuTENSOR: cutensorContractionPlan_t, cutensorAlgo_t