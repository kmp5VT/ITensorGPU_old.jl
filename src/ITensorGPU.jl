module ITensorGPU
#const ContractionPlans = Dict{String, Tuple{cutensorAlgo_t, cutensorContractionPlan_t}}()
include("import.jl")
#const ContractionPlans = Dict{String,cutensorAlgo_t}()

include("cuarray/set_types.jl")
include("traits.jl")
include("tensor/cudense.jl")
include("tensor/dense.jl")
include("tensor/culinearalgebra.jl")
include("tensor/cutruncate.jl")
include("tensor/cucombiner.jl")
include("tensor/cudiag.jl")
include("cuitensor.jl")
include("mps/cumps.jl")

export cpu, cuITensor, randomCuITensor, cuMPS, randomCuMPS, productCuMPS, randomCuMPO, cuMPO


end #module
