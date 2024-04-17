using ITensorGPU
using Documenter

DocMeta.setdocmeta!(ITensorGPU, :DocTestSetup, :(using ITensorGPU); recursive=true)

makedocs(;
    modules=[ITensorGPU],
    authors="ITensor developers",
    sitename="ITensorGPU.jl",
    format=Documenter.HTML(;
        canonical="https://ITensor.github.io/ITensorGPU.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=["Home" => "index.md"],
)

deploydocs(; repo="github.com/ITensor/ITensorGPU.jl", devbranch="main")
