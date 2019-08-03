using Pkg, InteractiveUtils
ENV["PYTHON"] = ""
Pkg.build("Conda")
Pkg.build("PyCall")
using Conda
Conda.add("pytorch"; channel = "pytorch")
pkg"precompile"