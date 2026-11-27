# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# Macro-cell finite volume (MacroCellFV) solver: finite volumes with face reconstruction
# on blocks of subcells.
#
# Each mesh element (macro-cell) holds an equidistant block of `n_cells` finite volume
# subcells per coordinate direction, stored with a `UniformFiniteVolumeBasis` as for
# `BlockFV`, whose containers, analysis, and visualization are reused. In contrast to
# `BlockFV`, which uses the subcell averages as face states and the volume and surface
# integrals of the DG solvers, the faces of each macro-cell are evaluated on a padded
# work array (pad) that holds the subcells of the macro-cell and `nhalo` halo subcells
# on each side. The pipeline of each RHS evaluation is, per macro-cell,
#   1. gather the subcells and the halos (from the neighbors or the ghost subcells of
#      physical boundaries) into the pad of the thread; only the halo subcells read by
#      the reconstruction are filled (`nhalo - 1` at -x towards a neighbor, whose
#      macro-cell computes the face at -x),
#   2. moment inversion (cell averages -> point values at the subcell centers, into a
#      second thread-local pad; `MomentInversionIdentity` uses the averages in place),
#   3. transformation to the reconstruction variables (`center2recon`), pointwise for
#      the point values,
#   4. face reconstruction of each subcell (once, both faces, one variable at a time)
#      into thread-local line buffers,
#   5. transformation of the face values to the variables of the numerical flux
#      (`face2flux`), numerical fluxes at all faces of the line, and the flux
#      differences,
# followed by the completion of the flux differences of the first subcell of each
# macro-cell with the interface flux, which is computed only once, by the macro-cell on
# its left side.
#
# The pad holds the transformed state (reconstruction variables) of a contiguous line of
# subcells across the macro-cell boundaries. The same kernel thus serves the faces
# inside a macro-cell and at its boundaries without index branching or repeated
# variable transformations in the stencils. Each thread owns one pad of
# `n_cells + 2 nhalo` subcells and one set of line buffers, so their memory does not grow
# with the number of elements. Pad and line buffers store each variable contiguously
# along the line; steps 4 and 5 are loops over the subcells or faces of the line, which
# vectorize.

"""
    VolumeIntegralMacroCellFV(; moment_inversion, reconstruction, schemes,
                                surface_flux, surface_gradient, center2recon, face2flux,
                                initial_condition_nodes)

Volume integral of the [`MacroCellFV`](@ref) solver. It collects the components of the
scheme; construct the solver with [`MacroCellFV`](@ref) instead of using this type
directly.

!!! warning "Experimental code"
    This code is experimental and may change in any future release.
"""
struct VolumeIntegralMacroCellFV{MomentInversion, Reconstruction, Schemes,
                                 SurfaceFlux, SurfaceGradient, Center2Recon,
                                 Face2Flux} <:
       AbstractVolumeIntegral
    moment_inversion::MomentInversion
    # The scheme of all reconstruction variables if `schemes === nothing`
    reconstruction::Reconstruction
    # One scheme per reconstruction variable (a tuple) or `nothing`
    schemes::Schemes
    surface_flux::SurfaceFlux
    surface_gradient::SurfaceGradient
    center2recon::Center2Recon
    face2flux::Face2Flux
    # Gauss-Legendre nodes per direction for the cell averages of the initial
    # condition (1: point value at the subcell center)
    initial_condition_nodes::Int
    # Stencil factor of the diffusive time step (see `diffusion_dt_factor`)
    diffusion_dt_factor::Float64
end

function VolumeIntegralMacroCellFV(;
                                   moment_inversion = MomentInversionIdentity(),
                                   reconstruction = nothing,
                                   schemes = nothing,
                                   surface_flux,
                                   surface_gradient = gradient_sum,
                                   center2recon = cons2cons,
                                   face2flux = cons2cons,
                                   initial_condition_nodes = 1)
    initial_condition_nodes >= 1 ||
        throw(ArgumentError("`initial_condition_nodes` must be at least 1"))
    if schemes === nothing
        reconstruction = reconstruction === nothing ? ReconstructionMUSCL() :
                         reconstruction
        reconstruction = point_reconstruction(moment_inversion, reconstruction)
    else
        reconstruction === nothing ||
            throw(ArgumentError("give either `reconstruction` (the same scheme for " *
                                "all reconstruction variables) or `schemes` (one " *
                                "scheme per reconstruction variable)"))
        schemes isa Tuple && !isempty(schemes) ||
            throw(ArgumentError("`schemes` must be a non-empty tuple"))
        schemes = map(scheme -> point_reconstruction(moment_inversion, scheme),
                      schemes)
    end
    all_schemes = schemes === nothing ? (reconstruction,) : schemes
    dt_factor = maximum(scheme -> diffusion_dt_factor(scheme, moment_inversion),
                        all_schemes)
    return VolumeIntegralMacroCellFV(moment_inversion, reconstruction, schemes,
                                     surface_flux, surface_gradient, center2recon,
                                     face2flux, Int(initial_condition_nodes),
                                     Float64(dt_factor))
end

function Base.show(io::IO, ::MIME"text/plain",
                   integral::VolumeIntegralMacroCellFV)
    @nospecialize integral
    setup = Pair{String, Any}["moment inversion" => integral.moment_inversion]
    if integral.schemes === nothing
        push!(setup, "reconstruction" => integral.reconstruction)
    else
        for (v, scheme) in enumerate(integral.schemes)
            push!(setup, "scheme of variable $v" => scheme)
        end
    end
    append!(setup,
            ["surface flux" => integral.surface_flux,
                "reconstruction variables" => integral.center2recon,
                "flux variables" => integral.face2flux])
    if has_gradients(integral) isa True
        push!(setup, "surface gradient" => integral.surface_gradient)
    end
    summary_box(io, "VolumeIntegralMacroCellFV", setup)
end

"""
    gradient_sum(d_left, d_right)

Derivative at a subcell face of [`MacroCellFV`](@ref) from the contributions of the
subcells on its left and right sides (the halves of a central stencil, see
[`GradientCentral`](@ref)): their sum.

!!! warning "Experimental code"
    This code is experimental and may change in any future release.
"""
@inline gradient_sum(d_left, d_right) = d_left + d_right

# Halo width: the moment inversion is applied to the halo cells used by the face
# reconstruction, so the stencil widths add up.
@inline function nhalo(volume_integral::VolumeIntegralMacroCellFV)
    @unpack moment_inversion, reconstruction, schemes = volume_integral
    return nhalo(moment_inversion) + nhalo_schemes(reconstruction, schemes)
end
@inline nhalo_schemes(reconstruction, ::Nothing) = nhalo(reconstruction)
@inline nhalo_schemes(reconstruction, schemes::Tuple) = maximum(nhalo, schemes)

# The scheme of the reconstruction variable `v`
@inline pad_scheme(volume_integral::VolumeIntegralMacroCellFV, v) = pad_scheme(volume_integral.reconstruction,
                                                                               volume_integral.schemes,
                                                                               v)
@inline pad_scheme(reconstruction, ::Nothing, v) = reconstruction
@inline pad_scheme(reconstruction, schemes::Tuple, v) = schemes[v]

# `True()` if any reconstruction variable has face derivatives (for diffusion)
@inline function has_gradients(volume_integral::VolumeIntegralMacroCellFV)
    @unpack reconstruction, schemes = volume_integral
    schemes === nothing && return has_gradient(reconstruction)
    return any(scheme -> has_gradient(scheme) isa True, schemes) ? True() : False()
end

"""
    MacroCellFV(; n_cells::Integer,
                  surface_flux,
                  reconstruction = ReconstructionMUSCL(),
                  schemes = nothing,
                  moment_inversion = MomentInversionIdentity(),
                  center2recon = cons2cons,
                  face2flux = cons2cons,
                  surface_gradient = gradient_sum,
                  initial_condition_nodes = 1,
                  RealT = Float64)

Create a macro-cell finite volume solver with `n_cells` equidistant finite volume
subcells per coordinate direction in each element (macro-cell) of the mesh, with
reconstruction of the face states. As for [`BlockFV`](@ref), the subcells
are stored with a [`UniformFiniteVolumeBasis`](@ref); for consistency with `BlockFV`,
the number of subcells can also be given as `n_nodes`. Currently supported on the 1D
and 2D [`TreeMesh`](@ref); in 2D only on periodic meshes without mortars so far. In
2D, the faces normal to x and to y are computed along the x- and y-lines of subcells
with the same (one-dimensional) reconstruction and face derivatives:
- With `MomentInversionIdentity`, they act on the subcell averages and give the face
  averages of the states and of the normal derivatives, so the scheme keeps the order
  of the stencils for linear fluxes (e.g., linear advection with
  [`LaplaceDiffusion2D`](@ref)) but is second-order accurate for nonlinear fluxes.
- With `MomentInversionCentral(order = 4)` (the only moment inversion to point values
  in 2D so far, `n_cells >= 5`), they act on the point values at the subcell centers
  and give point values at the face midpoints. The numerical fluxes there are
  converted to face averages along the faces (fourth order), so the scheme is
  fourth-order accurate also for nonlinear fluxes. No corner subcells are needed: at
  the ends of the face strips of the neighbors, the moment inversion and the
  conversion use one-sided stencils.

- `surface_flux`: numerical flux at the subcell faces, e.g.,
  [`flux_lax_friedrichs`](@ref) or [`flux_hllc`](@ref).
- `reconstruction`: [`ReconstructionConstant`](@ref) (first order, `nhalo = 1`),
  [`ReconstructionMUSCL`](@ref) (second order, `nhalo = 2`), or
  [`ReconstructionWENO5JS`](@ref)/[`ReconstructionWENO5Z`](@ref) (fifth order for
  smooth solutions, `nhalo = 3`), or [`ReconstructionCentral`](@ref) (central face
  values of even order, `nhalo = order/2`). The reconstruction reads the neighboring
  macro-cells through `nhalo` halo subcells per side, which requires `n_cells >= nhalo`.
  Orders above two require subcell averages as initial data, see
  `initial_condition_nodes`. A [`CombinedScheme`](@ref) adds face derivatives for
  diffusion.
- `schemes`: instead of `reconstruction`, one scheme per reconstruction variable (a
  tuple with one entry per output variable of `center2recon`), e.g., a WENO
  reconstruction of the density and the pressure and a [`CombinedScheme`](@ref) with
  [`GradientCentral`](@ref) for the velocity and the temperature. Variables without
  face derivatives cost no derivative work.
- `moment_inversion`: conversion of the subcell averages to point values at the
  subcell centers before the reconstruction: [`MomentInversionIdentity`](@ref) (the
  averages are used as point values), [`MomentInversionCentral`](@ref) (linear, of even
  order), or [`MomentInversionWENO`](@ref) (nonlinear, fourth order). With the moment
  inversions to point values, the WENO reconstructions interpolate the point values to
  the faces; other reconstructions are not supported. Their halo subcells add to those
  of the reconstruction.
- `center2recon`: transformation of the subcell values from conservative variables to
  the variables used for the reconstruction (e.g., [`cons2prim`](@ref)), applied after
  the moment inversion. With `MomentInversionIdentity`, it acts on the subcell
  averages, so a nonlinear transformation is only second-order accurate; with the
  moment inversions to point values, it acts on the point values and keeps the order.
  The number of reconstruction variables is the length of its output and may differ
  from the number of conservative variables (e.g., the primitive variables and the
  temperature for the compressible Navier-Stokes equations, see
  [`cons2prim_temperature_pressure`](@ref)).
- `face2flux`: transformation of the reconstructed face values to the variables
  expected by `surface_flux` (conservative variables for most numerical fluxes of
  Trixi.jl, e.g., [`prim2cons`](@ref) after `center2recon = cons2prim`).
  The names differ from `cons2recon`/`recon2cons` of the subcell finite volume methods
  in the DGSEM volume integrals, since the two transformations act at different places:
  on the subcell values before and on the face values after the reconstruction. A
  numerical flux that takes the reconstruction variables directly (e.g.,
  [`flux_hllc_primitive`](@ref) with `center2recon = cons2prim`) can be used with
  `face2flux = cons2cons`, which avoids the conversion at the faces.
  With diffusion, `face2flux` also selects the state and the gradients of the
  diffusive flux from the reconstruction variables at the faces (see
  [`macrocell_face2flux`](@ref)).
- `surface_gradient`: derivative at a face from the contributions of the subcells on
  both sides, [`gradient_sum`](@ref) for the central stencils.
- `initial_condition_nodes`: number of Gauss-Legendre nodes per coordinate direction
  for the subcell averages of the initial condition and of the ghost subcells of
  Dirichlet boundaries. The default `1` uses the value at the subcell center, which is
  second-order accurate.

With `reconstruction = ReconstructionConstant()`, the right-hand side is identical to
the one of `BlockFV(; n_nodes = n_cells, surface_flux)`.

**Diffusion.** With a [`SemidiscretizationHyperbolicParabolic`](@ref), the diffusive
fluxes of the parabolic equations (e.g., [`LaplaceDiffusion1D`](@ref) or
[`CompressibleNavierStokesDiffusion1D`](@ref) with `GradientVariablesPrimitive`) are
computed in the same pass over the subcells as the advective fluxes, from the face
derivatives of the schemes with [`GradientCentral`](@ref): the face flux is the
numerical flux minus the diffusive flux, `semidiscretize` returns a single (non-split)
`ODEProblem`, and the time step of the [`StepsizeCallback`](@ref) includes the diffusive
restriction with `cfl_parabolic > 0`. The state of the diffusive flux at a face is the
mean of the two reconstructed states (the central value for `ReconstructionCentral`).
The boundary conditions are imposed by the ghost subcells of the hyperbolic boundary
conditions; the parabolic boundary conditions must be the same (or both periodic).

**Source terms** are averaged over each subcell with the Gauss-Legendre quadrature of
`initial_condition_nodes`, with the subcell average as state.

!!! warning "Experimental code"
    This code is experimental and may change in any future release.
"""
const MacroCellFV = DG{Basis, Mortar, SurfaceIntegral,
                       VolumeIntegral} where {Basis <: UniformFiniteVolumeBasis,
                                              Mortar, SurfaceIntegral,
                                              VolumeIntegral <:
                                              VolumeIntegralMacroCellFV}

function MacroCellFV(; n_cells = nothing, n_nodes = nothing,
                     surface_flux,
                     reconstruction = nothing,
                     schemes = nothing,
                     moment_inversion = MomentInversionIdentity(),
                     center2recon = cons2cons,
                     face2flux = cons2cons,
                     surface_gradient = gradient_sum,
                     initial_condition_nodes = 1,
                     RealT = Float64)
    if (n_cells === nothing) == (n_nodes === nothing)
        throw(ArgumentError("give the number of subcells either as `n_cells` or as " *
                            "`n_nodes`"))
    end
    n_cells = n_cells === nothing ? n_nodes : n_cells
    basis = UniformFiniteVolumeBasis(RealT, n_cells)
    volume_integral = VolumeIntegralMacroCellFV(; moment_inversion, reconstruction,
                                                schemes, surface_flux,
                                                surface_gradient, center2recon,
                                                face2flux, initial_condition_nodes)
    # The surface integral is not used by the right-hand side, but it keeps the `DG`
    # interface (e.g., for callbacks) intact. As for `BlockFV`, the basis is passed as
    # mortar method.
    surface_integral = SurfaceIntegralWeakForm(surface_flux)
    return DG(basis, basis, surface_integral, volume_integral)
end

function Base.show(io::IO, mime::MIME"text/plain", dg::MacroCellFV)
    @nospecialize dg
    summary_header(io, "MacroCellFV")
    summary_line(io, "basis", dg.basis)
    summary_line(io, "volume integral", dg.volume_integral |> typeof |> nameof)
    summary_footer(io)
end

@inline nhalo(dg::MacroCellFV) = nhalo(dg.volume_integral)

# Number of reconstruction variables: the length of the output of `center2recon`
# (known from its type, so that the result is a compile-time constant)
@inline nreconvariables(::typeof(cons2cons), equations) = nvariables(equations)
@inline function nreconvariables(center2recon, equations)
    u = SVector(ntuple(_ -> 1.0, Val(nvariables(equations))))
    return length(center2recon(u, equations))
end
@inline function nreconvariables(volume_integral::VolumeIntegralMacroCellFV, equations)
    return nreconvariables(volume_integral.center2recon, equations)
end

# Number of variables (columns) of the pads: they hold the conservative variables
# (after the gather) and the reconstruction variables (after `center2recon`, in place).
# The point values are stored in a second pad of the same size (see
# `pad_moment_inversion!`).
@inline function npadvariables(volume_integral::VolumeIntegralMacroCellFV,
                               equations)
    return max(nvariables(equations), nreconvariables(volume_integral, equations))
end

# Same as `wrap_array` for `DGSEM`: with Polyester.jl threading, use a `PtrArray`, which
# does not allocate (the general fallback for `DG` allocates an array header for each
# call). The number of subcells is a type parameter of the `UniformFiniteVolumeBasis`.
@inline function wrap_array(u_ode::AbstractVector, mesh::AbstractMesh, equations,
                            dg::MacroCellFV, cache)
    @boundscheck begin
        @assert length(u_ode) ==
                nvariables(equations) * nnodes(dg)^ndims(mesh) * nelements(dg, cache)
    end
    # See the comments on the `DGSEM` version, in particular on the Heisenbug with
    # `Threads.@threads`
    if _PREFERENCE_THREADING === :polyester && LoopVectorization.check_args(u_ode)
        PtrArray(pointer(u_ode),
                 (StaticInt(nvariables(equations)),
                  ntuple(_ -> StaticInt(nnodes(dg)), ndims(mesh))...,
                  nelements(dg, cache)))
    else
        wrap_array_native(u_ode, mesh, equations, dg, cache)
    end
end

#####################################################################
# Coupling of the moment inversion, the reconstruction, and the numerical flux
# (the schemes are in `schemes/`)

# The moment inversions to point values at the subcell centers
const MomentInversionPoint = Union{MomentInversionCentral, MomentInversionWENO}

# The reconstruction used with the moment inversion: the WENO reconstructions of the
# point values with the moment inversions to point values
point_reconstruction(::MomentInversionIdentity, reconstruction) = reconstruction

function point_reconstruction(::MomentInversionPoint,
                              reconstruction::ReconstructionWENO5)
    return ReconstructionPointValues(reconstruction)
end

function point_reconstruction(::MomentInversionPoint,
                              reconstruction::ReconstructionCentral)
    return ReconstructionCentral(reconstruction.order; points = true)
end

function point_reconstruction(inversion::MomentInversionPoint, scheme::CombinedScheme)
    return CombinedScheme(point_reconstruction(inversion, scheme.reconstruction),
                          point_gradient(inversion, scheme.gradient))
end

# The derivative schemes for the point values
point_gradient(::MomentInversionIdentity, gradient) = gradient
point_gradient(::MomentInversionPoint, gradient::GradientNone) = gradient
function point_gradient(::MomentInversionPoint, gradient::GradientCentral)
    return GradientCentral(gradient.order; points = true)
end

function point_reconstruction(::MomentInversionPoint, reconstruction)
    throw(ArgumentError("the moment inversions to point values support only the WENO " *
                        "and the central reconstructions"))
end

# Numerical flux at a subcell face from the reconstructed states `w_ll` and `w_rr` (in
# reconstruction variables) on its two sides
@inline function macrocell_face_flux(w_ll, w_rr, orientation, equations,
                                     volume_integral::VolumeIntegralMacroCellFV)
    @unpack surface_flux, face2flux = volume_integral

    u_ll = face2flux(w_ll, equations)
    u_rr = face2flux(w_rr, equations)

    return surface_flux(u_ll, u_rr, orientation, equations)
end

# Advective-diffusive flux at a subcell face: the numerical flux of the reconstructed
# states `w_ll`, `w_rr` minus the diffusive flux of the face state `w_face` (the mean of
# the two states) and the derivatives `dw_face` of the reconstruction variables
@inline function macrocell_face_flux(w_ll, w_rr, w_face, dw_face, orientation,
                                     equations, equations_parabolic,
                                     volume_integral::VolumeIntegralMacroCellFV)
    @unpack surface_flux, face2flux = volume_integral

    u_ll, u_rr, u_diffusive, gradient = macrocell_face2flux(face2flux, w_ll, w_rr,
                                                            w_face, dw_face,
                                                            equations,
                                                            equations_parabolic)
    flux_advective = surface_flux(u_ll, u_rr, orientation, equations)
    gradients = macrocell_gradients(gradient, orientation, equations_parabolic)
    flux_diffusive = flux(u_diffusive, gradients, orientation, equations_parabolic)

    return flux_advective - flux_diffusive
end

# The gradients of the diffusive flux (one entry per coordinate direction) from the
# derivatives along the face normal `orientation`. In 1D, only the normal derivative
# exists; for 2D, see `macrocellfv_parabolic.jl`.
@inline macrocell_gradients(gradient, orientation, ::AbstractEquationsParabolic{1}) = (gradient,)

"""
    macrocell_face2flux(face2flux, w_ll, w_rr, w_face, dw_face,
                        equations, equations_parabolic)

Transformation of the reconstruction variables at a subcell face of
[`MacroCellFV`](@ref) with diffusion: from the reconstructed states `w_ll` and `w_rr`,
the face state `w_face` (their mean) and the derivatives `dw_face` of the
reconstruction variables along the face normal (zero for the variables without
derivatives), return `(u_ll, u_rr, u_diffusive, gradient)`: the states of the numerical
flux `surface_flux(u_ll, u_rr, orientation, equations)` and the state and the gradient
of the diffusive flux `flux(u_diffusive, (gradient,), orientation,
equations_parabolic)` (in the gradient variables of `equations_parabolic`).

The default uses `face2flux(w, equations)` for the states of the numerical flux and the
reconstruction variables as gradient variables (e.g., the conservative variables of
[`LaplaceDiffusion1D`](@ref) with `center2recon = face2flux = cons2cons`). Add a method
for the type of `face2flux` for other variables, see
[`face2flux_navier_stokes_primitive`](@ref).

!!! warning "Experimental code"
    This code is experimental and may change in any future release.
"""
@inline function macrocell_face2flux(face2flux, w_ll, w_rr, w_face, dw_face, equations,
                                     equations_parabolic)
    return face2flux(w_ll, equations), face2flux(w_rr, equations), w_face, dw_face
end

"""
    cons2prim_temperature_pressure(u, equations::CompressibleEulerEquations1D)

Conservative variables to the reconstruction variables `(rho, v1, p, p / rho)` of
[`MacroCellFV`](@ref) for the compressible Navier-Stokes equations: the primitive
variables and the temperature times the gas constant, `R T = p / rho` (so that the gas
constant of the parabolic equations is not needed here). Use it with
`face2flux = face2flux_navier_stokes_primitive` and a numerical flux in primitive
variables, e.g., [`flux_hllc_primitive`](@ref).

!!! warning "Experimental code"
    This code is experimental and may change in any future release.
"""
@inline function cons2prim_temperature_pressure(u,
                                                equations::CompressibleEulerEquations1D)
    rho, v1, p = cons2prim(u, equations)
    return SVector(rho, v1, p, p / rho)
end

"""
    face2flux_navier_stokes_primitive(w, equations)

Face transformation of [`MacroCellFV`](@ref) for the reconstruction variables
`(rho, v1, p, R T)` of [`cons2prim_temperature_pressure`](@ref): the primitive variables
`(rho, v1, p)` for a numerical flux in primitive variables (e.g.,
[`flux_hllc_primitive`](@ref)). With diffusion
([`CompressibleNavierStokesDiffusion1D`](@ref) with `GradientVariablesPrimitive`), the
state and the gradient of the diffusive flux are `(rho, v1, T)` and their derivatives.
Only the velocity and the temperature need face derivatives
([`GradientCentral`](@ref) in their schemes).

!!! warning "Experimental code"
    This code is experimental and may change in any future release.
"""
@inline function face2flux_navier_stokes_primitive(w, equations)
    return SVector(w[1], w[2], w[3])
end
end # @muladd
