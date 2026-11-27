# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# Diffusion with the `MacroCellFV` solver: a `SemidiscretizationHyperbolicParabolic`
# with `MacroCellFV` computes the advective and the diffusive fluxes in the same pass
# over the subcells (see `rhs_macrocellfv!`). Its `semidiscretize` returns a single
# `ODEProblem` with this fused right-hand side instead of a `SplitODEProblem`, and it
# needs no parabolic cache (no gradient arrays).

const SemidiscretizationHyperbolicParabolicMacroCellFV = SemidiscretizationHyperbolicParabolic{<:Any,
                                                                                               <:Any,
                                                                                               <:Any,
                                                                                               <:Any,
                                                                                               <:Any,
                                                                                               <:Any,
                                                                                               <:Any,
                                                                                               <:Any,
                                                                                               <:MacroCellFV}

# (one method per mesh type, since a `Union` of the mesh types would be ambiguous with
# the methods for `DG`)
function create_cache_parabolic(mesh::TreeMesh{1},
                                equations_hyperbolic::AbstractEquations,
                                dg::MacroCellFV, n_elements, uEltype)
    return NamedTuple()
end

function create_cache_parabolic(mesh::TreeMesh{2},
                                equations_hyperbolic::AbstractEquations,
                                dg::MacroCellFV, n_elements, uEltype)
    return NamedTuple()
end

function semidiscretize(semi::SemidiscretizationHyperbolicParabolicMacroCellFV, tspan;
                        jac_prototype_parabolic = nothing, colorvec_parabolic = nothing,
                        reset_threads = true)
    (jac_prototype_parabolic === nothing && colorvec_parabolic === nothing) ||
        throw(ArgumentError("`MacroCellFV` computes the advective and the diffusive " *
                            "terms in one right-hand side; split ODE problems (e.g., " *
                            "for IMEX methods) are not supported"))
    check_macrocellfv_parabolic(semi)
    if reset_threads
        Polyester.reset_threads!()
    end

    tspan = convert_tspan(semi, tspan)
    u0_ode = compute_coefficients(first(tspan), semi)
    iip = true
    specialize = SciMLBase.FullSpecialize
    return ODEProblem{iip, specialize}(rhs_macrocellfv!, u0_ode, tspan, semi)
end

function check_macrocellfv_parabolic(semi)
    @unpack boundary_conditions, boundary_conditions_parabolic, source_terms_parabolic, solver = semi
    has_gradients(solver.volume_integral) isa True ||
        throw(ArgumentError("diffusion with `MacroCellFV` needs face derivatives: use a " *
                            "`CombinedScheme` with `GradientCentral` as `reconstruction` " *
                            "or in `schemes`"))
    check_macrocellfv_parabolic(semi.mesh, semi.equations_parabolic, solver)
    source_terms_parabolic === nothing ||
        throw(ArgumentError("`MacroCellFV` does not support `source_terms_parabolic`; " *
                            "use `source_terms`"))
    # The ghost subcells of the hyperbolic boundary conditions also give the face
    # derivatives at the boundaries
    boundary_conditions == boundary_conditions_parabolic ||
        throw(ArgumentError("`MacroCellFV` imposes the boundary conditions by the ghost " *
                            "subcells of the hyperbolic boundary conditions; the " *
                            "parabolic boundary conditions must be the same"))
    return nothing
end

check_macrocellfv_parabolic(mesh::TreeMesh{1}, equations_parabolic, solver) = nothing

# In 2D, the isotropic diffusion (whose flux needs no derivatives along the faces) and
# the compressible Navier-Stokes equations with primitive gradient variables
check_macrocellfv_parabolic(mesh::TreeMesh{2}, ::LaplaceDiffusion2D, solver) = nothing
# The derivatives along the faces (see `calc_tangential_derivatives!`) read the subcells
# `g + 1 - R, ..., g + n + R` of the lines (radius `R` of the derivative stencils), which
# must have been filled and transformed, and their one-sided stencils at the ends of the
# face strips need `2 R + 2` lines
function check_macrocellfv_parabolic(mesh::TreeMesh{2},
                                     ::CompressibleNavierStokesDiffusion2D{GradientVariablesPrimitive},
                                     solver)
    volume_integral = solver.volume_integral
    radius = tangential_radius(volume_integral)
    radius in (1, 2) ||
        throw(ArgumentError("`MacroCellFV` with the compressible Navier-Stokes " *
                            "equations in 2D supports `GradientCentral` of order 2 " *
                            "and 4 so far"))
    n = nnodes(solver)
    n >= 2 * radius + 2 ||
        throw(ArgumentError("the derivatives along the faces need " *
                            "`n_cells >= $(2 * radius + 2)`"))
    # the first filled cell of the lines of the point values is 3, otherwise 2
    a_first = has_point_values(volume_integral.moment_inversion) isa True ? 3 : 2
    nhalo(solver) >= radius + a_first - 1 ||
        throw(ArgumentError("the derivatives along the faces need " *
                            "$(radius + a_first - 1) halo subcells; use a " *
                            "reconstruction with a wider stencil"))
    return nothing
end
function check_macrocellfv_parabolic(mesh::TreeMesh{2}, equations_parabolic, solver)
    throw(ArgumentError("`MacroCellFV` in 2D supports only `LaplaceDiffusion2D` and " *
                        "`CompressibleNavierStokesDiffusion2D` with " *
                        "`GradientVariablesPrimitive` so far"))
end

# The fused right-hand side of the advective and diffusive terms
function rhs_macrocellfv!(du_ode, u_ode,
                          semi::SemidiscretizationHyperbolicParabolicMacroCellFV, t)
    @unpack mesh, equations, equations_parabolic, boundary_conditions, source_terms, solver, cache = semi

    u = wrap_array(u_ode, mesh, equations, solver, cache)
    du = wrap_array(du_ode, mesh, equations, solver, cache)

    time_start = time_ns()
    @trixi_timeit timer() "rhs!" rhs_macrocellfv!(du, u, t, mesh, equations,
                                                  equations_parabolic,
                                                  boundary_conditions, source_terms,
                                                  solver, cache)
    runtime = time_ns() - time_start
    # The performance counters of the hyperbolic and the parabolic parts: the total time
    # in the first one
    put!(semi.performance_counter.counters[1], runtime)
    put!(semi.performance_counter.counters[2], 0)

    return nothing
end

for Mesh in (TreeMesh{1}, TreeMesh{2})
    @eval function rhs_parabolic!(backend::Nothing, du, u, t, mesh::$Mesh,
                                  equations_parabolic::AbstractEquationsParabolic,
                                  boundary_conditions_parabolic, source_terms_parabolic,
                                  dg::MacroCellFV, parabolic_scheme, cache,
                                  cache_parabolic)
        throw(ArgumentError("`MacroCellFV` has no separate parabolic right-hand side; " *
                            "use the `ODEProblem` of `semidiscretize`"))
    end
end

# Diffusive time step: `h^2 / (2 d ν factor)` with the subcell width `h`, the number of
# space dimensions `d`, the largest diffusivity `ν`, and the stencil factor of the face
# derivatives and the moment inversion (see `diffusion_dt_factor`); with
# `cfl_parabolic = 1`, the explicit Euler method is stable for `u_t = ν Δu` with periodic
# boundaries (the eigenvalues of the directions add up).
function max_dt_macrocellfv(u, t, mesh, constant_diffusivity::True, equations,
                            equations_parabolic::AbstractEquationsParabolic,
                            dg::MacroCellFV, cache)
    max_scaled_diffusivity = nextfloat(zero(t))
    diffusivity = max_diffusivity(equations_parabolic)
    for element in eachelement(dg, cache)
        inv_h = inv_subcell_width(mesh, dg, cache, element)
        max_scaled_diffusivity = Base.max(max_scaled_diffusivity,
                                          inv_h^2 * diffusivity)
    end
    return 1 / (2 * ndims(mesh) * dg.volume_integral.diffusion_dt_factor *
            max_scaled_diffusivity)
end

function max_dt_macrocellfv(u, t, mesh, constant_diffusivity::False, equations,
                            equations_parabolic::AbstractEquationsParabolic,
                            dg::MacroCellFV, cache)
    max_scaled_diffusivity = nextfloat(zero(t))
    # (outside of the `@batch` loop, which does not support closures)
    nodes = CartesianIndices(ntuple(_ -> nnodes(dg), Val(ndims(mesh))))
    @batch reduction=(max, max_scaled_diffusivity) for element in eachelement(dg, cache)
        max_diffusivity_ = zero(max_scaled_diffusivity)
        for node in nodes
            u_node = get_node_vars(u, equations, dg, Tuple(node)..., element)
            max_diffusivity_ = Base.max(max_diffusivity_,
                                        max_diffusivity(u_node, equations_parabolic))
        end
        inv_h = inv_subcell_width(mesh, dg, cache, element)
        max_scaled_diffusivity = Base.max(max_scaled_diffusivity,
                                          inv_h^2 * max_diffusivity_)
    end
    return 1 / (2 * ndims(mesh) * dg.volume_integral.diffusion_dt_factor *
            max_scaled_diffusivity)
end

# (one method per mesh type, since a `Union` of the mesh types would be ambiguous with
# the methods for `DG`)
for Mesh in (TreeMesh{1}, TreeMesh{2}), ConstantDiffusivity in (True, False)
    @eval function max_dt(u, t, mesh::$Mesh, constant_diffusivity::$ConstantDiffusivity,
                          equations,
                          equations_parabolic::AbstractEquationsParabolic,
                          dg::MacroCellFV, cache)
        return max_dt_macrocellfv(u, t, mesh, constant_diffusivity, equations,
                                  equations_parabolic, dg, cache)
    end
end

# For the isotropic `LaplaceDiffusion2D`, the flux in the direction `orientation`
# depends only on the derivative along this direction, so the tangential entry of the
# gradients is zero (no derivatives along the faces are computed).
@inline function macrocell_gradients(dw_face, tw_face::Nothing, orientation,
                                     ::LaplaceDiffusion2D)
    zero_gradient = zero(dw_face)
    if orientation == 1
        return (dw_face, zero_gradient)
    else
        return (zero_gradient, dw_face)
    end
end

# The compressible Navier-Stokes equations need the derivatives along the faces
# `tw_face` (see `calc_tangential_derivatives!`)
@inline function macrocell_gradients(dw_face, tw_face, orientation,
                                     ::CompressibleNavierStokesDiffusion2D)
    if orientation == 1
        return (dw_face, tw_face)
    else
        return (tw_face, dw_face)
    end
end

# `True()` if the diffusive flux in one direction needs the derivatives along the
# faces normal to it
have_tangential_derivatives(equations_parabolic) = False()
have_tangential_derivatives(::CompressibleNavierStokesDiffusion2D) = True()

# Face transformations for the compressible Navier-Stokes equations with the
# reconstruction variables `(rho, v1, [v2,] p, R T)` of `cons2prim_temperature_pressure`
# (see `face2flux_navier_stokes_primitive`): the diffusive flux takes the primitive
# gradient variables `(rho, v1, [v2,] T)`, with `T = (R T) / R` from the reconstruction
# variable `R T`
const Face2FluxNavierStokesPrimitive = Union{typeof(face2flux_navier_stokes_primitive),
                                             typeof(face2flux_navier_stokes_prim2cons)}

@inline function macrocell_face2flux(face2flux::Face2FluxNavierStokesPrimitive, w_ll,
                                     w_rr, w_face, gradients, equations,
                                     equations_parabolic::Union{CompressibleNavierStokesDiffusion1D{GradientVariablesPrimitive},
                                                                CompressibleNavierStokesDiffusion2D{GradientVariablesPrimitive}})
    inv_R = inv(equations_parabolic.R)
    u_diffusive = navier_stokes_gradient_variables(w_face, inv_R, equations)
    gradients_diffusive = map(gradient -> navier_stokes_gradient_variables(gradient,
                                                                           inv_R,
                                                                           equations),
                              gradients)
    return (face2flux(w_ll, equations), face2flux(w_rr, equations), u_diffusive,
            gradients_diffusive)
end

# `(rho, v1, [v2,] T)` (or their derivatives) from `(rho, v1, [v2,] p, R T)`
@inline function navier_stokes_gradient_variables(w, inv_R,
                                                  ::CompressibleEulerEquations1D)
    return SVector(w[1], w[2], w[4] * inv_R)
end
@inline function navier_stokes_gradient_variables(w, inv_R,
                                                  ::CompressibleEulerEquations2D)
    return SVector(w[1], w[2], w[3], w[5] * inv_R)
end

# Conservative reconstruction variables (`face2flux_navier_stokes_cons2prim`): the
# primitive variables `(rho, v1, v2, T)` of the face state, and their derivatives from
# those of the conservative variables `(rho, m1, m2, E)` by the chain rule,
#   v = m / rho,           ∂v = (∂m - v ∂rho) / rho,
#   p = (γ - 1) (E - rho |v|^2 / 2),
#                          ∂p = (γ - 1) (∂E - |v|^2 ∂rho / 2 - rho v ⋅ ∂v),
#   T = p / (R rho),       ∂T = (∂p - p / rho ∂rho) / (R rho)
@inline function macrocell_face2flux(::typeof(face2flux_navier_stokes_cons2prim), w_ll,
                                     w_rr, w_face, gradients,
                                     equations::CompressibleEulerEquations2D,
                                     equations_parabolic::CompressibleNavierStokesDiffusion2D{GradientVariablesPrimitive})
    rho, v1, v2, p = cons2prim(w_face, equations)
    inv_rho = inv(rho)
    R = equations_parabolic.R
    inv_R_rho = inv(R * rho)
    u_diffusive = SVector(rho, v1, v2, p * inv_R_rho)
    gradients_diffusive = map(gradients) do dw
        d_rho, d_m1, d_m2, d_E = dw
        d_v1 = (d_m1 - v1 * d_rho) * inv_rho
        d_v2 = (d_m2 - v2 * d_rho) * inv_rho
        d_p = (equations.gamma - 1) *
              (d_E - 0.5f0 * (v1^2 + v2^2) * d_rho - rho * (v1 * d_v1 + v2 * d_v2))
        d_T = (d_p - p * inv_rho * d_rho) * inv_R_rho
        return SVector(d_rho, d_v1, d_v2, d_T)
    end
    return w_ll, w_rr, u_diffusive, gradients_diffusive
end
end # @muladd
