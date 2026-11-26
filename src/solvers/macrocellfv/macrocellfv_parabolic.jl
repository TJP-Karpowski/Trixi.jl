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

function create_cache_parabolic(mesh::TreeMesh{1},
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

function rhs_parabolic!(backend::Nothing, du, u, t, mesh::TreeMesh{1},
                        equations_parabolic::AbstractEquationsParabolic,
                        boundary_conditions_parabolic, source_terms_parabolic,
                        dg::MacroCellFV, parabolic_scheme, cache, cache_parabolic)
    throw(ArgumentError("`MacroCellFV` has no separate parabolic right-hand side; use " *
                        "the `ODEProblem` of `semidiscretize`"))
end

# Diffusive time step: `h^2 / (2 ν factor)` with the subcell width `h`, the largest
# diffusivity `ν`, and the stencil factor of the face derivatives and the moment
# inversion (see `diffusion_dt_factor`); with `cfl_parabolic = 1`, the explicit Euler
# method is stable for `u_t = ν u_xx` with periodic boundaries.
function max_dt(u, t, mesh::TreeMesh{1}, constant_diffusivity::True, equations,
                equations_parabolic::AbstractEquationsParabolic, dg::MacroCellFV, cache)
    max_scaled_diffusivity = nextfloat(zero(t))
    diffusivity = max_diffusivity(equations_parabolic)
    for element in eachelement(dg, cache)
        inv_h = inv_subcell_width(mesh, dg, cache, element)
        max_scaled_diffusivity = Base.max(max_scaled_diffusivity,
                                          inv_h^2 * diffusivity)
    end
    return 1 / (2 * dg.volume_integral.diffusion_dt_factor * max_scaled_diffusivity)
end

function max_dt(u, t, mesh::TreeMesh{1}, constant_diffusivity::False, equations,
                equations_parabolic::AbstractEquationsParabolic, dg::MacroCellFV, cache)
    max_scaled_diffusivity = nextfloat(zero(t))
    @batch reduction=(max, max_scaled_diffusivity) for element in eachelement(dg, cache)
        max_diffusivity_ = zero(max_scaled_diffusivity)
        for i in eachnode(dg)
            u_node = get_node_vars(u, equations, dg, i, element)
            max_diffusivity_ = Base.max(max_diffusivity_,
                                        max_diffusivity(u_node, equations_parabolic))
        end
        inv_h = inv_subcell_width(mesh, dg, cache, element)
        max_scaled_diffusivity = Base.max(max_scaled_diffusivity,
                                          inv_h^2 * max_diffusivity_)
    end
    return 1 / (2 * dg.volume_integral.diffusion_dt_factor * max_scaled_diffusivity)
end

# Face transformation for the compressible Navier-Stokes equations (see
# `face2flux_navier_stokes_primitive`): the diffusive flux takes the primitive gradient
# variables `(rho, v1, T)`, with `T = (R T) / R` from the reconstruction variable `R T`
@inline function macrocell_face2flux(::typeof(face2flux_navier_stokes_primitive), w_ll,
                                     w_rr, w_face, dw_face, equations,
                                     equations_parabolic::CompressibleNavierStokesDiffusion1D{GradientVariablesPrimitive})
    inv_R = inv(equations_parabolic.R)
    u_diffusive = SVector(w_face[1], w_face[2], w_face[4] * inv_R)
    gradient = SVector(dw_face[1], dw_face[2], dw_face[4] * inv_R)
    return (face2flux_navier_stokes_primitive(w_ll, equations),
            face2flux_navier_stokes_primitive(w_rr, equations), u_diffusive, gradient)
end
end # @muladd
