# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# `MacroCellFV` on the 2D `TreeMesh`: the faces of each macro-cell are computed line by
# line with the line kernels of `line_kernels.jl`, first along the x-lines (the faces
# normal to x), then along the y-lines.
#
# Each thread owns two pads, one per direction, which hold the lines of the macro-cell
# with their halos from the face neighbors:
#
#   pad_x[a, v, j]   x-line `j`: subcell `(i, j)` at `a = i + g`, `i = 1 - g, ..., n + g`
#   pad_y[a, v, i]   y-line `i`: subcell `(i, j)` at `a = j + g`, `j = 1 - g, ..., n + g`
#
# Each line `pad_x[:, :, j]` is stored like the pad of the 1D solver (each variable
# contiguous along the line), so that the same vectorized line kernels serve both
# directions. The y-lines are stored transposed for this reason. The pads hold only the
# subcells of the macro-cell and the halos along the lines (the face strips of the
# neighbors), no corner subcells; the subcells of the macro-cell are stored in both pads.
#
# Each element computes the faces inside the macro-cell and its faces at +x and +y. The
# fluxes at its faces at -x and -y are computed by its neighbors and added by
# `finish_macrocell_faces!` after all elements.
#
# Only periodic meshes without mortars and the moment inversion `MomentInversionIdentity`
# are supported so far. With the identity, the reconstruction and the face derivatives
# act on the subcell averages along each line, which gives the face averages of the
# states and of the normal derivatives (exact to the order of the stencils for linear
# fluxes, second order for nonlinear fluxes).
#
# The neighbors of each element are stored in `neighbor_elements[direction, element]`
# (`direction = 1, 2, 3, 4` for -x, +x, -y, +y).
function create_cache_volume_integral(mesh::TreeMesh{2}, equations,
                                      volume_integral::VolumeIntegralMacroCellFV,
                                      dg::MacroCellFV, cache_containers, uEltype)
    n = nnodes(dg)
    g = nhalo(dg)
    g <= n ||
        throw(ArgumentError("the reconstruction needs $g halo subcells, which " *
                            "requires `n_cells >= $g`"))
    volume_integral.moment_inversion isa MomentInversionIdentity ||
        throw(ArgumentError("`MacroCellFV` in 2D supports only " *
                            "`MomentInversionIdentity` so far"))

    @unpack elements, interfaces, boundaries, mortars = cache_containers
    (nboundaries(boundaries) == 0 && nmortars(mortars) == 0) ||
        throw(ArgumentError("`MacroCellFV` in 2D supports only periodic meshes " *
                            "without mortars so far"))

    n_pad = npadvariables(volume_integral, equations)
    pad_x_threaded = [Array{uEltype, 3}(undef, n + 2 * g, n_pad, n)
                      for _ in 1:Threads.maxthreadid()]
    pad_y_threaded = [Array{uEltype, 3}(undef, n + 2 * g, n_pad, n)
                      for _ in 1:Threads.maxthreadid()]
    n_recon = nreconvariables(volume_integral, equations)
    n_derivative = has_gradients(volume_integral) isa True ? n + 2 : 0
    line_buffers_threaded = [(;
                              w_minus = Matrix{uEltype}(undef, n + 2, n_recon),
                              w_plus = Matrix{uEltype}(undef, n + 2, n_recon),
                              dw_minus = Matrix{uEltype}(undef, n_derivative, n_recon),
                              dw_plus = Matrix{uEltype}(undef, n_derivative, n_recon),
                              flux = Matrix{uEltype}(undef, n + 1,
                                                     nvariables(equations)))
                             for _ in 1:Threads.maxthreadid()]

    n_elements = nelements(elements)
    neighbor_elements = zeros(Int, 4, n_elements)
    for interface in 1:ninterfaces(interfaces)
        # The element at -x (-y) and +x (+y) of the interface
        minus_element = interfaces.neighbor_ids[1, interface]
        plus_element = interfaces.neighbor_ids[2, interface]
        orientation = interfaces.orientations[interface]
        neighbor_elements[2 * orientation, minus_element] = plus_element
        neighbor_elements[2 * orientation - 1, plus_element] = minus_element
    end
    all(!iszero, neighbor_elements) ||
        throw(ArgumentError("`MacroCellFV` in 2D supports only periodic meshes so far"))

    # Gauss-Legendre quadrature on [-1, 1] for the subcell averages of the initial
    # condition and of the source terms
    nodes, weights = gauss_nodes_weights(volume_integral.initial_condition_nodes,
                                         real(dg))
    subcell_quadrature = (; nodes, weights)

    return (; pad_x_threaded, pad_y_threaded, line_buffers_threaded,
            neighbor_elements, subcell_quadrature)
end

function resize_cache_volume_integral!(cache, mesh::TreeMesh{2},
                                       volume_integral::VolumeIntegralMacroCellFV,
                                       dg::MacroCellFV, new_size)
    # Only check the size here; AMR is not supported yet.
    @assert size(cache.neighbor_elements, 2) == new_size
    return nothing
end

function rhs_hyperbolic!(backend::Nothing, du, u, t, mesh::TreeMesh{2}, equations,
                         boundary_conditions, source_terms::Source,
                         dg::MacroCellFV, cache) where {Source}
    rhs_macrocellfv!(du, u, t, mesh, equations, nothing, boundary_conditions,
                     source_terms, dg, cache)
    return nothing
end

# Right-hand side of the advective (`equations_parabolic === nothing`) or the advective
# and diffusive terms (see `macrocellfv_parabolic.jl`), plus the source terms
function rhs_macrocellfv!(du, u, t, mesh::TreeMesh{2}, equations, equations_parabolic,
                          boundary_conditions, source_terms::Source,
                          dg::MacroCellFV, cache) where {Source}
    @trixi_timeit timer() "subcell fluxes and divergence" begin
        @threaded for element in eachelement(dg, cache)
            pad_x = cache.pad_x_threaded[Threads.threadid()]
            pad_y = cache.pad_y_threaded[Threads.threadid()]
            line_buffers = cache.line_buffers_threaded[Threads.threadid()]
            calc_macrocell_element!(du, pad_x, pad_y, line_buffers, u, t, element,
                                    source_terms, mesh, equations,
                                    equations_parabolic, dg.volume_integral, dg, cache)
        end
    end

    @trixi_timeit timer() "interface fluxes" begin
        @threaded for element in eachelement(dg, cache)
            finish_macrocell_faces!(du, element, mesh, equations, dg, cache)
        end
    end

    return nothing
end

# All faces of `element` except those at -x and -y: the faces inside the macro-cell
# and the faces at +x and +y (stored for both adjacent elements in
# `surface_flux_values`). The flux differences of the x-lines are stored in `du`, those
# of the y-lines are added; the subcells next to the faces at -x and -y miss the fluxes
# at these faces until `finish_macrocell_faces!`.
function calc_macrocell_element!(du, pad_x, pad_y, line_buffers, u, t, element,
                                 source_terms, mesh::TreeMesh{2}, equations,
                                 equations_parabolic, volume_integral,
                                 dg::MacroCellFV, cache)
    @unpack surface_flux_values = cache.elements
    @unpack neighbor_elements = cache
    @unpack flux = line_buffers
    @unpack center2recon = volume_integral
    g = nhalo(dg)
    n = nnodes(dg)
    n_recon = Val(nreconvariables(volume_integral, equations))
    inv_h = inv_subcell_width(mesh, dg, cache, element)

    gather_pads!(pad_x, pad_y, u, element, equations, dg, cache)

    # x-lines: the faces normal to x. The first filled pad cell is `a = 2`, since the
    # face at -x is computed by the neighbor at -x, so the reconstruction starts at the
    # subcell 1.
    right_element = neighbor_elements[2, element]
    for j in eachnode(dg)
        line = view(pad_x, :, :, j)
        pad_center2recon!(line, 2, n + 2 * g, center2recon, equations)
        reconstruct_line!(line_buffers, line, volume_integral, 1, n + 1, g, n_recon)
        calc_line_fluxes!(flux, line_buffers, 2, n + 1, inv_h, 1, equations,
                          equations_parabolic, volume_integral)
        @inbounds for v in eachvariable(equations)
            # The flux at the face at -x is added by `finish_macrocell_faces!`
            du[v, 1, j, element] = -inv_h * flux[2, v]
            for i in 2:n
                du[v, i, j, element] = -inv_h * (flux[i + 1, v] - flux[i, v])
            end
            surface_flux_values[v, j, 2, element] = flux[n + 1, v]
            surface_flux_values[v, j, 1, right_element] = flux[n + 1, v]
        end
    end

    # y-lines: the faces normal to y
    upper_element = neighbor_elements[4, element]
    for i in eachnode(dg)
        line = view(pad_y, :, :, i)
        pad_center2recon!(line, 2, n + 2 * g, center2recon, equations)
        reconstruct_line!(line_buffers, line, volume_integral, 1, n + 1, g, n_recon)
        calc_line_fluxes!(flux, line_buffers, 2, n + 1, inv_h, 2, equations,
                          equations_parabolic, volume_integral)
        @inbounds for v in eachvariable(equations)
            du[v, i, 1, element] -= inv_h * flux[2, v]
            for j in 2:n
                du[v, i, j, element] -= inv_h * (flux[j + 1, v] - flux[j, v])
            end
            surface_flux_values[v, i, 4, element] = flux[n + 1, v]
            surface_flux_values[v, i, 3, upper_element] = flux[n + 1, v]
        end
    end

    add_macrocell_source_terms!(du, u, t, source_terms, element, inv_h, mesh, equations,
                                dg, cache)

    return nothing
end

# Add the fluxes at the faces at -x and -y of `element`, computed by its neighbors
@inline function finish_macrocell_faces!(du, element, mesh::TreeMesh{2}, equations,
                                         dg::MacroCellFV, cache)
    @unpack surface_flux_values = cache.elements
    inv_h = inv_subcell_width(mesh, dg, cache, element)
    @inbounds for k in eachnode(dg), v in eachvariable(equations)
        du[v, 1, k, element] += inv_h * surface_flux_values[v, k, 1, element]
        du[v, k, 1, element] += inv_h * surface_flux_values[v, k, 3, element]
    end
    return nothing
end

# The subcells of `element` and the halos from the face neighbors in both pads (in
# conservative variables). At -x and -y, `g - 1` halo subcells suffice, since the faces
# there are computed by the neighbors; the first pad cell is not filled.
@inline function gather_pads!(pad_x, pad_y, u, element, equations, dg, cache)
    @unpack neighbor_elements = cache
    g = nhalo(dg)
    n = nnodes(dg)
    left, right, lower, upper = (neighbor_elements[1, element],
                                 neighbor_elements[2, element],
                                 neighbor_elements[3, element],
                                 neighbor_elements[4, element])

    @inbounds for j in eachnode(dg), v in eachvariable(equations)
        for i in eachnode(dg)
            pad_x[i + g, v, j] = u[v, i, j, element]
        end
        for k in 1:(g - 1)
            pad_x[g + 1 - k, v, j] = u[v, n + 1 - k, j, left]
        end
        for k in 1:g
            pad_x[n + g + k, v, j] = u[v, k, j, right]
        end
    end

    @inbounds for i in eachnode(dg), v in eachvariable(equations)
        for j in eachnode(dg)
            pad_y[j + g, v, i] = u[v, i, j, element]
        end
        for k in 1:(g - 1)
            pad_y[g + 1 - k, v, i] = u[v, i, n + 1 - k, lower]
        end
        for k in 1:g
            pad_y[n + g + k, v, i] = u[v, i, k, upper]
        end
    end

    return nothing
end

# Source terms: the averages of the source terms over the subcells (tensor-product
# Gauss-Legendre quadrature of `initial_condition_nodes` per direction, with the subcell
# average as state)
@inline function add_macrocell_source_terms!(du, u, t, source_terms::Nothing, element,
                                             inv_h, mesh::TreeMesh{2}, equations,
                                             dg::MacroCellFV, cache)
    return nothing
end

@inline function add_macrocell_source_terms!(du, u, t, source_terms, element, inv_h,
                                             mesh::TreeMesh{2}, equations,
                                             dg::MacroCellFV, cache)
    add_macrocell_source_terms_2d!(du, u, t, source_terms, element, inv_h, equations,
                                   dg, cache)
    return nothing
end

@noinline function add_macrocell_source_terms_2d!(du, u, t, source_terms, element,
                                                  inv_h, equations, dg, cache)
    h = inv(inv_h)
    for j in eachnode(dg), i in eachnode(dg)
        u_local = get_node_vars(u, equations, dg, i, j, element)
        x_c = get_node_coords(cache.elements.node_coordinates, equations, dg, i, j,
                              element)
        source = subcell_average(cache.subcell_quadrature, x_c, h) do x
            return source_terms(u_local, x, t, equations)
        end
        add_to_node_vars!(du, source, equations, dg, i, j, element)
    end
    return nothing
end

# Average of `func(x)` over the square subcell with center `x_c` and width `h` by the
# tensor-product Gauss-Legendre `quadrature` on [-1, 1]
@inline function subcell_average(func::F, quadrature, x_c, h) where {F}
    @unpack nodes, weights = quadrature
    average = zero(func(x_c))
    for qj in eachindex(nodes), qi in eachindex(nodes)
        x = SVector(x_c[1] + 0.5f0 * h * nodes[qi], x_c[2] + 0.5f0 * h * nodes[qj])
        average = average + 0.25f0 * weights[qi] * weights[qj] * func(x)
    end
    return average
end

# (a separate function: a closure inside the `@threaded` loop is not supported by
# Polyester.jl)
@inline function initial_condition_subcell_average(func::F, quadrature, x_c, h, t,
                                                   equations) where {F}
    return subcell_average(quadrature, x_c, h) do x
        return func(x, t, equations)
    end
end

# Initial condition: point values at the subcell centers (as for all DG solvers) or
# subcell averages by Gauss-Legendre quadrature
function compute_coefficients!(backend::Nothing, u, func, t, mesh::TreeMesh{2},
                               equations, dg::MacroCellFV, cache)
    if dg.volume_integral.initial_condition_nodes == 1
        return invoke(compute_coefficients!,
                      Tuple{Nothing, typeof(u), typeof(func), typeof(t),
                            Union{AbstractMesh{2}, AbstractMesh{3}}, typeof(equations),
                            DG,
                            typeof(cache)},
                      backend, u, func, t, mesh, equations, dg, cache)
    end
    @threaded for element in eachelement(dg, cache)
        h = 1 / inv_subcell_width(mesh, dg, cache, element)
        for j in eachnode(dg), i in eachnode(dg)
            x_c = get_node_coords(cache.elements.node_coordinates, equations, dg, i, j,
                                  element)
            u_avg = initial_condition_subcell_average(func, cache.subcell_quadrature,
                                                      x_c, h, t, equations)
            set_node_vars!(u, u_avg, equations, dg, i, j, element)
        end
    end

    return nothing
end
end # @muladd
