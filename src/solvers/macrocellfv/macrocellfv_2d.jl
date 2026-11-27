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
    @unpack moment_inversion = volume_integral
    check_moment_inversion_2d(moment_inversion, n)

    @unpack elements, interfaces, boundaries, mortars = cache_containers
    (nboundaries(boundaries) == 0 && nmortars(mortars) == 0) ||
        throw(ArgumentError("`MacroCellFV` in 2D supports only periodic meshes " *
                            "without mortars so far"))

    # The pads of the averages hold `m` transverse halo lines on each side for the
    # moment inversion (only their subcells inside the macro-cell are filled)
    m = nhalo(moment_inversion)
    n_pad = npadvariables(volume_integral, equations)
    pad_x_threaded = [Array{uEltype, 3}(undef, n + 2 * g, n_pad, n + 2 * m)
                      for _ in 1:Threads.maxthreadid()]
    pad_y_threaded = [Array{uEltype, 3}(undef, n + 2 * g, n_pad, n + 2 * m)
                      for _ in 1:Threads.maxthreadid()]
    # The pads of the point values, one per direction (empty without moment inversion
    # to point values)
    n_points = has_point_values(moment_inversion) isa True ? n : 0
    pad_points_x_threaded = [Array{uEltype, 3}(undef, n + 2 * g, n_pad, n_points)
                             for _ in 1:Threads.maxthreadid()]
    pad_points_y_threaded = [Array{uEltype, 3}(undef, n + 2 * g, n_pad, n_points)
                             for _ in 1:Threads.maxthreadid()]
    n_recon = nreconvariables(volume_integral, equations)
    n_derivative = has_gradients(volume_integral) isa True ? n + 2 : 0
    # With diffusion: the derivatives of the reconstruction variables across the lines
    # of each direction at the subcell centers, `tangential_x[a, v, line]` with the same
    # indices as the pads, and their values at the faces of a line, `tw_face[f, v]`
    # (used only if the diffusive flux needs them, see `have_tangential_derivatives`)
    n_tangential = has_gradients(volume_integral) isa True ? n : 0
    tangential_x_threaded = [Array{uEltype, 3}(undef, n + 2 * g, n_recon, n_tangential)
                             for _ in 1:Threads.maxthreadid()]
    tangential_y_threaded = [Array{uEltype, 3}(undef, n + 2 * g, n_recon, n_tangential)
                             for _ in 1:Threads.maxthreadid()]
    line_buffers_threaded = [(;
                              w_minus = Matrix{uEltype}(undef, n + 2, n_recon),
                              w_plus = Matrix{uEltype}(undef, n + 2, n_recon),
                              dw_minus = Matrix{uEltype}(undef, n_derivative, n_recon),
                              dw_plus = Matrix{uEltype}(undef, n_derivative, n_recon),
                              tw_face = Matrix{uEltype}(undef, n_derivative, n_recon))
                             for _ in 1:Threads.maxthreadid()]
    # The fluxes at the faces normal to one direction, `face_flux[f, v, line]`, and
    # their face averages (with point values; see `face_point2average!`)
    n_average = has_point_values(moment_inversion) isa True ? n : 0
    face_flux_threaded = [Array{uEltype, 3}(undef, n + 1, nvariables(equations), n)
                          for _ in 1:Threads.maxthreadid()]
    face_flux_average_threaded = [Array{uEltype, 3}(undef, n + 1,
                                                    nvariables(equations), n_average)
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

    return (; pad_x_threaded, pad_y_threaded, pad_points_x_threaded,
            pad_points_y_threaded, tangential_x_threaded, tangential_y_threaded,
            line_buffers_threaded, face_flux_threaded, face_flux_average_threaded,
            neighbor_elements, subcell_quadrature)
end

# The moment inversions supported in 2D: the identity and the fourth-order central
# inversion, whose one-sided closure at the ends of the halo strips needs four subcells
check_moment_inversion_2d(::MomentInversionIdentity, n) = nothing
function check_moment_inversion_2d(::MomentInversionCentral{1}, n)
    n >= 5 ||
        throw(ArgumentError("`MomentInversionCentral(order = 4)` in 2D requires " *
                            "`n_cells >= 5`"))
    return nothing
end
function check_moment_inversion_2d(moment_inversion, n)
    throw(ArgumentError("`MacroCellFV` in 2D supports only `MomentInversionIdentity` " *
                        "and `MomentInversionCentral(order = 4)` so far"))
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
            thread = Threads.threadid()
            buffers = (; pad_x = cache.pad_x_threaded[thread],
                       pad_y = cache.pad_y_threaded[thread],
                       pad_points_x = cache.pad_points_x_threaded[thread],
                       pad_points_y = cache.pad_points_y_threaded[thread],
                       tangential_x = cache.tangential_x_threaded[thread],
                       tangential_y = cache.tangential_y_threaded[thread],
                       line_buffers = cache.line_buffers_threaded[thread],
                       face_flux = cache.face_flux_threaded[thread],
                       face_flux_average = cache.face_flux_average_threaded[thread])
            calc_macrocell_element!(du, buffers, u, t, element, source_terms, mesh,
                                    equations, equations_parabolic,
                                    dg.volume_integral, dg, cache)
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
# `surface_flux_values`). The steps are
#   1. gather the subcells and the face strips of the neighbors into the pads of both
#      directions,
#   2. moment inversion (point values) and transformation to the reconstruction
#      variables of the lines of both pads,
#   3. if the diffusive flux needs them: the derivatives across the lines of each
#      direction (along the faces normal to it, see `calc_tangential_derivatives!`),
#   4. the faces of the x-lines and of the y-lines (`calc_macrocell_direction!`).
# The flux differences of the x-lines are stored in `du`, those of the y-lines are
# added; the subcells next to the faces at -x and -y miss the fluxes at these faces
# until `finish_macrocell_faces!`.
function calc_macrocell_element!(du, buffers, u, t, element, source_terms::Source,
                                 mesh::TreeMesh{2}, equations, equations_parabolic,
                                 volume_integral, dg::MacroCellFV,
                                 cache) where {Source}
    @unpack pad_x, pad_y, pad_points_x, pad_points_y = buffers
    inv_h = inv_subcell_width(mesh, dg, cache, element)

    gather_pads!(pad_x, pad_y, u, element, volume_integral.moment_inversion,
                 equations, dg, cache)
    # The pads read by the reconstruction (lines `line + offset`, filled cells
    # `a_first:a_last`)
    pads_x, offset, a_first, a_last = prepare_pad_2d!(pad_points_x, pad_x,
                                                      volume_integral, equations, dg)
    pads_y, _ = prepare_pad_2d!(pad_points_y, pad_y, volume_integral, equations, dg)

    tangential = have_tangential_derivatives(equations_parabolic)
    calc_tangential_derivatives!(buffers, pads_x, pads_y, offset, tangential,
                                 volume_integral, equations, dg)

    calc_macrocell_direction!(du, pads_x, offset, buffers.tangential_x, tangential,
                              buffers, element, inv_h, Val(1), equations,
                              equations_parabolic, volume_integral, dg, cache)
    calc_macrocell_direction!(du, pads_y, offset, buffers.tangential_y, tangential,
                              buffers, element, inv_h, Val(2), equations,
                              equations_parabolic, volume_integral, dg, cache)

    add_macrocell_source_terms!(du, u, t, source_terms, element, inv_h, mesh, equations,
                                dg, cache)

    return nothing
end

# Moment inversion of the lines of `pad` (point values in `pad_points`) and
# transformation of the filled cells to the reconstruction variables. Returns the pads
# read by the reconstruction, the offset of the line indices in them, and the range of
# their filled cells along the lines.
@inline function prepare_pad_2d!(pad_points, pad, volume_integral, equations, dg)
    @unpack moment_inversion, center2recon = volume_integral
    line_pads, offset, a_first, a_last = pad_moment_inversion_2d!(pad_points, pad,
                                                                  moment_inversion,
                                                                  nhalo(dg), equations,
                                                                  dg)
    for line in eachnode(dg)
        pad_center2recon!(view(line_pads, :, :, line + offset), a_first, a_last,
                          center2recon, equations)
    end
    return line_pads, offset, a_first, a_last
end

# The faces normal to the direction `orientation` of `element` (except those at its
# -x or -y side) from the lines of `line_pads` (reconstruction variables) in this
# direction:
#   1. face reconstruction, (if needed) the derivatives along the faces, and fluxes at
#      the faces of each line, stored in `face_flux[f, v, line]`,
#   2. with point values: conversion of the point fluxes at the face midpoints to face
#      averages along the faces (`face_point2average!`),
#   3. flux differences (stored for `orientation = 1`, added for `orientation = 2`) and
#      the fluxes at the faces at +x or +y for both adjacent elements.
# The reconstruction starts at the subcell 1, since the face at the minus side is
# computed by the neighbor there.
function calc_macrocell_direction!(du, line_pads, offset, tangential_derivatives,
                                   tangential, buffers, element, inv_h,
                                   orientation::Val{Orientation}, equations,
                                   equations_parabolic, volume_integral,
                                   dg::MacroCellFV, cache) where {Orientation}
    @unpack face_flux, face_flux_average = buffers
    line_buffers = line_buffers_2d(buffers.line_buffers, tangential)
    g = nhalo(dg)
    n = nnodes(dg)
    n_recon = Val(nreconvariables(volume_integral, equations))

    for line in eachnode(dg)
        line_pad = view(line_pads, :, :, line + offset)
        reconstruct_line!(line_buffers, line_pad, volume_integral, 1, n + 1, g, n_recon)
        tangential_face_derivatives!(line_buffers, tangential_derivatives, line,
                                     tangential, volume_integral, n_recon, g, n)
        calc_line_fluxes!(view(face_flux, :, :, line), line_buffers, 2, n + 1, inv_h,
                          Orientation, equations, equations_parabolic, volume_integral)
    end

    flux = face_point2average!(face_flux_average, face_flux,
                               volume_integral.moment_inversion, 2, n + 1, equations)
    add_flux_differences!(du, flux, element, inv_h, orientation, equations, dg)

    @unpack surface_flux_values = cache.elements
    plus_element = cache.neighbor_elements[2 * Orientation, element]
    @inbounds for line in eachnode(dg), v in eachvariable(equations)
        surface_flux_values[v, line, 2 * Orientation, element] = flux[n + 1, v, line]
        surface_flux_values[v, line, 2 * Orientation - 1, plus_element] = flux[n + 1, v,
                                                                               line]
    end

    return nothing
end

# The line buffers passed to the line kernels: with the derivatives along the faces
# only if the diffusive flux needs them (otherwise, `tw_face` is not read)
@inline line_buffers_2d(line_buffers, ::True) = line_buffers
@inline function line_buffers_2d(line_buffers, ::False)
    @unpack w_minus, w_plus, dw_minus, dw_plus = line_buffers
    return (; w_minus, w_plus, dw_minus, dw_plus)
end

# Flux differences of the faces normal to x (stored) or y (added); the subcells next to
# the faces at -x and -y miss the fluxes there until `finish_macrocell_faces!`
@inline function add_flux_differences!(du, flux, element, inv_h, ::Val{1}, equations,
                                       dg)
    n = nnodes(dg)
    @inbounds for j in eachnode(dg), v in eachvariable(equations)
        du[v, 1, j, element] = -inv_h * flux[2, v, j]
        for i in 2:n
            du[v, i, j, element] = -inv_h * (flux[i + 1, v, j] - flux[i, v, j])
        end
    end
    return nothing
end

@inline function add_flux_differences!(du, flux, element, inv_h, ::Val{2}, equations,
                                       dg)
    n = nnodes(dg)
    @inbounds for i in eachnode(dg), v in eachvariable(equations)
        du[v, i, 1, element] -= inv_h * flux[2, v, i]
        for j in 2:n
            du[v, i, j, element] -= inv_h * (flux[j + 1, v, i] - flux[j, v, i])
        end
    end
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
# there are computed by the neighbors; the first pad cell is not filled. For the moment
# inversion, the pads also hold `m = nhalo(moment_inversion)` transverse halo lines on
# each side, of which only the subcells next to the macro-cell (the face strips of the
# neighbors at -y and +y for the x-lines) are filled: the corner subcells are neither
# filled nor read.
@inline function gather_pads!(pad_x, pad_y, u, element, moment_inversion, equations,
                              dg, cache)
    @unpack neighbor_elements = cache
    g = nhalo(dg)
    n = nnodes(dg)
    m = nhalo(moment_inversion)
    left, right, lower, upper = (neighbor_elements[1, element],
                                 neighbor_elements[2, element],
                                 neighbor_elements[3, element],
                                 neighbor_elements[4, element])

    @inbounds for j in eachnode(dg), v in eachvariable(equations)
        for i in eachnode(dg)
            pad_x[i + g, v, j + m] = u[v, i, j, element]
        end
        for k in 1:(g - 1)
            pad_x[g + 1 - k, v, j + m] = u[v, n + 1 - k, j, left]
        end
        for k in 1:g
            pad_x[n + g + k, v, j + m] = u[v, k, j, right]
        end
    end
    @inbounds for k in 1:m, v in eachvariable(equations), i in eachnode(dg)
        pad_x[i + g, v, m + 1 - k] = u[v, i, n + 1 - k, lower]
        pad_x[i + g, v, n + m + k] = u[v, i, k, upper]
    end

    @inbounds for i in eachnode(dg), v in eachvariable(equations)
        for j in eachnode(dg)
            pad_y[j + g, v, i + m] = u[v, i, j, element]
        end
        for k in 1:(g - 1)
            pad_y[g + 1 - k, v, i + m] = u[v, i, n + 1 - k, lower]
        end
        for k in 1:g
            pad_y[n + g + k, v, i + m] = u[v, i, k, upper]
        end
    end
    @inbounds for k in 1:m, v in eachvariable(equations), j in eachnode(dg)
        pad_y[j + g, v, m + 1 - k] = u[v, n + 1 - k, j, left]
        pad_y[j + g, v, n + m + k] = u[v, k, j, right]
    end

    return nothing
end

#####################################################################
# Moment inversion in 2D (policy "local": in the pads of the element)

# Both methods return the pads read by the reconstruction, the offset of the line
# indices in them, and the range of their filled cells along the lines.
# Without moment inversion, the reconstruction reads the averages from the pad itself
# (without transverse halo lines)
@inline function pad_moment_inversion_2d!(pad_points, pad, ::MomentInversionIdentity, g,
                                          equations, dg)
    return pad, 0, 2, nnodes(dg) + 2 * g
end

# Fourth-order central moment inversion of the lines of `pad` (one direction):
# `u = ū - (δ_n² ū + δ_t² ū) / 24` with the central second differences `δ_n²` along the
# lines (normal to the faces) and `δ_t²` across the lines. The point values are computed
# in the pad cells read by the reconstruction, `a = 3, ..., n + 2 g - 1`.
# Across the lines, the subcells of the macro-cell use the transverse halo lines; the
# halo subcells (in the face strip of a neighbor) have no neighbors beyond the strip
# (no corner subcells), so they use the one-sided second difference
# `2 ū_1 - 5 ū_2 + 4 ū_3 - ū_4` (exact for cubic polynomials, offsets 0 to 3 into the
# strip) at the first and last line.
@inline function pad_moment_inversion_2d!(pad_points, pad,
                                          inversion::MomentInversionCentral{1}, g,
                                          equations, dg)
    n = nnodes(dg)
    a_first = 3
    a_last = n + 2 * g - 1
    invert_lines_2d!(pad_points, pad, inversion, a_first, a_last, g, n, equations)
    # Inadmissible point values are replaced by the averages
    @inbounds for line in 1:n, a in a_first:a_last
        u_center = get_pad_vars(view(pad_points, :, :, line), equations, a)
        if !macrocell_point_value_admissible(u_center, equations)
            for v in eachvariable(equations)
                pad_points[a, v, line] = pad[a, v, line + 1]
            end
        end
    end
    return pad_points, 0, a_first, a_last
end

@noinline function invert_lines_2d!(pad_points, pad, inversion, a_first, a_last, g, n,
                                    equations)
    c = convert(eltype(pad_points), inversion.coefficients[1]) # -1/24
    for line in 1:n, v in eachvariable(equations)
        l = line + 1 # the line in `pad`
        if 1 < line < n
            # All subcells: central second differences in both directions
            @inbounds @simd ivdep for a in a_first:a_last
                u_0 = pad[a, v, l]
                d_n = (pad[a - 1, v, l] + pad[a + 1, v, l]) - 2 * u_0
                d_t = (pad[a, v, l - 1] + pad[a, v, l + 1]) - 2 * u_0
                pad_points[a, v, line] = u_0 + c * (d_n + d_t)
            end
        else
            # The subcells of the macro-cell: central (transverse halo line)
            @inbounds @simd ivdep for a in (g + 1):(g + n)
                u_0 = pad[a, v, l]
                d_n = (pad[a - 1, v, l] + pad[a + 1, v, l]) - 2 * u_0
                d_t = (pad[a, v, l - 1] + pad[a, v, l + 1]) - 2 * u_0
                pad_points[a, v, line] = u_0 + c * (d_n + d_t)
            end
            # The halo subcells: one-sided across the lines, into the strip
            s = line == 1 ? 1 : -1
            for range in (a_first:g, (g + n + 1):a_last)
                @inbounds @simd ivdep for a in range
                    u_0 = pad[a, v, l]
                    d_n = (pad[a - 1, v, l] + pad[a + 1, v, l]) - 2 * u_0
                    d_t = closure_second_difference(u_0, pad[a, v, l + s],
                                                    pad[a, v, l + 2 * s],
                                                    pad[a, v, l + 3 * s],
                                                    pad[a, v, l + 4 * s])
                    pad_points[a, v, line] = u_0 + c * (d_n + d_t)
                end
            end
        end
    end
    return nothing
end

# One-sided second difference at offset 0 from the values at the offsets 0 to 4, exact
# for polynomials of degree 4: its error is O(h^5), one order more than the
# moment inversion, since the face derivatives of the point values differentiate the
# jump between the one-sided and the central inversion
@inline function closure_second_difference(u_0, u_1, u_2, u_3, u_4)
    RealT = typeof(u_0)
    return (convert(RealT, 35 / 12) * u_0 - convert(RealT, 26 / 3) * u_1) +
           (convert(RealT, 19 / 2) * u_2 - convert(RealT, 14 / 3) * u_3) +
           convert(RealT, 11 / 12) * u_4
end

#####################################################################
# Integration of the fluxes over the faces

# Without moment inversion, the face fluxes of the averages are face averages
@inline function face_point2average!(face_flux_average, face_flux,
                                     ::MomentInversionIdentity, f_first, f_last,
                                     equations)
    return face_flux
end

# With point values, the fluxes are point values at the face midpoints. Their averages
# over the faces are the fourth-order conversion along the faces normal to the same
# direction (across the lines), `F̄ = F + δ_t² F / 24` with the transverse second
# difference `δ_t²` (weights (1, 22, 1) / 24). At the first and last line, whose
# neighbor across the lines belongs to another macro-cell, the one-sided second
# difference `2 F_1 - 5 F_2 + 4 F_3 - F_4` (weights (26, -5, 4, -1) / 24) is used, so
# that no fluxes of the neighbors are needed. The element that computes a face at the
# boundary of the macro-cell converts it before it is stored for both elements.
@inline function face_point2average!(face_flux_average, face_flux,
                                     ::MomentInversionCentral, f_first, f_last,
                                     equations)
    face_point2average_4!(face_flux_average, face_flux, f_first, f_last, equations)
    return face_flux_average
end

@noinline function face_point2average_4!(flux_average, flux, f_first, f_last, equations)
    n = size(flux, 3)
    RealT = eltype(flux)
    w = inv(convert(RealT, 24))
    for line in 1:n, v in eachvariable(equations)
        if 1 < line < n
            @inbounds @simd ivdep for f in f_first:f_last
                F_0 = flux[f, v, line]
                d_t = (flux[f, v, line - 1] + flux[f, v, line + 1]) - 2 * F_0
                flux_average[f, v, line] = F_0 + w * d_t
            end
        else
            s = line == 1 ? 1 : -1
            @inbounds @simd ivdep for f in f_first:f_last
                F_0 = flux[f, v, line]
                d_t = (2 * F_0 - 5 * flux[f, v, line + s]) +
                      (4 * flux[f, v, line + 2 * s] - flux[f, v, line + 3 * s])
                flux_average[f, v, line] = F_0 + w * d_t
            end
        end
    end
    return nothing
end

#####################################################################
# Derivatives along the faces (tangential derivatives) for the diffusive fluxes
#
# The viscous fluxes of the compressible Navier-Stokes equations at a face normal to x
# need the y-derivatives of the velocity and the temperature there. They are the
# interpolation along the x-line of the y-derivatives at the subcell centers (I_x C_y):
# - the subcells of the macro-cell take the y-derivatives along the y-lines, whose pad
#   holds the halo subcells in y of the macro-cell (central stencils),
# - the halo subcells of the x-lines (in the face strips of the neighbors at -x and +x)
#   take them across the x-lines of the strip; there are no corner subcells, so at the
#   ends of the strip the stencils are one-sided, and one order more accurate than the
#   central ones, since the flux differences differentiate the jump between the
#   one-sided and the central derivatives once more.
# The order (2 or 4) is that of the `GradientCentral` stencils of the face derivatives
# (radius `R = order / 2`).

# The weights of the first derivative at offset 0 from the values at `offsets` (exact for
# polynomials of degree `length(offsets) - 1`, rational arithmetic)
function first_derivative_weights(offsets)
    m = length(offsets)
    A = [Rational{BigInt}(offset)^p for p in 0:(m - 1), offset in offsets]
    b = zeros(Rational{BigInt}, m)
    b[2] = 1
    return Tuple(Float64.(A \ b))
end

# Central weights of radius `R` (offsets `-R:R`), and the one-sided weights for the
# first `R` lines (line `j` with offsets `(1 - j):(2 R + 2 - j)`, exact for polynomials
# of degree `2 R + 1`)
const TANGENTIAL_CENTRAL_WEIGHTS = (first_derivative_weights(-1:1),
                                    first_derivative_weights(-2:2))
const TANGENTIAL_CLOSURE_WEIGHTS = ((first_derivative_weights(0:3),),
                                    (first_derivative_weights(0:5),
                                     first_derivative_weights(-1:4)))

# The radius of the central derivative stencils of the face derivatives (0 without)
@inline gradient_radius(scheme) = 0
@inline gradient_radius(::CombinedScheme{<:Any, GradientCentral{M}}) where {M} = M
@inline function tangential_radius(volume_integral::VolumeIntegralMacroCellFV)
    @unpack reconstruction, schemes = volume_integral
    schemes === nothing && return gradient_radius(reconstruction)
    return maximum(gradient_radius, schemes)
end

# Call `func(v)` for each reconstruction variable `v` with face derivatives (unrolled,
# so that the scheme of each variable is known at compile time)
@inline function foreach_gradient_variable(func::F, volume_integral,
                                           ::Val{NRecon}) where {F, NRecon}
    ntuple(Val(NRecon)) do v
        if has_gradient(pad_scheme(volume_integral, v)) isa True
            func(v)
        end
        return nothing
    end
    return nothing
end

@inline function calc_tangential_derivatives!(buffers, pads_x, pads_y, offset, ::False,
                                              volume_integral, equations, dg)
    return nothing
end

@inline function calc_tangential_derivatives!(buffers, pads_x, pads_y, offset, ::True,
                                              volume_integral, equations, dg)
    @unpack tangential_x, tangential_y = buffers
    g = nhalo(dg)
    n = nnodes(dg)
    radius = tangential_radius(volume_integral)
    n_recon = Val(nreconvariables(volume_integral, equations))
    foreach_gradient_variable(TangentialDerivatives(tangential_x, tangential_y, pads_x,
                                                    pads_y, offset, radius, g, n),
                              volume_integral, n_recon)
    return nothing
end

# The derivatives (per subcell width) of the variable `v` across the lines of `pad`
# (e.g., the y-derivatives for the x-lines) at the subcell centers
# `a = g + 2 - R, ..., g + n + R` of the lines, which the interpolation to the faces
# `2, ..., n + 1` reads. `pad_other` holds the lines of the other direction.
@noinline function tangential_derivatives_variable!(tangential, pad, pad_other, offset,
                                                    v, ::Val{R}, g, n) where {R}
    RealT = eltype(tangential)
    central = map(w -> convert(RealT, w), TANGENTIAL_CENTRAL_WEIGHTS[R])
    # The subcells of the macro-cell: central, along the lines of the other direction
    # (subcell `(a, line)` of `pad` is subcell `(g + line, a - g)` of `pad_other`)
    @inbounds for line_other in 1:n
        a = g + line_other
        for line in 1:n
            d = zero(RealT)
            for k in (-R):R
                d = d +
                    central[k + R + 1] * pad_other[g + line + k, v, line_other + offset]
            end
            tangential[a, v, line] = d
        end
    end
    # The halo subcells (face strips of the neighbors): across the lines of `pad`
    for line in 1:n
        if R < line <= n - R
            for range in ((g + 2 - R):g, (g + n + 1):(g + n + R))
                @inbounds @simd ivdep for a in range
                    d = zero(RealT)
                    for k in (-R):R
                        d = d + central[k + R + 1] * pad[a, v, line + k + offset]
                    end
                    tangential[a, v, line] = d
                end
            end
        else
            # One-sided: from the first or the last line of the strip
            j = line <= R ? line : n + 1 - line
            weights = map(w -> convert(RealT, w), TANGENTIAL_CLOSURE_WEIGHTS[R][j])
            sign = line <= R ? 1 : -1
            for range in ((g + 2 - R):g, (g + n + 1):(g + n + R))
                @inbounds @simd ivdep for a in range
                    d = zero(RealT)
                    for (q, w) in enumerate(weights)
                        # offsets `(1 - j):(2 R + 2 - j)`, mirrored at the last line
                        k = sign * (q - j)
                        d = d + sign * w * pad[a, v, line + k + offset]
                    end
                    tangential[a, v, line] = d
                end
            end
        end
    end
    return nothing
end

# The derivatives along the faces `2, ..., n + 1` of the line `line` (interpolation of
# the derivatives at the subcell centers along the line, of the order of the stencils),
# stored in `line_buffers.tw_face[f, v]`
@inline function tangential_face_derivatives!(line_buffers, tangential, line, ::False,
                                              volume_integral, n_recon, g, n)
    return nothing
end

@inline function tangential_face_derivatives!(line_buffers, tangential, line, ::True,
                                              volume_integral, n_recon, g, n)
    @unpack tw_face = line_buffers
    radius = tangential_radius(volume_integral)
    foreach_gradient_variable(InterpolateTangential(tw_face, tangential, line, radius,
                                                    g,
                                                    n),
                              volume_integral, n_recon)
    return nothing
end

# (callable structs instead of closures, which captured variables in boxes)
struct InterpolateTangential{TW, T}
    tw_face::TW
    tangential::T
    line::Int
    radius::Int
    g::Int
    n::Int
end
@inline function (f::InterpolateTangential)(v)
    interpolate_tangential_line!(f.tw_face, f.tangential, f.line, v, f.radius, f.g, f.n)
end

struct TangentialDerivatives{TX, TY, PX, PY}
    tangential_x::TX
    tangential_y::TY
    pads_x::PX
    pads_y::PY
    offset::Int
    radius::Int
    g::Int
    n::Int
end
@inline function (f::TangentialDerivatives)(v)
    @unpack tangential_x, tangential_y, pads_x, pads_y, offset, g, n = f
    if f.radius == 1
        tangential_derivatives_variable!(tangential_x, pads_x, pads_y, offset, v,
                                         Val(1),
                                         g, n)
        tangential_derivatives_variable!(tangential_y, pads_y, pads_x, offset, v,
                                         Val(1),
                                         g, n)
    else
        tangential_derivatives_variable!(tangential_x, pads_x, pads_y, offset, v,
                                         Val(2),
                                         g, n)
        tangential_derivatives_variable!(tangential_y, pads_y, pads_x, offset, v,
                                         Val(2),
                                         g, n)
    end
    return nothing
end

@noinline function interpolate_tangential_line!(tw_face, tangential, line, v, radius, g,
                                                n)
    RealT = eltype(tw_face)
    if radius == 1
        @inbounds @simd ivdep for f in 2:(n + 1)
            tw_face[f, v] = 0.5f0 * (tangential[g + f - 1, v, line] +
                             tangential[g + f, v, line])
        end
    else
        # (-1, 9, 9, -1) / 16
        w = inv(convert(RealT, 16))
        @inbounds @simd ivdep for f in 2:(n + 1)
            tw_face[f, v] = w *
                            (9 * (tangential[g + f - 1, v, line] +
                              tangential[g + f, v, line]) -
                             (tangential[g + f - 2, v, line] +
                              tangential[g + f + 1, v, line]))
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

@noinline function add_macrocell_source_terms_2d!(du, u, t, source_terms::Source,
                                                  element,
                                                  inv_h, equations, dg,
                                                  cache) where {Source}
    h = inv(inv_h)
    for j in eachnode(dg), i in eachnode(dg)
        u_local = get_node_vars(u, equations, dg, i, j, element)
        x_c = get_node_coords(cache.elements.node_coordinates, equations, dg, i, j,
                              element)
        source = subcell_average(SourceAt(source_terms, u_local, t, equations),
                                 cache.subcell_quadrature, x_c, h)
        add_to_node_vars!(du, source, equations, dg, i, j, element)
    end
    return nothing
end

# `source_terms(u, x, t, equations)` and `func(x, t, equations)` as functions of `x`
struct SourceAt{S, U, T, E}
    source_terms::S
    u::U
    t::T
    equations::E
end
@inline (f::SourceAt)(x) = f.source_terms(f.u, x, f.t, f.equations)

struct InitialConditionAt{F, T, E}
    func::F
    t::T
    equations::E
end
@inline (f::InitialConditionAt)(x) = f.func(x, f.t, f.equations)

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
    return subcell_average(InitialConditionAt(func, t, equations), quadrature, x_c, h)
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
