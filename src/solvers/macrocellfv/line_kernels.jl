# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# Line kernels of the `MacroCellFV` solver: the face reconstruction and the face fluxes
# along one coordinate line of subcells. They do not depend on the mesh or the
# dimension; the mesh-specific code (e.g., `macrocellfv_1d.jl`) gathers the line into
# the pad, calls the two passes below, and computes the flux differences.
#
# Data layout (`n` subcells per macro-cell, `g` halo subcells per side):
#
#   pad[a, v]        reconstruction variable `v` of the pad cell `a = c + g`, where the
#                    subcells of the macro-cell are `c = 1, ..., n` and the halo/ghost
#                    subcells `c <= 0` and `c > n`
#   w_minus[c + 1, v], w_plus[c + 1, v]
#                    face values of the subcell `c = 0, ..., n + 1` at its faces in
#                    negative and positive direction (for central stencils: the halves
#                    of the face value, see `ReconstructionCentral`)
#   dw_minus[c + 1, v], dw_plus[c + 1, v]
#                    the same for the face derivatives (per subcell width) of the
#                    variables with a gradient scheme; empty buffers without diffusion
#   flux[f, v]       numerical flux (conservative variable `v`) at the face
#                    `f = 1, ..., n + 1` between the subcells `f - 1` and `f`
#
#          subcell f-1           |           subcell f
#      ... w_minus  ...  w_plus  |  w_minus  ...  w_plus ...
#                             face f: left state  = w_plus[f, :]   (of subcell f - 1)
#                                     right state = w_minus[f + 1, :] (of subcell f)
#
# Both passes are loops over the subcells or faces of the line without dependencies
# between the iterations, so that they vectorize (`@simd ivdep`). They are separate,
# non-inlined functions: inlined into the element kernel, the loops were not vectorized.
# They take the buffers as individual arrays (not the named tuple of all line buffers),
# which kept the advective kernel as fast as before the extension by diffusion.

#####################################################################
# Pass 1: face values (and derivatives) of the subcells, one variable at a time

"""
    reconstruct_line!(line_buffers, pad, volume_integral, c_first, c_last, g,
                      n_recon::Val)

Face values (and, for the schemes with a gradient, face derivatives) of the subcells
`c_first:c_last` of the line in `pad` (with `g` halo cells per side) for all
`n_recon` reconstruction variables, written to `line_buffers` (see the data layout at the
top of `line_kernels.jl`). With one value-only scheme for all variables (pure
advection), all variables are processed in one non-inlined function
(`reconstruct_line_values!`). Otherwise, each variable is processed by its scheme
`pad_scheme(volume_integral, v)` in its own vectorized loop over the subcells:
`reconstruct_variable_values!` for value-only schemes,
`evaluate_variable!` for schemes with face derivatives
([`CombinedScheme`](@ref) with a gradient).

The loop over the variables is unrolled (`ntuple` with `Val(n_recon)`), so that a
different scheme type per variable (keyword `schemes`) is resolved at compile time.

The pad has `nhalo(dg) >= nhalo(scheme)` halo cells on each side, so the stencils of
the subcells `0:(n + 1)` lie inside the pad (checked by the tests with
`--check-bounds=yes`).
"""
@inline function reconstruct_line!(line_buffers, pad, volume_integral, c_first, c_last,
                                   g, n_recon::Val)
    reconstruct_line!(volume_integral.schemes, has_gradients(volume_integral),
                      line_buffers, pad, volume_integral, c_first, c_last, g, n_recon)
    return nothing
end

# One value-only scheme for all variables (pure advection): all variables in one
# non-inlined function. One non-inlined call per variable instead made the advective
# RHS of the Euler equations up to 12 % slower for 4 to 8 subcells per macro-cell.
@inline function reconstruct_line!(::Nothing, ::False, line_buffers, pad,
                                   volume_integral, c_first, c_last, g, n_recon)
    @unpack w_minus, w_plus = line_buffers
    reconstruct_line_values!(w_minus, w_plus, pad, volume_integral.reconstruction,
                             c_first, c_last, g, n_recon)
    return nothing
end

# Otherwise one variable at a time with its scheme (unrolled over the variables)
@inline function reconstruct_line!(schemes, gradients, line_buffers, pad,
                                   volume_integral, c_first, c_last, g, n_recon)
    ntuple(@inline(v->reconstruct_variable!(line_buffers, pad,
                                            pad_scheme(volume_integral, v), v,
                                            c_first, c_last, g)),
           n_recon)
    return nothing
end

@noinline function reconstruct_line_values!(w_minus, w_plus, pad, reconstruction,
                                            c_first, c_last, g,
                                            ::Val{NRecon}) where {NRecon}
    for v in Base.OneTo(NRecon)
        @inbounds @simd ivdep for c in c_first:c_last
            a = c + g
            wm, wp = reconstruct_cell(reconstruction,
                                      @inline(offset->@inbounds(pad[a + offset, v])))
            w_minus[c + 1, v] = wm
            w_plus[c + 1, v] = wp
        end
    end
    return nothing
end

# One reconstruction variable `v` with its scheme: values only, or values and face
# derivatives (`has_gradient`)
@inline function reconstruct_variable!(line_buffers, pad, scheme, v, c_first, c_last, g)
    return reconstruct_variable!(has_gradient(scheme), line_buffers, pad, scheme, v,
                                 c_first, c_last, g)
end

@inline function reconstruct_variable!(::False, line_buffers, pad, scheme, v, c_first,
                                       c_last, g)
    @unpack w_minus, w_plus = line_buffers
    reconstruct_variable_values!(w_minus, w_plus, pad, value_scheme(scheme), v, c_first,
                                 c_last, g)
    return nothing
end

@inline function reconstruct_variable!(::True, line_buffers, pad, scheme, v, c_first,
                                       c_last, g)
    @unpack w_minus, w_plus, dw_minus, dw_plus = line_buffers
    evaluate_variable!(w_minus, w_plus, dw_minus, dw_plus, pad, scheme, v, c_first,
                       c_last, g)
    return nothing
end

# The value scheme of a scheme (the reconstruction of a `CombinedScheme` without
# gradient)
@inline value_scheme(scheme) = scheme
@inline value_scheme(scheme::CombinedScheme) = scheme.reconstruction

"""
    reconstruct_variable_values!(w_minus, w_plus, pad, reconstruction, v,
                                 c_first, c_last, g)

Face values `w_minus[c + 1, v]`, `w_plus[c + 1, v]` of the subcells `c = c_first:c_last`
from `reconstruct_cell(reconstruction, stencil)`, where `stencil(offset)` is the value
of variable `v` of the pad cell `offset` cells away from the subcell `c`.
"""
@noinline function reconstruct_variable_values!(w_minus, w_plus, pad, reconstruction, v,
                                                c_first, c_last, g)
    @inbounds @simd ivdep for c in c_first:c_last
        a = c + g
        wm, wp = reconstruct_cell(reconstruction,
                                  @inline(offset->@inbounds(pad[a + offset, v])))
        w_minus[c + 1, v] = wm
        w_plus[c + 1, v] = wp
    end
    return nothing
end

"""
    evaluate_variable!(w_minus, w_plus, dw_minus, dw_plus, pad, scheme, v,
                       c_first, c_last, g)

Face values and face derivatives (per subcell width) of the subcells `c_first:c_last`
from `evaluate_cell(scheme, stencil)` (see [`CombinedScheme`](@ref)) in one loop over
the subcells. For central derivatives, `dw_plus[c + 1, v]` and `dw_minus[c + 2, v]` are
the two halves of the derivative at the face between the subcells `c` and `c + 1`,
which the face pass adds (`surface_gradient`). In the studies of the solver, this
generic loop was as fast as hand-written combinations that load the stencil once, and
faster than separate loops for the values and the derivatives.
"""
@noinline function evaluate_variable!(w_minus, w_plus, dw_minus, dw_plus, pad, scheme,
                                      v, c_first, c_last, g)
    @inbounds @simd ivdep for c in c_first:c_last
        a = c + g
        wm, wp, dm, dp = evaluate_cell(scheme,
                                       @inline(offset->@inbounds(pad[a + offset, v])))
        w_minus[c + 1, v] = wm
        w_plus[c + 1, v] = wp
        dw_minus[c + 1, v] = dm
        dw_plus[c + 1, v] = dp
    end
    return nothing
end

#####################################################################
# Pass 2: fluxes at the faces of the line

"""
    calc_line_fluxes!(flux, line_buffers, f_first, f_last, inv_h,
                      equations, equations_parabolic, volume_integral)

Fluxes `flux[f, :]` at the faces `f = f_first:f_last` of the line from the face values
(and derivatives) of pass 1. Without diffusion (`equations_parabolic === nothing`),
the numerical flux of the two face states; with diffusion, the numerical flux minus the
diffusive flux of the face state and the face derivatives (`macrocell_face_flux`).
`inv_h` is the inverse subcell width (for the face derivatives).
"""
@inline function calc_line_fluxes!(flux, line_buffers, f_first, f_last, inv_h,
                                   equations, equations_parabolic, volume_integral)
    @unpack w_minus, w_plus, dw_minus, dw_plus = line_buffers
    calc_line_fluxes!(flux, w_minus, w_plus, dw_minus, dw_plus, f_first, f_last, inv_h,
                      equations, equations_parabolic, volume_integral)
    return nothing
end

# Without diffusion, the derivative buffers are not passed to the face loop (passing
# them made the advective kernel slower)
@inline function calc_line_fluxes!(flux, line_buffers, f_first, f_last, inv_h,
                                   equations, equations_parabolic::Nothing,
                                   volume_integral)
    @unpack w_minus, w_plus = line_buffers
    calc_line_fluxes!(flux, w_minus, w_plus, nothing, nothing, f_first, f_last, inv_h,
                      equations, equations_parabolic, volume_integral)
    return nothing
end

@noinline function calc_line_fluxes!(flux, w_minus, w_plus, dw_minus, dw_plus, f_first,
                                     f_last, inv_h, equations, equations_parabolic,
                                     volume_integral)
    @inbounds @simd ivdep for f in f_first:f_last
        # The subcell `f - 1` on the left, the subcell `f` on the right of the face
        w_ll, w_rr = face_states(w_minus, w_plus, f, volume_integral, equations)
        flux_face = line_face_flux(w_ll, w_rr, dw_minus, dw_plus, f, inv_h, equations,
                                   equations_parabolic, volume_integral)
        for v in eachvariable(equations)
            flux[f, v] = flux_face[v]
        end
    end
    return nothing
end

# Advective flux only
@inline function line_face_flux(w_ll, w_rr, dw_minus, dw_plus, f, inv_h, equations,
                                equations_parabolic::Nothing, volume_integral)
    return macrocell_face_flux(w_ll, w_rr, 1, equations, volume_integral)
end

# Advective minus diffusive flux; the state of the diffusive flux is the mean of the two
# face states (the central value for central stencils)
@inline function line_face_flux(w_ll, w_rr, dw_minus, dw_plus, f, inv_h, equations,
                                equations_parabolic, volume_integral)
    w_face = 0.5f0 * (w_ll + w_rr)
    dw_face = face_derivatives(dw_minus, dw_plus, f, inv_h, volume_integral, equations)
    return macrocell_face_flux(w_ll, w_rr, w_face, dw_face, 1, equations,
                               equations_parabolic, volume_integral)
end

# The states of the reconstruction variables on the left and right sides of the face
# `f`: the stored face values, or for central stencils the assembled central value on
# both sides (`macrocell_face_values`)
@inline function face_states(w_minus, w_plus, f, volume_integral, equations)
    states = ntuple(@inline(v->macrocell_face_values(pad_scheme(volume_integral, v),
                                                     w_plus[f, v], w_minus[f + 1, v])),
                    Val(nreconvariables(volume_integral, equations)))
    return SVector(map(first, states)), SVector(map(last, states))
end

# The derivatives of the reconstruction variables at the face `f`: the two halves
# combined by `surface_gradient` and scaled by the inverse subcell width (zero for the
# variables without gradient scheme, whose buffers are not written)
@inline function face_derivatives(dw_minus, dw_plus, f, inv_h, volume_integral,
                                  equations)
    @unpack surface_gradient = volume_integral
    derivatives = ntuple(@inline(v->face_derivative(has_gradient(pad_scheme(volume_integral,
                                                                            v)),
                                                    surface_gradient, dw_minus,
                                                    dw_plus, f, v, inv_h)),
                         Val(nreconvariables(volume_integral, equations)))
    return SVector(derivatives)
end

@inline function face_derivative(::True, surface_gradient, dw_minus, dw_plus, f, v,
                                 inv_h)
    return surface_gradient(dw_plus[f, v], dw_minus[f + 1, v]) * inv_h
end
@inline function face_derivative(::False, surface_gradient, dw_minus, dw_plus, f, v,
                                 inv_h)
    return zero(eltype(dw_plus))
end
end # @muladd
