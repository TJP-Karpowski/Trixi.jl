# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# Central face stencils and the combination of a face reconstruction with a derivative
# scheme for the `MacroCellFV` solver.
#
# A central stencil of even order `2m` at the face between the subcells `c` and `c + 1`
# reads the subcells `c - m + 1, ..., c + m`. It is split into two halves, the part of
# the subcells `c - m + 1, ..., c` (left half) and the part of `c + 1, ..., c + m` (right
# half). Each subcell computes the left half of its face in positive direction and the
# right half of its face in negative direction, in the same vectorized loop over the
# subcells as the face reconstructions; the face loop adds the two halves. Each weight is
# thus applied once per face, and the loop over the subcells has no dependency between
# iterations. By the symmetry of the stencils, both halves use the same weights
# `h_k`, `k = 0, ..., m - 1` (weight of the subcell `k` cells away from the face):
#     value:      left half  Σ_k h_k a(c - k),   right half  Σ_k h_k a(c + 1 + k),
#     derivative: left half  Σ_k g_k a(c - k),   right half -Σ_k g_k a(c + 1 + k)
# (the derivative weights are antisymmetric). The derivative is per subcell width.

# Weights of the central stencils of order `2m` (exact for polynomials of degree
# `2m - 1`) at a face, for the averages (`points = false`) or for the point values at the
# subcell centers (`points = true`) of the subcells `j = -m + 1, ..., m` of unit width,
# where the face is at `x = 0` and the subcell `j` covers `[j - 1, j]`. Returns the
# half weights `(h_0, ..., h_{m-1})` of the value and `(g_0, ..., g_{m-1})` of the
# derivative (exact rational arithmetic).
function central_face_weights(m, points)
    moment(j, p) = points ? (j - 1 // 2)^p : (j^(p + 1) - (j - 1)^(p + 1)) // (p + 1)
    A = [Rational{BigInt}(moment(j, p)) for p in 0:(2 * m - 1), j in (-m + 1):m]
    b_value = [Rational{BigInt}(p == 0) for p in 0:(2 * m - 1)]
    b_derivative = [Rational{BigInt}(p == 1) for p in 0:(2 * m - 1)]
    value = A \ b_value
    derivative = A \ b_derivative
    # The subcell `j = -k` (`k` cells to the left of the face) is the entry `m - k`
    return (ntuple(k -> value[m - (k - 1)], m), ntuple(k -> derivative[m - (k - 1)], m))
end

function check_central_order(order)
    (order >= 2 && iseven(order)) ||
        throw(ArgumentError("`order` must be an even integer of at least 2"))
    return order ÷ 2
end

"""
    ReconstructionCentral(order = 4)

Central (linear, non-dissipative) face values of even `order`: the values at the face
between two subcells are the same on both sides, the combination of the `order` subcells
closest to the face that is exact for polynomials of degree `order - 1`. With
`MomentInversionIdentity`, the stencils act on the subcell averages, with the moment
inversions to point values ([`MomentInversionCentral`](@ref),
[`MomentInversionWENO`](@ref)) on the point values at the subcell centers. With equal
states on both sides, the numerical flux reduces to the central flux (for consistent
numerical fluxes such as [`flux_lax_friedrichs`](@ref)); the scheme has no numerical
dissipation, so it is meant for smooth (e.g., diffusion-dominated) problems.

The reconstruction reads `order/2` halo subcells per side.

!!! warning "Experimental code"
    This code is experimental and may change in any future release.
"""
struct ReconstructionCentral{M}
    order::Int
    # `true` for the weights of point values (set with a moment inversion to point
    # values, see `point_reconstruction`)
    points::Bool
    weights::NTuple{M, Float64}
end

function ReconstructionCentral(order = 4; points = false)
    m = check_central_order(order)
    value, _ = central_face_weights(m, points)
    return ReconstructionCentral{m}(Int(order), points, Float64.(value))
end

function Base.show(io::IO, reconstruction::ReconstructionCentral)
    print(io, "ReconstructionCentral(order = ", reconstruction.order,
          reconstruction.points ? ", point values)" : ")")
end

@inline nhalo(::ReconstructionCentral{M}) where {M} = M

# The two halves of the value at the faces of the subcell: `(w_minus, w_plus)` are the
# right half of its face in negative direction and the left half of its face in
# positive direction
@inline function reconstruct_cell(reconstruction::ReconstructionCentral, stencil)
    return central_halves(reconstruction.weights, stencil)
end

# The two halves `(Σ_k h_k a(k - 1), Σ_k h_k a(1 - k))` of a central stencil with the
# half weights `h_1, ..., h_M` (`k = 1` is the subcell itself), written as a fold over a
# tuple, which is unrolled: a `for` loop over `k` prevented the vectorization of the loop
# over the subcells for `M = 3` (sixth order)
@inline function central_halves(weights::NTuple{M}, stencil) where {M}
    u_0 = stencil(0)
    RealT = typeof(u_0)
    w_0 = convert(RealT, weights[1]) * u_0
    ks = ntuple(k -> k + 1, Val(M - 1))
    w_right = foldl(@inline((w, k)->w + convert(RealT, weights[k]) * stencil(k - 1)),
                    ks;
                    init = w_0)
    w_left = foldl(@inline((w, k)->w + convert(RealT, weights[k]) * stencil(1 - k)), ks;
                   init = w_0)
    return w_right, w_left
end

# Face states from the values `w_left` (from the subcell on the left side of the face,
# `w_plus`) and `w_right` (`w_minus` of the subcell on the right side): reconstructions
# give two states, the central stencils one state (the sum of the two halves)
@inline macrocell_face_values(reconstruction, w_left, w_right) = (w_left, w_right)

@inline function macrocell_face_values(::ReconstructionCentral, w_left, w_right)
    w = w_left + w_right
    return w, w
end

"""
    GradientCentral(order = 2)

Central derivative of even `order` at the faces between subcells, for the diffusive
fluxes of `MacroCellFV`: the combination of the `order` subcells closest to the face
that is exact for polynomials of degree `order - 1` (fourth order: `(a_{-1} - 15 a_0 +
15 a_1 - a_2) / (12 h)` for averages, `(a_{-1} - 27 a_0 + 27 a_1 - a_2) / (24 h)` for
point values at the subcell centers). Combine it with a face reconstruction by
[`CombinedScheme`](@ref). With `MomentInversionIdentity`, the stencil acts on the subcell
averages (exact derivative of a linear function of the conservative variables, e.g., for
[`LaplaceDiffusion1D`](@ref)); with the moment inversions to point values, on the point
values at the subcell centers (also for nonlinear reconstruction variables, e.g., the
temperature).

The derivative reads `order/2` halo subcells per side.

!!! warning "Experimental code"
    This code is experimental and may change in any future release.
"""
struct GradientCentral{M}
    order::Int
    points::Bool
    weights::NTuple{M, Float64}
end

function GradientCentral(order = 2; points = false)
    m = check_central_order(order)
    _, derivative = central_face_weights(m, points)
    return GradientCentral{m}(Int(order), points, Float64.(derivative))
end

function Base.show(io::IO, gradient::GradientCentral)
    print(io, "GradientCentral(order = ", gradient.order,
          gradient.points ? ", point values)" : ")")
end

@inline nhalo(::GradientCentral{M}) where {M} = M

# The two halves of the derivative (per subcell width) at the faces of the subcell (as
# `reconstruct_cell` of `ReconstructionCentral`); the derivative at a face is the sum of
# the left half of the subcell on its left and the right half of the subcell on its
# right
@inline function differentiate_cell(gradient::GradientCentral, stencil)
    # (antisymmetric weights: the right half has the opposite sign)
    d_right, d_left = central_halves(gradient.weights, stencil)
    return -d_right, d_left
end

"""
    GradientNone()

No derivative of a reconstruction variable (the variable does not enter the diffusive
fluxes as a gradient), see [`CombinedScheme`](@ref).

!!! warning "Experimental code"
    This code is experimental and may change in any future release.
"""
struct GradientNone end

@inline nhalo(::GradientNone) = 0

"""
    CombinedScheme(reconstruction, gradient = GradientNone())

Face values and face derivatives of one reconstruction variable of
[`MacroCellFV`](@ref): `reconstruction` gives the face values (e.g.,
[`ReconstructionWENO5Z`](@ref), [`ReconstructionMUSCL`](@ref), or
[`ReconstructionCentral`](@ref)), `gradient` the derivatives at the faces for the
diffusive fluxes ([`GradientCentral`](@ref), or [`GradientNone`](@ref) for variables
whose gradients are not needed). Both are evaluated in the same loop over the subcells.
A scheme without gradient can be given directly instead of
`CombinedScheme(scheme, GradientNone())`.

!!! warning "Experimental code"
    This code is experimental and may change in any future release.
"""
struct CombinedScheme{Reconstruction, Gradient}
    reconstruction::Reconstruction
    gradient::Gradient
end

CombinedScheme(reconstruction) = CombinedScheme(reconstruction, GradientNone())

function Base.show(io::IO, scheme::CombinedScheme)
    print(io, "CombinedScheme(", scheme.reconstruction, ", ", scheme.gradient, ")")
end

@inline function nhalo(scheme::CombinedScheme)
    return max(nhalo(scheme.reconstruction), nhalo(scheme.gradient))
end

@inline function reconstruct_cell(scheme::CombinedScheme, stencil)
    return reconstruct_cell(scheme.reconstruction, stencil)
end

@inline function macrocell_face_values(scheme::CombinedScheme, w_left, w_right)
    return macrocell_face_values(scheme.reconstruction, w_left, w_right)
end

# `True()` if the scheme computes face derivatives
@inline has_gradient(scheme) = False()
@inline has_gradient(::CombinedScheme{<:Any, GradientNone}) = False()
@inline has_gradient(::CombinedScheme) = True()

# Values and derivatives (per subcell width) at the faces of a subcell:
# `(w_minus, w_plus, d_minus, d_plus)`
@inline function evaluate_cell(scheme::CombinedScheme, stencil)
    w_minus, w_plus = reconstruct_cell(scheme.reconstruction, stencil)
    d_minus, d_plus = differentiate_cell(scheme.gradient, stencil)
    return w_minus, w_plus, d_minus, d_plus
end

# Stencil factor of the diffusive time step restriction: the largest magnitude of the
# Fourier symbol of the linear second-derivative operator (moment inversion `inversion`,
# derivative at the faces by `gradient`, flux differences) on subcells of unit width,
# relative to the one of the second-order stencil (4). The time step of the explicit
# Euler method for `u_t = ν u_xx` is stable for `ν Δt / h^2 <= 1 / (2 factor)`.
function diffusion_dt_factor(gradient::GradientCentral{M}, inversion) where {M}
    g = gradient.weights
    # Full derivative weights of the subcells `j = -M + 1, ..., M` (face at the right
    # side of the subcell `j = 0`)
    weights = [j <= 0 ? g[1 - j] : -g[j] for j in (-M + 1):M]
    lambda_max = 0.0
    for theta in range(0, pi, length = 1001)
        derivative_symbol = sum(weights[k] * cis(j * theta)
                                for (k, j) in enumerate((-M + 1):M))
        symbol = inversion_symbol(inversion, theta) * derivative_symbol *
                 (1 - cis(-theta))
        lambda_max = max(lambda_max, abs(symbol))
    end
    return lambda_max / 4
end

diffusion_dt_factor(::GradientNone, inversion) = 0.0
diffusion_dt_factor(scheme, inversion) = 0.0
function diffusion_dt_factor(scheme::CombinedScheme, inversion)
    return diffusion_dt_factor(scheme.gradient, inversion)
end

# Fourier symbol of the moment inversion (cell averages -> point values); the WENO
# moment inversion is estimated by the fourth-order central one (its linear weights)
inversion_symbol(::MomentInversionIdentity, theta) = 1.0
function inversion_symbol(inversion::MomentInversionCentral{M}, theta) where {M}
    return 1 + sum(inversion.coefficients[k] * (2 * cos(k * theta) - 2) for k in 1:M)
end
function inversion_symbol(::MomentInversionWENO, theta)
    # `MomentInversionCentral(order = 4)`: `c_1 = -1/24`
    return 1 - (2 * cos(theta) - 2) / 24
end
end # @muladd
