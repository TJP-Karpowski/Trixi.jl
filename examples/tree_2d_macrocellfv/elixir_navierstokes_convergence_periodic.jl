using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the compressible Navier-Stokes equations: manufactured solution
# with source terms on [-1, 1]^2, periodic, with a density that is constant in space and
# a divergence-free velocity field:
#   rho = 2 + sin(π t) / 2,
#   v   = V(t) (sin(π x) cos(π y), -cos(π x) sin(π y)),
#   p   = 2 + P(t) cos(π x) cos(π y),       V(t) = P(t) = cos(π t) / 2

prandtl_number() = 0.72
mu() = 0.01

equations = CompressibleEulerEquations2D(1.4)
equations_parabolic = CompressibleNavierStokesDiffusion2D(equations, mu = mu(),
                                                          Prandtl = prandtl_number())

function initial_condition_navier_stokes_separable(x, t, equations)
    RealT = eltype(x)
    rho = 2 + sinpi(t) / 2
    V = cospi(t) / 2
    P = V
    s_x, c_x = sincospi(x[1])
    s_y, c_y = sincospi(x[2])
    v1 = V * s_x * c_y
    v2 = -V * c_x * s_y
    p = 2 + P * c_x * c_y
    return prim2cons(SVector{4, RealT}(rho, v1, v2, p), equations)
end
initial_condition = initial_condition_navier_stokes_separable

# The source terms: rho_t + div(rho v) = rho_t (div v = 0); the shear stress τ_12
# vanishes, so the viscous momentum terms are -μ Δv = 2 π² μ v
@inline function source_terms_navier_stokes_separable(u, x, t, equations)
    RealT = eltype(x)
    gamma = equations.gamma
    mu_ = mu()
    pi_ = convert(RealT, pi)
    s_t, c_t = sincospi(t)
    rho = 2 + s_t / 2
    drho = pi_ * c_t / 2
    V = c_t / 2
    dV = -pi_ * s_t / 2
    P = V
    dP = dV
    s_x, c_x = sincospi(x[1])
    s_y, c_y = sincospi(x[2])
    # κ ∇T = μ γ / ((γ - 1) Pr) ∇p / rho
    kappa = mu_ * gamma / ((gamma - 1) * prandtl_number() * rho)
    d_rhoV = drho * V + rho * dV
    q_plus = (s_x * c_y)^2 + (c_x * s_y)^2
    q_minus = (s_x * c_y)^2 - (c_x * s_y)^2

    du1 = drho
    du2 = (d_rhoV - pi_ * P + 2 * pi_^2 * mu_ * V) * s_x * c_y +
          pi_ * rho * V^2 * s_x * c_x
    du3 = -(d_rhoV + pi_ * P + 2 * pi_^2 * mu_ * V) * c_x * s_y +
          pi_ * rho * V^2 * s_y * c_y
    # E_t + v ⋅ ∇(E + p) - div(τ v) - div(κ ∇T)
    du4 = (dP / (gamma - 1) + 2 * pi_^2 * kappa * P) * c_x * c_y +
          (drho * V^2 / 2 + rho * V * dV) * q_plus -
          gamma / (gamma - 1) * pi_ * P * V * q_minus +
          pi_ * rho * V^3 * c_x * c_y * q_minus -
          2 * mu_ * pi_^2 * V^2 * (2 * (c_x * c_y)^2 - q_plus)
    return SVector(du1, du2, du3, du4)
end
source_terms = source_terms_navier_stokes_separable

# Macro-cell finite volume solver with the reconstruction variables
# `(rho, v1, v2, p, R T)` (`cons2prim_temperature_pressure`) of the point values at the
# subcell centers (fourth-order moment inversion): WENO5-Z face values of the primitive
# variables, converted to the conservative variables for the numerical flux, and
# fourth-order central face derivatives of the velocity and the temperature for the
# viscous fluxes, including their derivatives along the faces. The fluxes at the face
# midpoints are converted to face averages. The source terms are averaged over the
# subcells with the Gauss-Legendre quadrature of the initial condition.
n_cells = 8
schemes = (ReconstructionWENO5Z(),
           CombinedScheme(ReconstructionWENO5Z(), GradientCentral(4)),
           CombinedScheme(ReconstructionWENO5Z(), GradientCentral(4)),
           ReconstructionWENO5Z(),
           CombinedScheme(ReconstructionCentral(4), GradientCentral(4)))
moment_inversion = MomentInversionCentral(order = 4)
center2recon = cons2prim_temperature_pressure
face2flux = face2flux_navier_stokes_prim2cons
solver = MacroCellFV(n_cells = n_cells, surface_flux = flux_hllc,
                     schemes = schemes,
                     moment_inversion = moment_inversion,
                     center2recon = center2recon, face2flux = face2flux,
                     initial_condition_nodes = 3)

coordinates_min = (-1.0, -1.0)
coordinates_max = (1.0, 1.0)
initial_refinement_level = 2
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = initial_refinement_level,
                periodicity = true)

semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                             initial_condition, solver;
                                             source_terms = source_terms,
                                             boundary_conditions = (boundary_condition_periodic,
                                                                    boundary_condition_periodic))

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 0.5)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

stepsize_callback = StepsizeCallback(cfl = 0.5, cfl_parabolic = 0.5)

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = 1, # solve needs some value here but it will be overwritten by the stepsize_callback
            ode_default_options()..., callback = callbacks);
