using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the compressible Navier-Stokes equations: manufactured solution
# with source terms on [-1, 1], periodic (as `tree_1d_dgsem/
# elixir_navierstokes_convergence_periodic.jl`, with a larger viscosity so that the
# viscous terms matter)

prandtl_number() = 0.72
mu() = 0.01

equations = CompressibleEulerEquations1D(1.4)
equations_parabolic = CompressibleNavierStokesDiffusion1D(equations, mu = mu(),
                                                          Prandtl = prandtl_number())

# This convergence test setup was originally derived by Andrew Winters (@andrewwinters5000)
# (Simplified version of the 2D)
function initial_condition_navier_stokes_convergence_test(x, t, equations)
    # Amplitude and shift
    RealT = eltype(x)
    A = 0.5f0
    c = 2

    # convenience values for trig. functions
    pi_x = convert(RealT, pi) * x[1]
    pi_t = convert(RealT, pi) * t

    rho = c + A * sin(pi_x) * cos(pi_t)
    v1 = sin(pi_x) * cos(pi_t)
    p = rho^2

    return prim2cons(SVector(rho, v1, p), equations)
end
initial_condition = initial_condition_navier_stokes_convergence_test

@inline function source_terms_navier_stokes_convergence_test(u, x, t, equations)
    RealT = eltype(x)

    @unpack gamma, inv_gamma_minus_one = equations
    # we currently need to hardcode these parameters until we fix the "combined equation" issue
    # see also https://github.com/trixi-framework/Trixi.jl/pull/1160
    Pr = prandtl_number()
    mu_ = mu()

    # Same settings as in `initial_condition`
    # Amplitude and shift
    A = 0.5f0
    c = 2

    # convenience values for trig. functions
    pi_x = convert(RealT, pi) * x[1]
    pi_t = convert(RealT, pi) * t

    # compute the manufactured solution and all necessary derivatives
    rho = c + A * sin(pi_x) * cos(pi_t)
    rho_t = -convert(RealT, pi) * A * sin(pi_x) * sin(pi_t)
    rho_x = convert(RealT, pi) * A * cos(pi_x) * cos(pi_t)
    rho_xx = -convert(RealT, pi) * convert(RealT, pi) * A * sin(pi_x) * cos(pi_t)

    v1 = sin(pi_x) * cos(pi_t)
    v1_t = -convert(RealT, pi) * sin(pi_x) * sin(pi_t)
    v1_x = convert(RealT, pi) * cos(pi_x) * cos(pi_t)
    v1_xx = -convert(RealT, pi) * convert(RealT, pi) * sin(pi_x) * cos(pi_t)

    p = rho * rho
    p_t = 2 * rho * rho_t
    p_x = 2 * rho * rho_x
    p_xx = 2 * rho * rho_xx + 2 * rho_x * rho_x

    E = p * inv_gamma_minus_one + 0.5f0 * rho * v1^2
    E_t = p_t * inv_gamma_minus_one + 0.5f0 * rho_t * v1^2 + rho * v1 * v1_t
    E_x = p_x * inv_gamma_minus_one + 0.5f0 * rho_x * v1^2 + rho * v1 * v1_x

    # Some convenience constants
    T_const = gamma * inv_gamma_minus_one / Pr
    inv_rho_cubed = 1 / (rho^3)

    # compute the source terms
    # density equation
    du1 = rho_t + rho_x * v1 + rho * v1_x

    # x-momentum equation
    du2 = (rho_t * v1 + rho * v1_t
           + p_x + rho_x * v1^2 + 2 * rho * v1 * v1_x -
           # stress tensor from x-direction
           v1_xx * mu_ * 4 / 3)

    # total energy equation
    du3 = (E_t + v1_x * (E + p) + v1 * (E_x + p_x) -
           # stress tensor and temperature gradient terms from x-direction
           v1_xx * v1 * mu_ * 4 / 3 -
           v1_x * v1_x * mu_ * 4 / 3 -
           T_const * inv_rho_cubed *
           (p_xx * rho * rho -
            2 * p_x * rho * rho_x +
            2 * p * rho_x * rho_x -
            p * rho * rho_xx) * mu_)

    return SVector(du1, du2, du3)
end

# Macro-cell finite volume solver with the reconstruction variables
# `(rho, v1, p, R T)` (`cons2prim_temperature_pressure`) of the point values at the
# subcell centers (moment inversion of order 6): WENO5-Z face values of the primitive
# variables for the numerical flux in primitive variables, and fourth-order central face
# derivatives of the velocity and the temperature for the viscous fluxes (the temperature
# gets central face values; it enters only the viscous fluxes). Only these two variables
# cost derivative work.
n_cells = 8
schemes = (ReconstructionWENO5Z(),
           CombinedScheme(ReconstructionWENO5Z(), GradientCentral(4)),
           ReconstructionWENO5Z(),
           CombinedScheme(ReconstructionCentral(4), GradientCentral(4)))
solver = MacroCellFV(n_cells = n_cells, surface_flux = flux_hllc_primitive,
                     schemes = schemes,
                     moment_inversion = MomentInversionCentral(),
                     center2recon = cons2prim_temperature_pressure,
                     face2flux = face2flux_navier_stokes_primitive,
                     initial_condition_nodes = 5)

coordinates_min = -1.0
coordinates_max = 1.0
initial_refinement_level = 3
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = initial_refinement_level,
                periodicity = true)

semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                             initial_condition, solver;
                                             source_terms = source_terms_navier_stokes_convergence_test,
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
