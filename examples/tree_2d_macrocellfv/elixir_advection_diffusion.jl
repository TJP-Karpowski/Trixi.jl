using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the linear advection-diffusion equation on [-1, 1]^2, periodic,
# with the exact solution of an advected and decaying product of sine waves

advection_velocity = (1.0, 0.5)
equations = LinearScalarAdvectionEquation2D(advection_velocity)
diffusivity = 0.01
equations_parabolic = LaplaceDiffusion2D(diffusivity, equations)

# (a closure, so that `diffusivity` can be changed with `trixi_include`)
initial_condition = let nu = diffusivity
    function initial_condition_diffusive_sine(x, t,
                                              equations::LinearScalarAdvectionEquation2D)
        RealT = eltype(x)
        a = equations.advection_velocity
        x1 = x[1] - a[1] * t
        x2 = x[2] - a[2] * t
        decay = exp(-2 * nu * convert(RealT, pi)^2 * t)
        return SVector(1 + 0.5f0 * sinpi(x1) * sinpi(x2) * decay)
    end
end

# Macro-cell finite volume solver: WENO5-Z face values and fourth-order central face
# derivatives of the subcell averages along the x- and y-lines of each macro-cell,
# computed in the same loop over the subcells. For the linear advective and diffusive
# fluxes, they give the face averages of the fluxes to fourth order. The advective and
# the diffusive fluxes are evaluated in one pass with the
# `SemidiscretizationHyperbolicParabolic` below. The initial condition is set to the
# subcell averages (Gauss-Legendre quadrature with 3 nodes per direction).
n_cells = 8
reconstruction = CombinedScheme(ReconstructionWENO5Z(), GradientCentral(4))
solver = MacroCellFV(n_cells = n_cells, surface_flux = flux_lax_friedrichs,
                     reconstruction = reconstruction,
                     initial_condition_nodes = 3)

coordinates_min = (-1.0, -1.0)
coordinates_max = (1.0, 1.0)
initial_refinement_level = 3
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = initial_refinement_level,
                periodicity = true)

semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                             initial_condition, solver;
                                             boundary_conditions = (boundary_condition_periodic,
                                                                    boundary_condition_periodic))

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     extra_analysis_errors = (:conservation_error,))

alive_callback = AliveCallback(analysis_interval = analysis_interval)

# Advective and diffusive time step restrictions
stepsize_callback = StepsizeCallback(cfl = 0.5, cfl_parabolic = 0.5)

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = 1, # solve needs some value here but it will be overwritten by the stepsize_callback
            ode_default_options()..., callback = callbacks);
