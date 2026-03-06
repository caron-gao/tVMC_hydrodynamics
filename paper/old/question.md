# Theoretical Derivation: Simplified t-VMC for BEC Trap Quenches

## Project Context
We are investigating the non-equilibrium dynamics of an $N$-particle BEC in a 3D harmonic trap with interactions. We have successfully implemented a Time-Dependent Variational Monte Carlo (t-VMC) simulation using a Jastrow ansatz truncated at the second order:
$$ \Psi(\mathbf{X}, t) = \exp\left( \sum_{i} u_1(\mathbf{r}_i, t) + \sum_{i<j} u_2(\mathbf{r}_{ij}, t) \right) $$
Our numerical simulations of a trap frequency quench ($\omega_i \to \omega_f$) indicate that the two-body correlation factor $u_2$ changes very little during the dynamics, suggesting it is "stiff" compared to the one-body density variations.

## The Problem
Based on the observation that $u_2$ remains nearly static, I want to explore a simplified theoretical framework where we freeze the two-body correlations and only evolve the one-body term.

Please address the following theoretical tasks regarding the Equations of Motion (EOM) derived from the Time-Dependent Variational Principle (TDVP):

Assume that $u_2(\mathbf{r}_{ij})$ is a non-zero, real-valued function that is **fixed in time** (time-independent), while $u_1(\mathbf{r}, t)$ remains fully variational and time-dependent.
*   Derive the final equation of motion for the active field $u_1(\mathbf{r}, t)$.
*   Analyze if this EOM can be reformulated into a **Hydrodynamical Theory** (involving density $\rho$ and velocity fields $\mathbf{v}$) or a similar "effective potential" theory.
*   **Goal:** Determine if we can simulate this specific regime (fixed $u_2$, varying $u_1$) *without* performing costly Monte Carlo sampling at every time step.