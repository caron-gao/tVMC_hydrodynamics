## 1) Setup and the “frozen–\(u_2\)” variational manifold

Take the usual \(N\)-boson Hamiltonian (time dependence only in the trap),
\[
\hat H(t)=\sum_{i=1}^N\left[-\frac{\hbar^2}{2m}\nabla_i^2+V_{\rm tr}(\mathbf r_i,t)\right]+\sum_{i<j}v(r_{ij}),
\qquad 
V_{\rm tr}(\mathbf r,t)=\tfrac12 m\omega(t)^2 r^2,
\]
and the truncated Jastrow-Feenberg ansatz
\[
\Psi(\mathbf X,t)=\exp U(\mathbf X,t),\qquad 
U(\mathbf X,t)=\sum_{i=1}^N u_1(\mathbf r_i,t)+\sum_{i<j}u_2(r_{ij}),
\]
with **\(u_2\) fixed, real, time-independent**, and \(u_1(\mathbf r,t)\) fully time-dependent (generically complex).

Define normalized expectations w.r.t. the instantaneous \(|\Psi|^2\) measure:
\[
\langle A\rangle_t \equiv \frac{\int d\mathbf X\,|\Psi(\mathbf X,t)|^2\,A(\mathbf X,t)}{\int d\mathbf X\,|\Psi(\mathbf X,t)|^2},
\qquad \Delta A \equiv A-\langle A\rangle_t,
\]
and the local energy
\[
E_L(\mathbf X,t)\equiv \frac{(\hat H(t)\Psi)(\mathbf X,t)}{\Psi(\mathbf X,t)}.
\]

A crucial simplification here is that, because \(u_2\) is real and fixed,
\[
|\Psi|^2=\exp\!\left(2\sum_i \Re u_1(\mathbf r_i,t)+2\sum_{i<j}u_2(r_{ij})\right),
\]
so the Monte Carlo *sampling measure* depends on \(\Re u_1\) (and \(u_2\)), but **not** on \(\Im u_1\).

---

## 2) TDVP EOM for the active field \(u_1(\mathbf r,t)\)

Use the Dirac–Frenkel/McLachlan TDVP projected Schrödinger equation on the tangent space spanned by derivatives of \(\ln\Psi\). For a general parameter set \(\{\theta_\alpha\}\),
\[
\sum_\beta S_{\alpha\beta}\,\dot\theta_\beta=-\frac{i}{\hbar}F_\alpha,
\quad
S_{\alpha\beta}=\langle \Delta O_\alpha^\*\,\Delta O_\beta\rangle_t,
\quad
F_\alpha=\langle \Delta O_\alpha^\*\,\Delta E_L\rangle_t,
\]
with logarithmic derivatives \(O_\alpha=\partial_{\theta_\alpha}\ln\Psi\).

Here the “parameter” is the field \(u_1(\mathbf r,t)\). The functional logarithmic derivative is
\[
O(\mathbf r;\mathbf X)\equiv \frac{\delta \ln\Psi(\mathbf X,t)}{\delta u_1(\mathbf r,t)}
=\sum_{i=1}^N\delta^{(3)}(\mathbf r-\mathbf r_i)
\equiv \hat\rho(\mathbf r),
\]
i.e. **the tangent operators are exactly density operators**.

Therefore the TDVP equation becomes the **continuum linear integral equation**
\[
\boxed{
\int d^3r'\;S(\mathbf r,\mathbf r';t)\,\partial_t u_1(\mathbf r',t)
=
-\frac{i}{\hbar}\,F(\mathbf r;t)
}
\tag{1}
\]
with
\[
\boxed{
S(\mathbf r,\mathbf r';t)=\big\langle \Delta \hat\rho(\mathbf r)\,\Delta \hat\rho(\mathbf r')\big\rangle_t
}
\tag{2}
\]
(the connected density–density correlator / static structure kernel), and
\[
\boxed{
F(\mathbf r;t)=\big\langle \Delta \hat\rho(\mathbf r)\,\Delta E_L\big\rangle_t
=
\big\langle \Delta \hat\rho(\mathbf r)\,\big(E_L-\langle E_L\rangle_t\big)\big\rangle_t.
}
\tag{3}
\]

Equivalently, if you formally invert the kernel \(S\),
\[
\boxed{
\partial_t u_1(\mathbf r,t)=
-\frac{i}{\hbar}\int d^3r'\; S^{-1}(\mathbf r,\mathbf r';t)\,F(\mathbf r';t).
}
\tag{4}
\]

### What is \(S(\mathbf r,\mathbf r')\) explicitly?
Using \(\hat\rho(\mathbf r)\hat\rho(\mathbf r')=\sum_i\delta(\mathbf r-\mathbf r_i)\delta(\mathbf r'-\mathbf r_i)+\sum_{i\neq j}\delta(\mathbf r-\mathbf r_i)\delta(\mathbf r'-\mathbf r_j)\), you can write
\[
S(\mathbf r,\mathbf r')=\rho(\mathbf r)\,\delta^{(3)}(\mathbf r-\mathbf r')
+\rho^{(2)}(\mathbf r,\mathbf r')-\rho(\mathbf r)\rho(\mathbf r'),
\tag{5}
\]
where \(\rho(\mathbf r)=\langle\hat\rho(\mathbf r)\rangle_t\) and \(\rho^{(2)}\) is the (ordered) two-body density.

### Where does \(u_2\) enter?
It enters through the local energy \(E_L\) and through the correlators taken with \(|\Psi|^2\).
Using \(U=\ln\Psi\), one has
\[
E_L(\mathbf X,t)=
\sum_{i=1}^N\left[
-\frac{\hbar^2}{2m}\Big(\nabla_i^2 U+(\nabla_i U)^2\Big)
+V_{\rm tr}(\mathbf r_i,t)\right]
+\sum_{i<j}v(r_{ij}),
\tag{6}
\]
with
\[
\nabla_i U=\nabla u_1(\mathbf r_i,t)+\sum_{j\neq i}\nabla_i u_2(r_{ij}),
\qquad
\nabla_i^2 U=\nabla^2 u_1(\mathbf r_i,t)+\sum_{j\neq i}\nabla_i^2 u_2(r_{ij}).
\tag{7}
\]

### Gauge/null modes
Because \(u_1(\mathbf r)\to u_1(\mathbf r)+c(t)\) only changes normalization/global phase, \(S\) has corresponding null directions. Numerically one fixes a gauge (remove the constant basis function, impose \(\int d^3r\,w(\mathbf r)\,u_1(\mathbf r,t)=0\), etc.) before inverting (1).

That’s the **final TDVP EOM** for “frozen \(u_2\), active \(u_1\)”.

---

## 3) Can this be reformulated as hydrodynamics (density \(\rho\), velocity \(\mathbf v\))?

Write
\[
u_1(\mathbf r,t)=a(\mathbf r,t)+i\theta(\mathbf r,t),
\qquad a,\theta\in\mathbb R.
\]
Because \(u_2\) is real, the many-body phase is
\[
\arg\Psi(\mathbf X,t)=\sum_{i=1}^N\theta(\mathbf r_i,t),
\]
so the flow is **purely one-body and irrotational**, with velocity potential \(\theta\):
\[
\boxed{
\mathbf v(\mathbf r,t)=\frac{\hbar}{m}\nabla\theta(\mathbf r,t)
}
\tag{8}
\]
(no backflow terms appear unless you also allow time-dependent two-/three-body *phase* correlations).

In that sense, this manifold is precisely the one behind **zero-temperature superfluid hydrodynamics**. If you can supply an **internal energy functional**
\[
E_{\rm int}[\rho;u_2]\equiv \langle \hat T+\hat V_{\rm int}\rangle_t
\quad\text{evaluated in the frozen-}u_2\text{ Jastrow state at density }\rho,
\]
then the TDVP action reduces to the standard hydrodynamic form (Berry term \(\sim \hbar\int \rho\,\partial_t\theta\), flow kinetic energy \(\sim \int \tfrac{m}{2}\rho v^2\)), and you get the closed PDEs
\[
\boxed{
\partial_t\rho+\nabla\!\cdot(\rho\mathbf v)=0
}
\tag{9}
\]
\[
\boxed{
m\,\partial_t\mathbf v+\nabla\!\left(\tfrac{m}{2}v^2+V_{\rm tr}(\mathbf r,t)+\mu_{u_2}[\rho](\mathbf r,t)\right)=0,
\qquad
\mu_{u_2}[\rho](\mathbf r,t)=\frac{\delta E_{\rm int}[\rho;u_2]}{\delta\rho(\mathbf r)}.
}
\tag{10}
\]

So: **yes**, the “\(u_2\) frozen, \(u_1\) active” TDVP dynamics *can* be cast as hydrodynamics **provided** you can express (or approximate) the correlated internal energy as a functional of \(\rho\).

What changes relative to GP? Only the functional \(\mu_{u_2}[\rho]\). In the simplest local-density approximation (LDA),
\[
E_{\rm int}^{\rm LDA}\approx \int d^3r\,\varepsilon(\rho(\mathbf r)),
\qquad
\mu(\rho)=\frac{d\varepsilon}{d\rho},
\]
and (9)–(10) become standard superfluid hydrodynamics with an EOS \(\mu(\rho)\) (possibly plus gradient/“quantum pressure” corrections).

---

## 4) Can you avoid Monte Carlo at every time step?

### Exact answer (within TDVP): generally **no**
Even with \(u_2\) frozen, the TDVP EOM (1) requires, at each time,
- \(S(\mathbf r,\mathbf r';t)=\langle\Delta\hat\rho(\mathbf r)\Delta\hat\rho(\mathbf r')\rangle_t\),
- \(F(\mathbf r;t)=\langle\Delta\hat\rho(\mathbf r)\Delta E_L\rangle_t\),

and these are expectation values in the **instantaneous** \(|\Psi(u_1(t),u_2)|^2\) distribution. Since \(\Re u_1(\mathbf r,t)\) changes during a breathing/quench, the sampling measure changes, and there is no exact deterministic shortcut unless you can evaluate those correlators by some other many-body solver.

### Practical/controlled ways to *approximately* eliminate time-step MC
If your observation “\(u_2\) is stiff” really means “short-range correlations remain in (quasi-)equilibrium as the density profile breathes”, then you can close the dynamics at the hydrodynamic level:

Do *one-time* QMC/VMC (or use known theory) for the **uniform** system to get an equation of state \(\mu(\rho)\) compatible with your frozen \(u_2\) (or directly with the physical interaction). Then solve (9)–(10) in the trap numerically (PDE), or even via scaling solutions for harmonic quenches.  
This removes MC from the real-time loop entirely; MC only enters once to fit \(\mu(\rho)\).