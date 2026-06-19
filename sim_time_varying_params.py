# -*- coding: utf-8 -*-
"""
Numerical Simulation: Time-Varying Psychological Parameters (Eqs. 17-19)
=========================================================================
Simulates extended H-SEIR with dynamic theta(t), tau(t), eps(t).

Fig_TimeVarying_Params.png — 4 panels:
(a) Psychological parameter trajectories
(b) Time-dependent R0(t) 
(c) Static vs dynamic I(t) comparison
(d) Phase portrait (theta_bar vs R0)

Output: D:\论文\H-SEIR输出\Fig_TimeVarying_Params.png (300 DPI)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib import rcParams
import os

# ================================================================
# Publication-quality settings
# ================================================================
rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8.5,
    'figure.dpi': 300,
    'savefig.dpi': 300,
})

# ================================================================
# Parameters — calibrated to produce R0 ~ 2-3 range
# ================================================================
N = 5000
k_avg = 6.0

# Basal rates (scaled for R0 ~ 2 range)
beta0 = 0.02       # basal SE transmission rate  
sigma = 0.12       # EI transition rate
gamma_base = 0.05  # IR removal rate
mu = 0.40          # refutation effectiveness
r0_intervention = 0.04  # counter-rumor intensity (ramped)

# Initial psychological means
theta0_mean = 0.43   # E[Beta(2.3,3.1)]
tau0_mean = 0.50     # E[Beta(2,2)]
eps0_mean = 0.40     # E[Beta(3,2)]

# Psychology dynamics (Eqs.17-19) — tuned for realistic rumor-outbreak psychology
zeta0 = 0.006        # theta baseline relaxation
zeta_S_factor = 0.12 # fact-check -> increased reflection (strong)
zeta_F_factor = 0.08 # sharing feedback -> learning

alpha_T_factor = 0.03   # trusted source exposure on trust (weak)
gamma_B_factor = 0.10   # betrayal/trust decay (dominant — trust erodes)
gamma0_tau = 0.010      # tau relaxation

eta_E_factor = 0.14    # emotional arousal effect on epsilon (strong transient)
delta_D_factor = 0.06  # desensitization (eventual calming)
delta0_eps = 0.018     # epsilon relaxation

# Reference static R0
R0_static = beta0 * k_avg * (1-theta0_mean)*(1+tau0_mean)*(1+eps0_mean) / gamma_base
print(f"Static R0 = {R0_static:.3f}")

# ================================================================
# ODE system
# ================================================================
def hseir_dynamic_psi(t, y):
    """y = [S, E, I, R, th, tau, eps]"""
    S, E, I, R, th, tau, eps = y
    
    # Clamp psychology params to valid domain
    th = np.clip(th, 0.02, 0.98)
    tau = np.clip(tau, 0.02, 0.98)
    eps = np.clip(eps, 0.02, 0.98)
    
    Ntot = max(S + E + I + R, 1.0)
    If = I / Ntot          # infected fraction
    Sf = S / Ntot         # susceptible fraction
    
    # Time-dependent transmission
    beta_t = beta0 * (1 - th) * (1 + tau) * (1 + eps)
    lam = beta_t * k_avg * I / Ntot
    
    # H-SEIR dynamics
    dS = -lam * S
    dE = lam * S - sigma * E
    dI = sigma * E - (gamma_base + mu * r0_intervention) * I
    dR = (gamma_base + mu * r0_intervention) * I
    
    # === Psychology dynamics (Eqs. 17–19) ===
    # Eq.17: dθ/dt — cognitive reflection increases with fact-checking & feedback
    fc_rate = 0.30 * np.sqrt(If)        # sublinear fact-check response  
    feedback = 0.20 * If                 # proportional to current spreaders
    dth = zeta_S_factor * Sf * fc_rate + zeta_F_factor * feedback - zeta0*(th - theta0_mean)
    dth *= (1-th)*(th+0.02)  # soft bounds
    
    # Eq.18: dτ/dt — trust dynamics: pure decay during outbreak ("trust erosion")
    # As outbreak intensity grows, people become less trusting of unverified info
    erosion = gamma_B_factor * If * (tau - 0.15)   # erodes trust, floor at 0.15
    baseline_recover = alpha_T_factor * np.exp(-t/250) * (tau0_mean - tau)  # slow recovery
    dtau = -erosion + baseline_recover
    dtau *= (1-tau)*(tau+0.05)  # soft bounds
    
    # Eq.19: dε/dt — emotional arousal peaks then desensitizes
    arousal = eta_E_factor * If**0.55      # nonlinear saturation (sharp peak)
    desens = delta_D_factor * (1-np.exp(-t/100)) * If * eps  # cumulative desensitization
    deps = arousal - desens - delta0_eps*(eps - eps0_mean)
    deps *= (1-eps)*(eps+0.03)  # soft bounds
    
    return [dS, dE, dI, dR, dth, dtau, deps]

def hseir_static(t, y):
    """Static-parameter H-SEIR for comparison."""
    S, E, I, R = y
    Ntot = max(S+E+I+R, 1.0)
    beta_s = beta0 * (1-theta0_mean)*(1+tau0_mean)*(1+eps0_mean)
    lam = beta_s * k_avg * I / Ntot
    return [-lam*S, lam*S - sigma*E, sigma*E-(gamma_base+mu*r0_intervention)*I,
            (gamma_base+mu*r0_intervention)*I]

# ================================================================
# Run simulations
# ================================================================
t_span = (0, 450)
t_eval = np.linspace(0, 450, 900)
I0, E0 = 10, 25
S0 = N - I0 - E0
y0_dyn = [S0, E0, I0, 0, theta0_mean, tau0_mean, eps0_mean]
y0_sta = [S0, E0, I0, 0]

print("Running dynamic simulation...")
sol_d = solve_ivp(hseir_dynamic_psi, t_span, y0_dyn, t_eval=t_eval, method='LSODA', rtol=1e-9, atol=1e-11)
print("Running static simulation...")
sol_s = solve_ivp(hseir_static, t_span, y0_sta, t_eval=t_eval, method='LSODA', rtol=1e-9, atol=1e-11)

t = sol_d.t
Sd, Ed, Id, Rd, thd, taud, epsd = sol_d.y
Ss, Es, Is, Rs = sol_s.y

# Compute time-dependent R0
R0_t = beta0*k_avg*(1-thd)*(1+taud)*(1+epsd)/gamma_base

# Statistics
pk_d, pk_s = np.max(Id), np.max(Is)
tp_d, tp_s = t[np.argmax(Id)], t[np.argmax(Is)]
fr_d, fr_s = Rd[-1]/N, Rs[-1]/N

print(f"\n=== Dynamic Model ===")
print(f"  Peak I: {pk_d:.0f} ({pk_d/N*100:.1f}%) @ t={tp_d:.0f}")
print(f"  Final rumor size: {fr_d*100:.1f}%")
print(f"  θ: {thd[0]:.3f} → {thd[-1]:.3f} ({(thd[-1]-thd[0])*100:+.1f}%)")
print(f"  τ: {taud[0]:.3f} → {taud[-1]:.3f} ({(taud[-1]-taud[0])*100:+.1f}%)")
print(f"  ε: {epsd[0]:.3f} → {epsd[-1]:.3f} ({(epsd[-1]-epsd[0])*100:+.1f}%)")
print(f"  R₀: {R0_t[0]:.3f} → {R0_t[-1]:.3f} (Δ{(R0_t[-1]/R0_t[0]-1)*100:+.1f}%)")
print(f"\n=== Static Model ===")
print(f"  Peak I: {pk_s:.0f} ({pk_s/N*100:.1f}%) @ t={tp_s:.0f}")  
print(f"  Final rumor size: {fr_s*100:.1f}%")

# ================================================================
# Figure (4-panel layout)
# ================================================================
fig, axes = plt.subplots(2, 2, figsize=(9, 7))

cmap = {'theta':'#2166AC', 'tau':'#B2182B', 'eps':'#4DAF4A',
        'R0':'#FF7F00', 'Idyn':'#377EB8', 'Ista':'#E41A1C'}

# --- Panel (a): Parameter evolution ---
ax = axes[0,0]
ax.plot(t, thd, color=cmap['theta'], lw=2, label=r'$\bar{\theta}(t)$')
ax.plot(t, taud, color=cmap['tau'],   lw=2, label=r'$\bar{\tau}(t)$')
ax.plot(t, epsd, color=cmap['eps'],   lw=2, label=r'$\bar{\varepsilon}(t)$')
for val, col in [(theta0_mean,cmap['theta']), (tau0_mean,cmap['tau']), (eps0_mean,cmap['eps'])]:
    ax.axhline(val, color=col, ls='--', alpha=0.35, lw=1)
ax.set_xlabel('Time (a.u.)'); ax.set_ylabel('Parameter value')
ax.set_title('(a) Psychological parameter evolution (Eqs.\u201319)', fontstyle='italic')
ax.legend(framealpha=0.9); ax.grid(alpha=0.3); ax.set_xlim(0,450); ax.set_ylim(0.28,0.70)

# --- Panel (b): R0(t) ---
ax = axes[0,1]
ax.fill_between(t, 1, R0_t, where=R0_t>1, alpha=0.20, color='#D73027')
ax.fill_between(t, 1, R0_t, where=R0_t<=1, alpha=0.20, color='#4575B4')
ax.plot(t, R0_t, color=cmap['R0'], lw=2)
ax.axhline(1, color='black', lw=0.8); ax.axhline(R0_static, color='gray', ls=':', lw=1,
           label=f'Static $R_0$={R0_static:.2f}')
ci = np.where(np.diff(np.sign(R0_t-1)))[0]
if len(ci)>0:
    tc = t[ci[0]]; ax.scatter([tc],[1], s=50, c='black', zorder=5)
    ax.annotate(f'$t_{{crit}}\\approx${tc:.0f}', xy=(tc,1), xytext=(tc+40,R0_t.max()*0.82),
                arrowprops=dict(arrowstyle='->',color='black'), fontsize=8)
ax.set_xlabel('Time (a.u.)'); ax.set_ylabel(r'Basic reproduction number $R_0$')
ax.set_title('(b) Time-dependent $R_0$', fontstyle='italic')
ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_xlim(0,450)

# --- Panel (c): I(t) comparison ---
ax = axes[1,0]
ax.plot(t, Id/N*100, color=cmap['Idyn'], lw=2,
        label=f'Dynamic $\\psi(t)$ (peak {pk_d/N*100:.1f}%, final {fr_d*100:.1f}%)')
ax.plot(t, Is/N*100, color=cmap['Ista'], lw=2, ls='--',
        label=f'Static $\\psi_0$ (peak {pk_s/N*100:.1f}%, final {fr_s*100:.1f}%)')
ax.fill_between(t, 0, Id/N*100, alpha=0.06, color=cmap['Idyn'])
ax.fill_between(t, 0, Is/N*100, alpha=0.04, color=cmap['Ista'])
red = (1-fr_d/fr_s)*100 if fr_s>0 else 0
ax.annotate(f'Rumor size $\\downarrow${red:.1f}%', xy=(380,(Id[-1]+Is[-1])/2/N*100),
            fontsize=9, fontweight='bold', color='#333',
            bbox=dict(boxstyle='round,pad=0.3',fc='white',ec='gray',alpha=0.9))
ax.set_xlabel('Time (a.u.)'); ax.set_ylabel('Infected fraction (%)')
ax.set_title('(c) Propagation dynamics: static vs dynamic', fontstyle='italic')
ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_xlim(0,450)

# --- Panel (d): Phase portrait ---
ax = axes[1,1]
sc = ax.scatter(thd, R0_t, c=t, cmap='viridis', s=3, alpha=0.7)
ax.plot(theta0_mean, R0_static, 'k*', ms=12, mec='black',
        label=f'Static eq.', zorder=5)
for idx in np.linspace(15, len(t)-15, 6, dtype=int):
    if idx<len(t)-1:
        ax.annotate('', xy=(thd[idx+1],R0_t[idx+1]), xytext=(thd[idx],R0_t[idx]),
                    arrowprops=dict(arrowstyle='->',color='gray',lw=0.7,alpha=0.5))
plt.colorbar(sc, ax=ax, label='Time (a.u.)', shrink=0.8).ax.tick_params(labelsize=8)
ax.set_xlabel(r'Mean cognitive reflection $\bar{\theta}$')
ax.set_ylabel(r'Reproduction number $R_0$')
ax.set_title('(d) Phase trajectory ($\\bar{\theta}$, $R_0$)', fontstyle='italic')
ax.legend(fontsize=8); ax.grid(alpha=0.3)

plt.tight_layout(pad=1.5)
out = r'D:\论文\H-SEIR输出\Fig_TimeVarying_Params.png'
fig.savefig(out, dpi=300, facecolor='white')
plt.close()
print(f"\nFigure saved: {out} ({os.path.getsize(out)/1024:.1f} KB)")
