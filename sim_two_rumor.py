# -*- coding: utf-8 -*-
"""
Numerical Simulation: Two-Rumor Competition (Eqs. 20-23)
=========================================================
Simulates the coupled multi-rumor H-SEIR framework with n=2 competing rumors.

Produces Fig_TwoRumor_Competition.png — 4 panels:
(a) I₁(t) and I₂(t) propagation curves showing competition
(b) Shared susceptible pool S(t) consumption
(c) Rumor dominance phase diagram (R₀¹ vs R₀²)
(d) Cross-inhibition strength ρ effect on competitive exclusion time

Output: D:\论文\H-SEIR输出\Fig_TwoRumor_Competition.png (300 DPI, Times New Roman)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib import rcParams
import os

rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman'],
    'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 12,
    'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'legend.fontsize': 8.5, 'figure.dpi': 300, 'savefig.dpi': 300,
})

# ================================================================
# Parameters
# ================================================================
N = 5000; k_avg = 6.0

# Two rumors with different transmissibility
beta1 = 0.022      # rumor 1 (dominant): higher transmission
beta2 = 0.016      # rumor 2 (subordinate): lower transmission
sigma1, sigma2 = 0.12, 0.11   # EI rates
gamma1, gamma2 = 0.05, 0.055  # IR rates
mu_r = 0.03        # counter-rumor intensity (shared)

# Psychological params (same for both rumors in baseline scenario)
theta_m = 0.43; tau_m = 0.50; eps_m = 0.40

# Effective transmission rates
betta1 = beta1 * (1-theta_m)*(1+tau_m)*(1+eps_m)
betta2 = beta2 * (1-theta_m)*(1+tau_m)*(1+eps_m)

R01 = betta1*k_avg/gamma1
R02 = betta2*k_avg/gamma2
print(f"R₀⁽¹⁾ = {R01:.3f}, R₀⁽²⁾ = {R02:.3f}")

# Cross-inhibition
rho = 0.0008  # cross-inhibition coefficient

# ================================================================
# ODE system: two-rumor H-SEIR (Eqs. 20–23)
# ================================================================
def two_rumor_ode(t, y):
    """
    State: [S, E1, I1, R1, E2, I2, R2]
    
    Eq.20: dS/dt = -Σ βₗ(1-θ)(1+τ)(1+ε)Θ_Iℓ · S
    Eq.21: dEℓ/dt = βℓ...·S - σℓ·Eℓ  
    Eq.22: dIℓ/dt = σℓ·Eℓ - (γℓ+μr)·Iℓ - Σ_{m≠ℓ} ρ·Iℓ·Im/N
    Eq.23: dRℓ/dt = (γℓ+μr)·Iℓ
    """
    S, E1, I1, R1, E2, I2, R2 = y
    Ntot = max(S+E1+I1+R1+E2+I2+R2, 1.0)
    
    # Force of infection for each rumor (mean-field)
    Theta1 = k_avg * I1 / Ntot
    Theta2 = k_avg * I2 / Ntot
    
    # Eq.20: Susceptible dynamics (shared pool)
    dS = -betta1*Theta1*S - betta2*Theta2*S
    
    # Eq.21-23: Compartmental dynamics per rumor
    dE1 = betta1*Theta1*S - sigma1*E1
    dI1 = sigma1*E1 - (gamma1 + mu_r)*I1 - rho*I1*I2/N  # cross-inhibition from rumor 2
    dR1 = (gamma1 + mu_r)*I1
    
    dE2 = betta2*Theta2*S - sigma2*E2
    dI2 = sigma2*E2 - (gamma2 + mu_r)*I2 - rho*I2*I1/N  # cross-inhibition from rumor 1
    dR2 = (gamma2 + mu_r)*I2
    
    return [dS, dE1, dI1, dR1, dE2, dI2, dR2]


def single_rumor_ode(t, y, beta, sigma, gamma):
    """Single-rumor reference (no competition)."""
    S, E, I, R = y
    Ntot = max(S+E+I+R, 1.0)
    b_eff = beta*(1-theta_m)*(1+tau_m)*(1+eps_m)
    lam = b_eff*k_avg*I/Ntot
    return [-lam*S, lam*S-sigma*E, sigma*E-(gamma+mu_r)*I, (gamma+mu_r)*I]

# ================================================================
# Main simulation
# ================================================================
t_span = (0, 500); t_eval = np.linspace(0, 500, 1000)

# Initial conditions: both rumors seeded simultaneously
I10, I20 = 5, 5
E10, E20 = 12, 12
S0 = N - I10 - I20 - E10 - E20
y0 = [S0, E10, I10, 0, E20, I20, 0]

print("Running two-rumor simulation...")
sol = solve_ivp(two_rumor_ode, t_span, y0, t_eval=t_eval, method='LSODA', rtol=1e-9, atol=1e-11)
print(f"Success: {sol.success}")

t = sol.t
S, E1, I1, R1, E2, I2, R2 = sol.y

# Single-rumor references (no competition)
print("Running single-rumor reference (rumor 1)...")
sol1 = solve_ivp(lambda t,y: single_rumor_ode(t,y,beta1,sigma1,gamma1), 
                 t_span, [N-17,12,5,0], t_eval=t_eval, method='LSODA')
print("Running single-rumor reference (rumor 2)...")
sol2 = solve_ivp(lambda t,y: single_rumor_ode(t,y,beta2,sigma2,gamma2),
                 t_span, [N-17,12,5,0], t_eval=t_eval, method='LSODA')

I1_alone = sol1.y[2]; I2_alone = sol2.y[2]
R1_alone = sol1.y[3]; R2_alone = sol2.y[3]

# Statistics
pk1, pk2 = np.max(I1), np.max(I2)
tp1, tp2 = t[np.argmax(I1)], t[np.argmax(I2)]
fr1, fr2 = R1[-1]/N, R2[-1]/N
fr1a, fr2a = R1_alone[-1]/N, R2_alone[-1]/N

# Dominance ratio at end
dom_ratio = fr1 / (fr2 + 1e-10)
exclusion_time = None
for i in range(len(t)-1):
    if I1[i] > 3*I2[i] and I1[i] > 10:
        exclusion_time = t[i]
        break

print(f"\n=== Two-Rumor Results ===")
print(f"  Rumor 1: peak={pk1:.0f}({pk1/N*100:.1f}%) @t={tp1:.0f}, final={fr1*100:.1f}%")
print(f"  Rumor 2: peak={pk2:.0f}({pk2/N*100:.1f}%) @t={tp2:.0f}, final={fr2*100:.1f}%")
print(f"  Dominance ratio R₁/R₂ = {dom_ratio:.2f}")
print(f"  Exclusion onset ~t={exclusion_time}" if exclusion_time else "  No clear exclusion")
print(f"\n  Alone: R1_final={fr1a*100:.1f}%, R2_final={fr2a*100:.1f}%")
print(f"  Competition cost: R1 ↓{(1-fr1/fr1a)*100:.1f}%, R2 ↓{(1-fr2/fr2a)*100:.1f}%")

# ================================================================
# Panel (c): Phase diagram over ρ sweep
# ================================================================
rho_vals = np.linspace(0, 0.003, 15)
final_ratios = []
exclusion_times = []
peak_ratios = []

for rho_v in rho_vals:
    def ode_rho(t, y):
        Sv, E1v, I1v, R1v, E2v, I2v, R2v = y
        Ntv = max(Sv+E1v+I1v+R1v+E2v+I2v+R2v, 1.0)
        dSv = -betta1*k_avg*I1v/Ntv*Sv - betta2*k_avg*I2v/Ntv*Sv
        dE1v = betta1*k_avg*I1v/Ntv*Sv - sigma1*E1v
        dI1v = sigma1*E1v - (gamma1+mu_r)*I1v - rho_v*I1v*I2v/Ntv
        dR1v = (gamma1+mu_r)*I1v
        dE2v = betta2*k_avg*I2v/Ntv*Sv - sigma2*E2v
        dI2v = sigma2*E2v - (gamma2+mu_r)*I2v - rho_v*I2v*I1v/Ntv
        dR2v = (gamma2+mu_r)*I2v
        return [dSv,dE1v,dI1v,dR1v,dE2v,dI2v,dR2v]
    
    sr = solve_ivp(ode_rho, t_span, y0, t_eval=t_eval, method='LSODA')
    f1 = sr.y[3][-1]/N; f2 = sr.y[6][-1]/N
    final_ratios.append(f1/(f2+1e-10))
    peak_ratios.append(np.max(sr.y[2]) / (np.max(sr.y[6])+1e-10))
    
    et = None
    for i in range(len(sr.t)-1):
        if sr.y[2,i] > 3*sr.y[6,i] and sr.y[2,i] > 10:
            et = sr.t[i]; break
    exclusion_times.append(et if et else 500)

# ================================================================
# Create figure (4-panel)
# ================================================================
fig, axes = plt.subplots(2, 2, figsize=(9, 7))

cl = {'I1':'#2166AC', 'I2':'#B2182B', 'S':'#4DAF4A', 
      'I1a':'#6BAED6', 'I2a':'#D6604D'}

# --- Panel (a): I1(t), I2(t) ---
ax = axes[0,0]
ax.plot(t, I1/N*100, color=cl['I1'], lw=2.2, label=f'$I_1(t)$ (R₀$^{(1)}$={R01:.2f})')
ax.plot(t, I2/N*100, color=cl['I2'], lw=2.2, label=f'$I_2(t)$ (R₀$^{(2)}$={R02:.2f})')
ax.plot(t, I1_alone/N*100, color=cl['I1a'], lw=1.2, ls=':', alpha=0.7, label='$I_1^\\mathrm{alone}(t)$')
ax.plot(t, I2_alone/N*100, color=cl['I2a'], lw=1.2, ls=':', alpha=0.7, label='$I_2^\\mathrm{alone}(t)$')
ax.fill_between(t, 0, I1/N*100, alpha=0.06, color=cl['I1'])
ax.fill_between(t, 0, I2/N*100, alpha=0.04, color=cl['I2'])
ax.set_xlabel('Time (a.u.)'); ax.set_ylabel('Infected fraction (%)')
ax.set_title('(a) Two-rumor propagation dynamics (Eqs.\u201323)', fontstyle='italic')
ax.legend(framealpha=0.9, ncol=2, fontsize=7.5); ax.grid(alpha=0.3); ax.set_xlim(0,500)

# Annotate dominance
if dom_ratio > 2:
    ax.annotate(f'Dominant: $R_1/R_2$={dom_ratio:.1f}', xy=(420, (I1[-1]+I2[-1])/2/N*100),
                fontsize=8, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.25',fc='#E8F5E9',ec='gray'))

# --- Panel (b): Shared susceptible pool ---
ax = axes[0,1]
ax.fill_between(t, 0, S/N*100, color=cl['S'], alpha=0.35)
ax.plot(t, S/N*100, color=cl['S'], lw=2, label=r'Susceptible pool $S(t)$')
ax.axhline(10, color='red', ls='--', lw=0.8, alpha=0.6, label='Depletion threshold (10%)')
ax.set_xlabel('Time (a.u.)'); ax.set_ylabel('Fraction of population (%)')
ax.set_title('(b) Shared susceptible pool depletion', fontstyle='italic')
ax.legend(framealpha=0.9); ax.grid(alpha=0.3); ax.set_xlim(0,500); ax.set_ylim(0,105)
ax.annotate(f'Final $S$ = {S[-1]/N*100:.1f}%', xy=(450, S[-1]/N*100+3), fontsize=8)

# --- Panel (c): ρ phase diagram ---
ax = axes[1,0]
colors_rho = plt.cm.coolwarm(np.linspace(0.15, 0.85, len(rho_vals)))
sc = ax.scatter(rho_vals*1000, final_ratios, c=rho_vals, cmap='coolwarm', s=50, edgecolors='black', linewidths=0.5, zorder=5)
ax.axhline(1, color='gray', ls='--', lw=0.8, alpha=0.6, label='Parity line')
ax.fill_between(rho_vals*1000, 1, final_ratios, where=np.array(final_ratios)>1, 
                alpha=0.15, color=cl['I1'], label='Rumor 1 dominant')
ax.fill_between(rho_vals*1000, 1, final_ratios, where=np.array(final_ratios)<=1,
                alpha=0.15, color=cl['I2'], label='Rumor 2 dominant')
plt.colorbar(sc, ax=ax, shrink=0.8).set_label('$\\rho \\times 10^3$')
ax.set_xlabel(r'Cross-inhibition strength $\rho \times 10^{-3}$')
ax.set_ylabel(r'Final size ratio $R_1 / R_2$')
ax.set_title('(c) Competitive exclusion vs inhibition strength $\\rho$', fontstyle='italic')
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# --- Panel (d): Exclusion time & peak ratio ---
ax = axes[1,1]
ax_twin = ax.twinx()
ln1 = ax.plot(rho_vals*1000, [e if e else 500 for e in exclusion_times], 
              'o-', color=cl['I1'], ms=5, lw=1.5, mew=1, label='Exclusion onset time')
ax.fill_between(rho_vals*1000, 0, [e if e else 500 for e in exclusion_times], alpha=0.1, color=cl['I1'])
ln2 = ax_twin.plot(rho_vals*1000, peak_ratios, 's--', color=cl['I2'], ms=5, lw=1.5, mew=1, label='Peak ratio $I_1^{max}/I_2^{max}$')
ax.set_xlabel(r'Cross-inhibition strength $\rho \times 10^{-3}$')
ax.set_ylabel('Exclusion onset time (a.u.)', color=cl['I1'])
ax_twin.set_ylabel('Peak ratio', color=cl['I2'])
ax.tick_params(axis='y', labelcolor=cl['I1']); ax_twin.tick_params(axis='y', labelcolor=cl['I2'])
ax.set_title('(d) Competition intensity effects', fontstyle='italic')
lns = [ln1[0], ln2[0]]
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc='center right', fontsize=8); ax.grid(alpha=0.3)

plt.tight_layout(pad=1.5)
out = r'D:\论文\H-SEIR输出\Fig_TwoRumor_Competition.png'
fig.savefig(out, dpi=300, facecolor='white')
plt.close()
print(f"\nFigure saved: {out} ({os.path.getsize(out)/1024:.1f} KB)")
