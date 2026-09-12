#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Turbulence : cascade de Kolmogorov et exposants anomaux.
Modele en couches (GOY). On verifie d'abord que le terme non lineaire
conserve exactement l'energie, puis on mesure le spectre et les exposants
des fonctions de structure.
"""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- modele GOY
N   = 22
lam = 2.0
k0  = 2.0**-4
k   = k0*lam**np.arange(N)
delta = 0.5

def nl(u, a, b, c):
    """Terme non lineaire GOY : i (a u*_{n+1}u*_{n+2} + b u*_{n-1}u*_{n+1}
                                    + c u*_{n-1}u*_{n-2})."""
    uc = np.conj(u)
    t = np.zeros(N, dtype=complex)
    t[:-2] += a[:-2]*uc[1:-1]*uc[2:]
    t[1:-1] += b[1:-1]*uc[:-2]*uc[2:]
    t[2:]   += c[2:]*uc[1:-1]*uc[:-2]
    return 1j*t

def test_conservation(a, b, c, ntry=200, seed=3):
    r = np.random.default_rng(seed); worst = 0.0
    for _ in range(ntry):
        u = r.standard_normal(N) + 1j*r.standard_normal(N)
        dE = 2*np.real(np.sum(np.conj(u)*nl(u, a, b, c)))
        worst = max(worst, abs(dE)/np.sum(np.abs(u)**2))
    return worst

print("="*74)
print("(0) CHOIX DES COEFFICIENTS : conservation exacte de l'energie")
print("="*74)
cands = {
 "a=k_n, b=-d k_{n-1}, c=-(1-d) k_{n-2}":
   (k.copy(), -delta*np.roll(k,1), -(1-delta)*np.roll(k,2)),
 "a=k_n, b=-d k_n/lam, c=-(1-d) k_n/lam^2":
   (k.copy(), -delta*k/lam, -(1-delta)*k/lam**2),
 "a=k_{n+1}, b=-d k_n, c=-(1-d) k_{n-1}":
   (np.roll(k,-1), -delta*k, -(1-delta)*np.roll(k,1)),
}
best = None
for name,(a,b,c) in cands.items():
    w = test_conservation(a,b,c)
    print(f"  {name:44s} |dE|/E = {w:.3e}")
    if best is None or w < best[0]: best = (w, name, a, b, c)
print(f"\n  -> retenu : {best[1]}   (|dE|/E = {best[0]:.2e})")
A, B, C = best[2], best[3], best[4]

# helicite generalisee
H = ((-1)**np.arange(N))*k
r = np.random.default_rng(11)
u = r.standard_normal(N) + 1j*r.standard_normal(N)
dH = 2*np.real(np.sum(H*np.conj(u)*nl(u,A,B,C)))
print(f"  invariant d'helicite  H = sum (-1)^n k_n |u_n|^2 : "
      f"|dH|/H = {abs(dH)/abs(np.sum(H*np.abs(u)**2)):.3e}")

# ---------------------------------------------------------------- integration
nu = 2e-8
fshell = 1
famp = 5e-3*(1+1j)

def rhs(u):
    d = nl(u, A, B, C) - nu*k**2*u
    d[fshell] += famp
    return d

def run(T, dt, u0, tstat=None, nsamp=4000):
    u = u0.copy(); n = int(T/dt)
    if tstat is None: tstat = 0.4*T
    istat = int(tstat/dt); every = max(1, (n-istat)//nsamp)
    S = []; Es = []
    for i in range(n):
        k1 = rhs(u); k2 = rhs(u+0.5*dt*k1)
        k3 = rhs(u+0.5*dt*k2); k4 = rhs(u+dt*k3)
        u = u + dt*(k1+2*k2+2*k3+k4)/6
        if not np.all(np.isfinite(u)): raise RuntimeError(f"divergence a i={i}")
        if i >= istat and (i-istat) % every == 0:
            S.append(np.abs(u).copy()); Es.append(np.sum(np.abs(u)**2))
    return u, np.array(S), np.array(Es)

print()
print("="*74)
print("(1) INTEGRATION VERS L'ETAT STATISTIQUEMENT STATIONNAIRE")
print("="*74)
print(f"  N={N} couches, k_0={k0}, lambda={lam}, nu={nu:.0e}")
print(f"  k_max = {k[-1]:.1f} ; forcage sur la couche n={fshell}")
rng = np.random.default_rng(1)
u0 = 1e-3*(rng.standard_normal(N)+1j*rng.standard_normal(N))*k**(-1/3)
dt = 3e-4
u1, _, _ = run(60.0, dt, u0, tstat=59.0, nsamp=10)
uf, S, Es = run(600.0, dt, u1, tstat=150.0, nsamp=6000)
print(f"  {len(S)} echantillons ; energie totale moyenne = {Es.mean():.5e} "
      f"(ecart-type {Es.std()/Es.mean()*100:.1f} %)")
eps = nu*np.sum(k**2*np.mean(S**2, axis=0))
inj = 2*np.real(np.conj(famp)*np.mean([1],axis=0))*0  # place-tenue
print(f"  dissipation moyenne eps = nu sum k^2 <|u|^2> = {eps:.5e}")

# --------------------------------------------------------------- (2) spectre
print()
print("="*74)
print("(2) SPECTRE : <|u_n|^2> ~ k_n^(-2/3)  (equivaut a E(k) ~ k^(-5/3))")
print("="*74)
m2 = np.mean(S**2, axis=0)
nin = np.arange(5, 17)          # zone inertielle
G2 = np.exp((np.log(m2[:-2])+np.log(m2[1:-1])+np.log(m2[2:]))/3.0)
kg2 = k[1:-1]
sel2 = np.array([i for i,n in enumerate(range(1,N-1)) if n in nin])
p = np.polyfit(np.log(kg2[sel2]), np.log(G2[sel2]), 1)
print(f"  ajustement sur les couches {nin[0]}..{nin[-1]} :")
print(f"    pente mesuree = {p[0]:+.5f}   K41 : -2/3 = {-2/3:+.5f}")
print(f"    donc E(k) ~ k^({p[0]-1:+.4f})   K41 : -5/3 = {-5/3:+.4f}")

# ------------------------------------------- (3) exposants anomaux zeta_p
print()
print("="*74)
print("(3) EXPOSANTS DES FONCTIONS DE STRUCTURE  <|u_n|^p> ~ k_n^(-zeta_p)")
print("="*74)
ps = np.array([1,2,3,4,5,6,7,8])
print("  Le modele GOY presente une oscillation de periode 3 en n (artefact")
print("  connu). On l'elimine par la moyenne geometrique sur chaque triade :")
print("  G_p(n) = (S_p(n-1) S_p(n) S_p(n+1))^(1/3), licite car k_{n-1}k_n k_{n+1}=k_n^3.")
comp = np.mean(S**2, axis=0)*k**(2/3)
for rr in range(3):
    sel = [n for n in range(4,18) if n%3==rr]
    print(f"    avant correction, <|u|^2> k^(2/3) moyen pour n mod 3 = {rr} : "
          f"{np.mean(comp[sel]):.4e}")
zet, err = [], []
for pp in ps:
    Sp = np.mean(S**pp, axis=0)
    G = np.exp((np.log(Sp[:-2])+np.log(Sp[1:-1])+np.log(Sp[2:]))/3.0)
    kg = k[1:-1]
    sel = np.array([i for i,n in enumerate(range(1,N-1)) if n in nin])
    cf, cov = np.polyfit(np.log(kg[sel]), np.log(G[sel]), 1, cov=True)
    zet.append(-cf[0]); err.append(np.sqrt(cov[0,0]))
zet = np.array(zet); err = np.array(err)
z3 = zet[2]
print(f"{'p':>3}{'zeta_p mesure':>16}{'incert.':>10}{'K41 = p/3':>12}"
      f"{'zeta_p/zeta_3':>15}{'ecart a p/3':>13}")
for pp, z, e in zip(ps, zet, err):
    print(f"{pp:3d}{z:16.4f}{e:10.4f}{pp/3:12.4f}{z/z3:15.4f}{z-pp/3:+13.4f}")
print(f"\n  zeta_3 mesure = {z3:.5f}   (valeur exacte attendue : 1)")
print(f"  ecart : {abs(z3-1)*100:.2f} %")
print(f"  -> zeta_3 = 1 est l'analogue en modele en couches de la loi des 4/5.")
print(f"  -> zeta_p < p/3 pour p > 3 : intermittence (K41 est faux en detail).")

# She-Leveque
zSL = ps/9 + 2*(1-(2/3)**(ps/3))
print(f"\n  comparaison a She-Leveque  zeta_p = p/9 + 2[1-(2/3)^(p/3)] :")
print(f"{'p':>3}{'mesure':>10}{'She-Leveque':>14}{'K41':>9}")
for pp, z, zs in zip(ps, zet/z3, zSL):
    print(f"{pp:3d}{z:10.4f}{zs:14.4f}{pp/3:9.4f}")

# ------------------------------------------------------------------ figures
plt.rcParams.update({"font.size":9,"axes.grid":True,"grid.alpha":.3,
                     "figure.dpi":150,"savefig.bbox":"tight","axes.linewidth":.8})
C1,C2,C3="#1f3b73","#b3452c","#2f7d4f"
fig,axs=plt.subplots(1,2,figsize=(9.4,3.3))
axs[0].loglog(k, m2, "o-", color=C1, ms=4, mfc="white", mew=1.2, label=r"$\langle|u_n|^2\rangle$")
kk=k[nin]
axs[0].loglog(kk, m2[nin[0]]*(kk/kk[0])**(-2/3), "k--", lw=1.1, label=r"$k^{-2/3}$ (K41)")
axs[0].axvspan(k[nin[0]],k[nin[-1]],color=C3,alpha=.10)
axs[0].set_xlabel(r"$k_n$"); axs[0].set_ylabel(r"$\langle|u_n|^2\rangle$")
axs[0].legend(fontsize=8); axs[0].set_title("cascade inertielle et dissipation",fontsize=9.5)
axs[1].errorbar(ps, zet/z3, yerr=err/z3, fmt="o", color=C1, ms=5, mfc="white",
                mew=1.4, capsize=3, label="modele en couches")
axs[1].plot(ps, ps/3, "k--", lw=1.1, label=r"K41 : $\zeta_p=p/3$")
axs[1].plot(ps, zSL, "-.", color=C2, lw=1.3, label="She-Leveque")
axs[1].set_xlabel(r"$p$"); axs[1].set_ylabel(r"$\zeta_p/\zeta_3$")
axs[1].legend(fontsize=8); axs[1].set_title("exposants anomaux",fontsize=9.5)
fig.savefig("/home/claude/t_turb.pdf")
np.savez("/home/claude/turb.npz", k=k, m2=m2, ps=ps, zet=zet, err=err, z3=z3)
print("\nfigure ecrite.")
