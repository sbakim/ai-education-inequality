"""Core model: AI adoption, digital-skill diffusion and learning on a homophilous network.
All dynamics are vectorized with numpy.
"""
import numpy as np
from dataclasses import dataclass, field, replace, asdict

GROUPS = ("L", "M", "H")
G_IDX = {"L": 0, "M": 1, "H": 2}

@dataclass
class Config:
    # population
    N: int = 500
    frac: tuple = (0.4, 0.4, 0.2)          # L, M, H shares
    C: int = 25                             # classrooms
    # network (degree-balanced block model)
    kbar: float = 8.0
    h: float = 0.6                          # homophily in [0,1)
    topology: str = "sbm"                  # 'sbm' | 'dcsbm'
    # classroom sorting
    rho: float = 0.5
    # physical access: population mean fixed, H-L gap varied
    pbar: float = 0.6
    dp: float = 0.5                         # p_H - p_L
    shape_mid: float = 0.615                # stylized p_M placement: p_M = p_L + shape_mid*dp
    # initial digital skill Beta(mu*kap,(1-mu)*kap)
    ds0: float = 0.3                        # mu_H - mu_L, mu_M = 0.5
    skill_conc: float = 8.0
    # adoption (complex contagion)
    theta0: float = 0.2
    lam: float = 0.5
    sigma0: float = 0.10                    # seed fraction among access-holders
    contagion: str = "complex"             # 'complex' | 'simple'
    beta_c: float = 0.08                    # per-contact prob (simple contagion variant)
    # skill dynamics
    eta: float = 0.05
    gamma_s: float = 0.10
    # learning (latent additive)
    beta_alpha: float = 0.010
    mu_alpha: float = 1.0
    sigma_u: float = 0.15                   # alpha_i = mu_alpha + u_i (random effect)
    beta_T: float = 0.005
    beta_AI: float = 0.012
    sigma_eps: float = 0.010
    A0: float = 0.0
    learning: str = "latent"               # 'latent' | 'saturating' (robustness)
    Amax: float = 2.0                       # saturating variant only
    # teacher support: Normal(0.6,0.2) clipped to [0.1,1]
    T_mu: float = 0.6
    T_sd: float = 0.2
    # scheduling / horizon
    sync: bool = True
    patience: int = 5
    T_post: int = 25
    T_max: int = 100
    # policy
    policy: str = "no_ai"
    budget: int = 60
    skill_boost: float = 0.2
    deseg_h: float = 0.3                    # h used by 'deseg' (applied before network generation)
    deseg_rho: float = 0.0

def access_probs(cfg):
    # Constant-population-mean access design:
    # frac_L*pL + frac_M*pM + frac_H*pH = pbar,
    # pM = pL + shape_mid*dp, pH = pL + dp.
    fL, fM, fH = cfg.frac
    pL = cfg.pbar - (fM * cfg.shape_mid + fH) * cfg.dp
    pM = pL + cfg.shape_mid * cfg.dp
    pH = pL + cfg.dp
    raw = np.array([pL, pM, pH], dtype=float)
    if np.all((raw >= 0.0) & (raw <= 1.0)):
        assert abs(np.dot(np.asarray(cfg.frac, float), raw) - cfg.pbar) < 1e-12
    return np.clip(raw, 0.0, 1.0)

def gen_network(cfg, ses, rng):
    """Degree-balanced block model: E[deg]=kbar for every SES group at any h.
    p_out = kbar*(1-h)/(N-1);  p_in,g = omega_g*kbar/(n_g-1),
    omega_g = omega0_g + h*(1-omega0_g), omega0_g=(n_g-1)/(N-1)."""
    N = cfg.N
    n = np.array([(ses == g).sum() for g in range(3)])
    omega0 = (n - 1) / (N - 1)
    omega = omega0 + cfg.h * (1 - omega0)
    p_in = omega * cfg.kbar / (n - 1)
    p_out = cfg.kbar * (1 - cfg.h) / (N - 1)
    P = np.full((3, 3), p_out)
    np.fill_diagonal(P, p_in)
    Pm = P[ses[:, None], ses[None, :]]
    if cfg.topology == "dcsbm":  # degree-corrected variant (E6): lognormal propensities, group-mean 1
        prop = rng.lognormal(mean=-0.125, sigma=0.5, size=N)
        for g in range(3):
            prop[ses == g] /= prop[ses == g].mean()
        Pm = np.clip(Pm * prop[:, None] * prop[None, :], 0, 1)
    U = rng.random((N, N))
    A = (U < Pm)
    A = np.triu(A, 1)
    A = A | A.T
    np.fill_diagonal(A, False)
    return A

def ses_assortativity(A, ses):
    """Newman categorical assortativity for SES on edges."""
    idx = np.array(np.nonzero(np.triu(A, 1)))
    if idx.shape[1] == 0:
        return np.nan
    gi, gj = ses[idx[0]], ses[idx[1]]
    e = np.zeros((3, 3))
    for a, b in zip(gi, gj):
        e[a, b] += 1; e[b, a] += 1
    e /= e.sum()
    ai = e.sum(1)
    tr = np.trace(e)
    return (tr - (ai ** 2).sum()) / (1 - (ai ** 2).sum())

def assign_classrooms(cfg, ses, rng):
    N, C = cfg.N, cfg.C
    size = N // C
    slots = np.repeat(np.arange(C), size)
    order = np.argsort(ses, kind="stable")          # SES-sorted stream
    n_sorted = int(round(cfg.rho * N))
    # choose which students are sorted-assigned (take a random subset of the sorted stream positions)
    pick = rng.choice(N, size=n_sorted, replace=False)
    is_sorted = np.zeros(N, bool); is_sorted[pick] = True
    room = np.empty(N, int)
    # sorted students occupy slots in SES order; the rest fill remaining slots randomly
    sorted_stream = order[is_sorted[order]]
    room[sorted_stream] = slots[:n_sorted]
    rest = order[~is_sorted[order]]
    rest_slots = slots[n_sorted:].copy(); rng.shuffle(rest_slots)
    room[rest] = rest_slots
    return room

def build_world(cfg, seed):
    """Everything drawn before dynamics. Structural policies act on cfg BEFORE generation."""
    cfg = apply_structural_policy(cfg)
    rng = np.random.default_rng(seed)
    N = cfg.N
    n = [int(round(f * N)) for f in cfg.frac]; n[2] = N - n[0] - n[1]
    ses = np.concatenate([np.full(k, g) for g, k in enumerate(n)])
    rng.shuffle(ses)
    p = access_probs(cfg)
    c = (rng.random(N) < p[ses]).astype(np.int8)
    muL, muH = 0.5 - cfg.ds0 / 2, 0.5 + cfg.ds0 / 2
    mus = np.array([muL, 0.5, muH]); kap = cfg.skill_conc
    s = rng.beta(np.maximum(mus[ses] * kap, 1e-3), np.maximum((1 - mus[ses]) * kap, 1e-3))
    alpha = cfg.mu_alpha + rng.normal(0, cfg.sigma_u, N)   # random-effect formulation
    A = gen_network(cfg, ses, rng)
    deg = A.sum(1)
    room = assign_classrooms(cfg, ses, rng)
    Tc = np.clip(rng.normal(cfg.T_mu, cfg.T_sd, cfg.C), 0.1, 1.0)
    T = Tc[room]
    theta = cfg.theta0 * (1 - cfg.lam * T)
    a = np.zeros(N, np.int8)
    Aach = np.full(N, cfg.A0, float)
    world = dict(cfg=cfg, rng=rng, ses=ses, c=c, s=s, alpha=alpha, adj=A, deg=deg,
                 room=room, T=T, theta=theta, a=a, ach=Aach, ai_on=True)
    apply_agent_policy(world)
    # seed adoption uniformly among access-holders (unless a targeting policy already seeded)
    if world["a"].sum() == 0 and world["ai_on"]:
        holders = np.flatnonzero(world["c"] == 1)
        k = int(round(cfg.sigma0 * len(holders)))
        if k > 0:
            world["a"][rng.choice(holders, size=k, replace=False)] = 1
    return world

# ---------------- policies ----------------
def apply_structural_policy(cfg):
    if cfg.policy == "deseg":
        return replace(cfg, h=cfg.deseg_h, rho=cfg.deseg_rho)
    return cfg

def _seed_and_equip(world, targets):
    world["c"][targets] = 1
    world["a"][targets] = 1

def cross_frac(world):
    A, ses = world["adj"], world["ses"]
    deg = np.maximum(world["deg"], 1)
    same = (ses[:, None] == ses[None, :])
    return (A & ~same).sum(1) / deg

def betweenness(world, rng):
    import networkx as nx
    G = nx.from_numpy_array(world["adj"])
    return np.array(list(nx.betweenness_centrality(G, k=64, seed=int(rng.integers(1e9))).values()))

def apply_agent_policy(world):
    cfg, rng = world["cfg"], world["rng"]
    pol, B = cfg.policy, cfg.budget
    L = np.flatnonzero(world["ses"] == 0)
    if pol == "no_ai":
        world["ai_on"] = False
    elif pol in ("universal", "deseg"):
        pass                                            # availability only: physical access c_i unchanged
    elif pol == "access_full":
        world["c"][:] = 1                               # universal physical-access equalization
    elif pol == "access_budget":
        no_acc = L[world["c"][L] == 0]
        k = min(B, len(no_acc))
        if k: world["c"][rng.choice(no_acc, size=k, replace=False)] = 1
    elif pol == "skills":
        order = L[np.argsort(world["s"][L])][:B]
        world["s"][order] = np.clip(world["s"][order] + cfg.skill_boost, 0, 1)
    elif pol == "target_random":
        t = rng.choice(L, size=min(B, len(L)), replace=False); _seed_and_equip(world, t)
    elif pol == "target_lowskill":
        t = L[np.argsort(world["s"][L])][:B]; _seed_and_equip(world, t)
    elif pol == "target_degree":
        t = L[np.argsort(-world["deg"][L])][:B]; _seed_and_equip(world, t)
    elif pol == "target_betweenness":
        bc = betweenness(world, rng); t = L[np.argsort(-bc[L])][:B]; _seed_and_equip(world, t)
    elif pol == "target_bridge":
        cf = cross_frac(world); elig = L[world["deg"][L] >= 3]
        t = elig[np.argsort(-cf[elig])][:B]; _seed_and_equip(world, t)
    elif pol == "combined":
        b = B // 3
        no_acc = L[world["c"][L] == 0]
        if len(no_acc): world["c"][rng.choice(no_acc, size=min(b, len(no_acc)), replace=False)] = 1
        order = L[np.argsort(world["s"][L])][:b]
        world["s"][order] = np.clip(world["s"][order] + cfg.skill_boost, 0, 1)
        bc = betweenness(world, rng); t = L[np.argsort(-bc[L])][:B - 2 * b]; _seed_and_equip(world, t)
    else:
        raise ValueError(pol)

# ---------------- dynamics ----------------
def step(world):
    cfg = world["cfg"]; A = world["adj"]; rng = world["rng"]
    a, s = world["a"], world["s"]
    deg = np.maximum(world["deg"], 1)
    if world["ai_on"]:
        na = A @ a
        if cfg.contagion == "complex":
            w = na / deg
            new = (world["c"] == 1) & (w >= world["theta"])
        else:                                            # simple contagion (E6)
            pr = 1 - (1 - cfg.beta_c) ** na
            new = (world["c"] == 1) & (rng.random(cfg.N) < pr)
        if cfg.sync:
            a_next = np.where(new, 1, a).astype(np.int8)
        else:                                            # asynchronous: random half updates (E6)
            upd = rng.random(cfg.N) < 0.5
            a_next = a.copy(); a_next[upd & new] = 1
    else:
        a_next = a
    # skills (S1): uses current a
    up = A & (s[None, :] > s[:, None])
    cnt = up.sum(1)
    sbar = np.where(cnt > 0, (up * s[None, :]).sum(1) / np.maximum(cnt, 1), s)
    gain = cfg.eta * a * (1 - s) + cfg.gamma_s * np.maximum(sbar - s, 0)
    s_next = s + gain
    # learning (L1)
    ai_term = cfg.beta_AI * a * s if world["ai_on"] else 0.0
    inc = cfg.beta_alpha * world["alpha"] + cfg.beta_T * world["T"] + ai_term \
          + rng.normal(0, cfg.sigma_eps, cfg.N)
    if cfg.learning == "latent":
        ach_next = world["ach"] + inc
    else:                                                # saturating robustness variant
        ach_next = world["ach"] + inc * (1 - world["ach"] / cfg.Amax)
    changed = int((a_next != a).sum())
    world["a"], world["s"], world["ach"] = a_next, s_next, ach_next
    return changed

def run_simulation(cfg, seed, trajectories=False):
    world = build_world(cfg, seed)
    cfg = world["cfg"]
    ses = world["ses"]
    quiet, t, tstar = 0, 0, None
    traj = []
    def snap():
        return [t] + [world["a"][ses == g].mean() for g in range(3)] \
                   + [world["s"][ses == g].mean() for g in range(3)] \
                   + [world["ach"][ses == g].mean() for g in range(3)]
    while t < cfg.T_max:
        if trajectories: traj.append(snap())
        ch = step(world)
        t += 1
        # validation each step
        assert world["s"].min() >= -1e-12 and world["s"].max() <= 1 + 1e-12, "skill bounds violated"
        assert np.isfinite(world["ach"]).all(), "achievement NaN/inf"
        quiet = quiet + 1 if ch == 0 else 0
        if tstar is None and quiet >= cfg.patience:
            tstar = t - cfg.patience
        if tstar is not None and t >= tstar + cfg.T_post:
            break
    if trajectories: traj.append(snap())
    if cfg.contagion == "complex" and cfg.sync:
        pass  # monotonicity guaranteed by construction (a_next >= a)
    out = metrics(world)
    out.update(t_star=tstar if tstar is not None else cfg.T_max, T_end=t,
               assort=ses_assortativity(world["adj"], ses),
               deg_all=float(world["deg"].mean()),
               **{f"deg_{g}": float(world["deg"][ses == G_IDX[g]].mean()) for g in GROUPS})
    return (out, np.array(traj)) if trajectories else out

def metrics(world):
    from scipy.stats import wasserstein_distance
    ses, ach = world["ses"], world["ach"]
    a, s, c = world["a"], world["s"], world["c"]
    m = {}
    for g in GROUPS:
        i = ses == G_IDX[g]
        m[f"ach_{g}"] = ach[i].mean(); m[f"x_{g}"] = a[i].mean()
        m[f"s_{g}"] = s[i].mean(); m[f"as_{g}"] = (a[i] * s[i]).mean()
        m[f"acc_{g}"] = c[i].mean()
    nH, nL = (ses == 2).sum(), (ses == 0).sum()
    vH, vL = ach[ses == 2].var(ddof=1), ach[ses == 0].var(ddof=1)
    sp = np.sqrt(((nH - 1) * vH + (nL - 1) * vL) / (nH + nL - 2))
    m["G_HL"] = m["ach_H"] - m["ach_L"]
    m["d_HL"] = m["G_HL"] / sp if sp > 0 else 0.0
    m["sd"] = ach.std(ddof=1)
    m["q90_10"] = np.quantile(ach, 0.9) - np.quantile(ach, 0.1)
    gm = np.array([ach[ses == g].mean() for g in range(3)])
    ns = np.array([(ses == g).sum() for g in range(3)])
    ssb = (ns * (gm - ach.mean()) ** 2).sum()
    m["bg_share"] = ssb / ((ach - ach.mean()) ** 2).sum()
    m["wass_HL"] = wasserstein_distance(ach[ses == 2], ach[ses == 0])
    m["x_all"] = a.mean(); m["as_gap"] = m["as_H"] - m["as_L"]
    m["mean_ach"] = ach.mean()
    return m

def run_mc(cfg, R, base_seed=0):
    import pandas as pd
    rows = [run_simulation(cfg, base_seed + 1000 * r) for r in range(R)]
    return pd.DataFrame(rows)

def paired_contrast(cfg_a, cfg_b, R, base_seed=0, cols=("d_HL", "G_HL", "mean_ach", "sd", "bg_share", "x_L", "x_H", "as_gap")):
    """Common random numbers: same seed -> same population/network draws for both configs."""
    import pandas as pd
    rows = []
    for r in range(R):
        sd = base_seed + 1000 * r
        oa, ob = run_simulation(cfg_a, sd), run_simulation(cfg_b, sd)
        rows.append({f"{k}_a": oa[k] for k in cols} | {f"{k}_b": ob[k] for k in cols} |
                    {f"d_{k}": ob[k] - oa[k] for k in cols})
    return pd.DataFrame(rows)

def mean_field_boundary(cfg, h_grid, dp_grid, iters=200):
    """Two-block-family mean-field: predicted adoption fixed point per SES group and
    the implied effective-use differential under universal availability. Derived calculation,
    compared with (not fitted to) simulation."""
    from scipy.stats import binom
    N = cfg.N; n = np.asarray(cfg.frac, float) * N
    thbar = cfg.theta0 * (1 - cfg.lam * cfg.T_mu)
    k = int(round(cfg.kbar))
    out = np.zeros((len(h_grid), len(dp_grid)))
    for ih, h in enumerate(h_grid):
        omega0 = (n - 1) / (N - 1); omega = omega0 + h * (1 - omega0)
        # neighbour-composition matrix m[g,g']: share of g's neighbours in g'
        M = np.zeros((3, 3))
        for g in range(3):
            M[g, g] = omega[g]
            others = [gg for gg in range(3) if gg != g]
            tot = sum(n[gg] for gg in others)
            for gg in others:
                M[g, gg] = (1 - omega[g]) * n[gg] / tot
        for idp, dp in enumerate(dp_grid):
            p = access_probs(replace(cfg, dp=dp))
            x = p * cfg.sigma0
            for _ in range(iters):
                w = M @ x
                # P(Binom(k, w) >= ceil(theta*k)) smoothed threshold response
                thr = int(np.ceil(thbar * k))
                adopt_pr = 1 - binom.cdf(thr - 1, k, np.clip(w, 0, 1))
                x = p * (cfg.sigma0 + (1 - cfg.sigma0) * adopt_pr)
            out[ih, idp] = x[2] - x[0]     # adoption differential H - L (drives as-gap sign)
    return out
