#!/usr/bin/env python3
"""
Week 2: ARIMA specification for ONS UK vacancies (VACS02, All vacancies, AP2Y).

Usage:  python3 analysis.py <path to vacs02 xlsx> <output dir>

Steps
  1. Load the 3-month rolling-average vacancy LEVELS (thousands), build a monthly index.
  2. Chronological split: last 24 observations are HELD OUT and never used for any
     test, model selection or diagnostic below.
  3. Stationarity: ADF (+ KPSS) on levels, first and second differences, log series,
     and pre-/post-COVID subsamples.
  4. ACF / PACF, then an ARIMA(p,1,q) grid (with/without drift) ranked by AIC/AICc/BIC.
  5. Residual diagnostics for the recommended model (Ljung-Box, ARCH, normality).
"""
import sys, re, json, warnings
import numpy as np, pandas as pd
import openpyxl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.graphics.gofplots import qqplot

warnings.filterwarnings("ignore")
XLSX, OUT = sys.argv[1], sys.argv[2]
import os
os.makedirs(OUT + "/figures", exist_ok=True)
os.makedirs(OUT + "/tables", exist_ok=True)

H = 24  # hold-out length (months)
BLUE, ORANGE, GREY, RED = "#2a6fbb", "#d9730d", "#6b7280", "#b3261e"
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e5e7eb", "grid.linewidth": 0.6,
    "axes.titlesize": 10, "axes.titleweight": "bold", "lines.linewidth": 1.4,
})

# ------------------------------------------------------------------ 1. data
wb = openpyxl.load_workbook(XLSX, data_only=True)
ws = wb["levels"]
pat = re.compile(r"^([A-Z][a-z]{2})\s*-\s*([A-Z][a-z]{2})\s+(\d{4})$")
rows = []
for r in ws.iter_rows(min_row=9, values_only=True):
    m = pat.match(str(r[0]).strip()) if r[0] else None
    if m and isinstance(r[2], (int, float)):
        end = pd.to_datetime(f"{m.group(2)} {m.group(3)}", format="%b %Y")
        rows.append((end.to_period("M"), r[0].strip(), float(r[2])))
df = pd.DataFrame(rows, columns=["period", "label", "vac"]).set_index("period")
_ords = np.array([p.ordinal for p in df.index])
assert (np.diff(_ords) == 1).all(), "gap in months"
y_all = df["vac"]
y_all.index.name = "month_ending"
train, test = y_all.iloc[:-H], y_all.iloc[-H:]
summary = {
    "n_total": len(y_all), "first": str(y_all.index[0]), "last": str(y_all.index[-1]),
    "n_train": len(train), "train_end": str(train.index[-1]),
    "n_test": len(test), "test_start": str(test.index[0]),
    "train_min": float(train.min()), "train_max": float(train.max()),
    "train_min_date": str(train.idxmin()), "train_max_date": str(train.idxmax()),
}
print(summary)

def fmt_dates(ix):
    return ix.to_timestamp()

# ---------------------------------------------------------- 2. figure: series
fig, ax = plt.subplots(2, 1, figsize=(7.2, 5.4), sharex=True)
ax[0].plot(fmt_dates(train.index), train, color=BLUE, label="Training")
ax[0].plot(fmt_dates(test.index), test, color=GREY, label="Held out (not used)")
ax[0].axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2021-12-31"), color=ORANGE, alpha=0.12, lw=0)
ax[0].text(pd.Timestamp("2019-10-01"), 1180, "COVID-19\nwindow", fontsize=7.5, color=ORANGE, va="top", ha="right")
ax[0].set_ylabel("Vacancies (thousands)")
ax[0].set_title("UK vacancies, all industries (3-month rolling average, levels)")
ax[0].legend(frameon=False, loc="upper left")
d1_all = y_all.diff()
ax[1].plot(fmt_dates(train.index), d1_all.loc[train.index], color=BLUE)
ax[1].plot(fmt_dates(test.index), d1_all.loc[test.index], color=GREY)
ax[1].axhline(0, color="black", lw=0.6)
ax[1].axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2021-12-31"), color=ORANGE, alpha=0.12, lw=0)
ax[1].set_ylabel("First difference (thousands)")
ax[1].set_title("First difference of the series")
fig.tight_layout(); fig.savefig(f"{OUT}/figures/fig1_series.png"); plt.close(fig)

# ----------------------------------------------------- 3. stationarity tests
def adf_row(x, label, reg, autolag):
    x = x.dropna()
    stat, p, usedlag, nobs, crit, _ = adfuller(x, regression=reg, autolag=autolag)
    return dict(series=label, test=f"ADF ({'const' if reg=='c' else 'const+trend'}, {autolag} lags)",
                stat=stat, p=p, lags=usedlag, n=nobs, crit5=crit["5%"], null="unit root",
                reject5=bool(p < 0.05))

def kpss_row(x, label, reg):
    x = x.dropna()
    stat, p, lags, crit = kpss(x, regression=reg, nlags="auto")
    return dict(series=label, test=f"KPSS ({'level' if reg=='c' else 'trend'})",
                stat=stat, p=p, lags=lags, n=len(x), crit5=crit["5%"], null="stationary",
                reject5=bool(p < 0.05))

def battery(x, name, kp=True):
    out = []
    for reg in ("c", "ct"):
        for al in ("AIC", "BIC"):
            out.append(adf_row(x, name, reg, al))
    if kp:
        for reg in ("c", "ct"):
            out.append(kpss_row(x, name, reg))
    return out

lg = np.log(train)
tests = []
tests += battery(train, "Level")
tests += battery(train.diff(), "1st difference")
tests += battery(train.diff().diff(), "2nd difference")
tests += battery(lg, "log Level")
tests += battery(lg.diff(), "log 1st difference")
pre = train[train.index <= pd.Period("2019-12", "M")]
post = train[train.index >= pd.Period("2022-01", "M")]
tests += battery(pre, "Pre-COVID level (to Dec 2019)")
tests += battery(pre.diff(), "Pre-COVID 1st diff")
tests += battery(post, "Post-COVID level (from Jan 2022)")
tests += battery(post.diff(), "Post-COVID 1st diff")
T = pd.DataFrame(tests)
T.to_csv(f"{OUT}/tables/stationarity_tests.csv", index=False)
print(T[["series", "test", "stat", "p", "lags", "n", "reject5"]].round(3).to_string())

# Diagnostic: is the unit root a statistical artefact of the 3-month averaging?
# Persistence of the level: AR(1) coefficient on levels and on first differences.
rho_level = np.corrcoef(train.values[1:], train.values[:-1])[0, 1]
d1 = train.diff().dropna()
acf_d1 = acf(d1, nlags=24, fft=False)
pacf_d1 = pacf(d1, nlags=24, method="ywm")
acf_lv = acf(train, nlags=24, fft=False)
n_d1 = len(d1)
band = 1.96 / np.sqrt(n_d1)
print("ACF d1 lags1-6", acf_d1[1:7].round(3), "PACF d1 lags1-6", pacf_d1[1:7].round(3), "band", round(band, 3))

# figure: stationarity diagnostics (ACF/PACF levels vs differences)
fig, ax = plt.subplots(2, 3, figsize=(7.4, 4.8))
plot_acf(train, lags=24, ax=ax[0, 0], color=BLUE, vlines_kwargs={"colors": BLUE}, title="ACF: level")
plot_pacf(train, lags=24, ax=ax[0, 1], method="ywm", color=BLUE, vlines_kwargs={"colors": BLUE}, title="PACF: level")
ax[0, 2].plot(fmt_dates(train.index), train, color=BLUE); ax[0, 2].set_title("Level"); ax[0, 2].tick_params(axis="x", labelsize=7)
plot_acf(d1, lags=24, ax=ax[1, 0], color=ORANGE, vlines_kwargs={"colors": ORANGE}, title="ACF: 1st difference")
plot_pacf(d1, lags=24, ax=ax[1, 1], method="ywm", color=ORANGE, vlines_kwargs={"colors": ORANGE}, title="PACF: 1st difference")
ax[1, 2].plot(fmt_dates(d1.index), d1, color=ORANGE); ax[1, 2].axhline(0, color="black", lw=0.6)
ax[1, 2].set_title("1st difference"); ax[1, 2].tick_params(axis="x", labelsize=7)
for a in ax[:, :2].ravel():
    a.set_xlabel("Lag")
fig.tight_layout(); fig.savefig(f"{OUT}/figures/fig2_acf_pacf.png"); plt.close(fig)

# figure: rolling mean / sd of differences (stability of variance)
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
r = d1.rolling(24)
ax[0].plot(fmt_dates(d1.index), r.mean(), color=BLUE); ax[0].axhline(0, color="black", lw=0.6)
ax[0].set_title("24-month rolling mean of 1st difference")
ax[1].plot(fmt_dates(d1.index), r.std(), color=ORANGE)
ax[1].set_title("24-month rolling s.d. of 1st difference")
fig.tight_layout(); fig.savefig(f"{OUT}/figures/fig3_rolling.png"); plt.close(fig)

# ------------------------------------------------------ 4. ARIMA grid (d = 1)
def fit(y, order, trend):
    mod = ARIMA(y.values.astype(float), order=order, trend=trend)
    res = mod.fit()
    return res

def aicc(res, nobs):
    k = len(res.params)
    return res.aic + 2 * k * (k + 1) / (nobs - k - 1)

def lb(res, lags, p, q):
    r = res.resid[1:]  # first residual is the diffuse start-up value
    out = {}
    for L in lags:
        t = acorr_ljungbox(r, lags=[L], model_df=p + q, return_df=True)
        out[L] = float(t["lb_pvalue"].iloc[0]) if L > p + q else np.nan
    return out

grid = []
n_eff = len(train) - 1
for p in range(0, 5):
    for q in range(0, 5):
        for trend in ("n", "t"):
            try:
                res = fit(train, (p, 1, q), trend)
                k = len(res.params)
                lbp = lb(res, [6, 12, 24], p, q)
                grid.append(dict(p=p, d=1, q=q, drift=(trend == "t"), k=k, llf=res.llf,
                                 AIC=res.aic, AICc=aicc(res, n_eff), BIC=res.bic,
                                 sigma=np.sqrt(res.params[-1]),
                                 LB6=lbp[6], LB12=lbp[12], LB24=lbp[24],
                                 converged=bool(res.mle_retvals.get("converged", True))))
            except Exception as e:
                print("fail", p, q, trend, e)
G = pd.DataFrame(grid)
for c in ("AIC", "AICc", "BIC"):
    G[f"d{c}"] = G[c] - G[c].min()
G["rank_AIC"] = G["AIC"].rank().astype(int)
G["rank_AICc"] = G["AICc"].rank().astype(int)
G["rank_BIC"] = G["BIC"].rank().astype(int)
G["label"] = G.apply(lambda r: f"ARIMA({r.p},1,{r.q})" + (" + drift" if r.drift else ""), axis=1)
G.to_csv(f"{OUT}/tables/arima_grid_full.csv", index=False)
print("\nTop 15 by BIC"); print(G.sort_values("BIC").head(15)[["label", "k", "AIC", "AICc", "BIC", "LB12", "LB24", "converged"]].round(3).to_string())
print("\nTop 10 by AIC"); print(G.sort_values("AIC").head(10)[["label", "k", "AIC", "AICc", "BIC", "LB12", "LB24"]].round(3).to_string())
print("\nTop 10 by AICc"); print(G.sort_values("AICc").head(10)[["label", "k", "AIC", "AICc", "BIC", "LB12", "LB24"]].round(3).to_string())

# drift significance for the BIC-best no-drift vs drift pair
G.to_pickle(f"{OUT}/tables/grid.pkl")
json.dump(summary, open(f"{OUT}/tables/summary.json", "w"), indent=1)
T.to_pickle(f"{OUT}/tables/tests.pkl")
np.save(f"{OUT}/tables/acf_d1.npy", acf_d1); np.save(f"{OUT}/tables/pacf_d1.npy", pacf_d1)
print("\nrho_level", rho_level, "band", band)


# =========================================================== STAGE 2
# ---- helpers for LaTeX output
def pfmt(p, kind="adf"):
    if kind == "kpss":
        if p <= 0.0101: return "$\\leq$0.01"
        if p >= 0.0999: return "$\\geq$0.10"
        return f"{p:.3f}"
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"

def tex_write(name, s):
    open(f"{OUT}/tables/{name}.tex", "w").write(s)

# ---- Table: stationarity tests on the training sample
def cell(sub, test_prefix, with_lag=True):
    r = sub[sub.test.str.startswith(test_prefix)].iloc[0]
    kind = "kpss" if "KPSS" in r.test else "adf"
    s = f"{r.stat:.2f}".replace("-", "$-$") + f" ({pfmt(r.p, kind)})"
    return s

def stat_table(series_list, caption, label):
    lines = [r"\begin{table}[H]\centering\small", f"\\caption{{{caption}}}\\label{{{label}}}",
             r"\resizebox{\textwidth}{!}{\begin{tabular}{lcccccc}\toprule",
             r" & \multicolumn{4}{c}{ADF (H$_0$: unit root)} & \multicolumn{2}{c}{KPSS (H$_0$: stationary)}\\",
             r"\cmidrule(lr){2-5}\cmidrule(lr){6-7}",
             r"Series & const, AIC & const, BIC & c+trend, AIC & c+trend, BIC & level & trend\\\midrule"]
    for s in series_list:
        sub = T[T.series == s]
        row = [s.replace("log ", "log~").replace("1st", "1st").replace("2nd", "2nd")]
        row += [cell(sub, "ADF (const, AIC"), cell(sub, "ADF (const, BIC"),
                cell(sub, "ADF (const+trend, AIC"), cell(sub, "ADF (const+trend, BIC"),
                cell(sub, "KPSS (level"), cell(sub, "KPSS (trend")]
        lines.append(" & ".join(row) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}}",
              r"\par\smallskip\footnotesize Each cell: test statistic (p-value). ``AIC''/``BIC'' = rule used to choose the number of augmenting lags. "
              r"ADF 5\% critical values are about $-2.87$ (constant) and $-3.42$ (constant + trend). KPSS p-values are interpolated and truncated to [0.01, 0.10].",
              r"\end{table}"]
    return "\n".join(lines)

tex_write("stationarity_main", stat_table(["Level", "1st difference", "2nd difference", "log Level", "log 1st difference"],
          "Unit-root and stationarity tests on the training sample (June 2001 -- September 2024).", "tab:stat_main"))
tex_write("stationarity_sub", stat_table(["Pre-COVID level (to Dec 2019)", "Pre-COVID 1st diff", "Post-COVID level (from Jan 2022)", "Post-COVID 1st diff"],
          "Sub-sample tests. The post-COVID sample has only 33 observations, so these tests have little power.", "tab:stat_sub"))

# variance rule / over-differencing evidence
d2 = train.diff().diff().dropna()
overdiff = {"sd_level": float(train.std()), "sd_d1": float(d1.std()), "sd_d2": float(d2.std()),
            "acf1_d1": float(acf_d1[1]), "acf1_d2": float(acf(d2, nlags=2, fft=False)[1])}
print(overdiff)

# ---- shortlist
short = [((0, 0), False), ((0, 1), False), ((1, 1), False), ((0, 2), False), ((1, 2), False), ((2, 2), False),
         ((0, 3), False), ((1, 3), False), ((1, 4), False), ((1, 2), True)]
rows = []
fits = {}
for (p, q), dr in short:
    res = fit(train, (p, 1, q), "t" if dr else "n")
    fits[(p, q, dr)] = res
    g = G[(G.p == p) & (G.q == q) & (G.drift == dr)].iloc[0]
    r_ = res.resid[1:]
    lbp = acorr_ljungbox(r_, lags=[6, 12, 18, 24], model_df=p + q, return_df=True)["lb_pvalue"].values if p + q < 6 else [np.nan] * 4
    nsig = int((np.abs(acf(r_, nlags=24, fft=False)[1:]) > 1.96 / np.sqrt(len(r_))).sum())
    rows.append(dict(label=g.label, p=p, q=q, drift=dr, k=int(g.k), AIC=g.AIC, AICc=g.AICc, BIC=g.BIC,
                     sigma=g.sigma, LB6=lbp[0], LB12=lbp[1], LB18=lbp[2], LB24=lbp[3], nsig=nsig,
                     rank_BIC=int(g.rank_BIC), rank_AIC=int(g.rank_AIC)))
S = pd.DataFrame(rows)
S["dAIC"] = S.AIC - G.AIC.min(); S["dBIC"] = S.BIC - G.BIC.min()
S.to_csv(f"{OUT}/tables/shortlist.csv", index=False)
print(S.round(3).to_string())

def lbcell(v):
    if np.isnan(v): return "--"
    s = f"{v:.3f}"
    return f"\\textbf{{{s}}}" if v < 0.05 else s

L = [r"\begin{table}[H]\centering\small", r"\caption{Candidate ARIMA$(p,1,q)$ specifications on the training sample ($n=280$, 279 differenced observations). "
     r"$\Delta$ = difference from the best value in the full 50-model grid; bold Ljung--Box p-values are below 0.05.}\label{tab:candidates}",
     r"\resizebox{\textwidth}{!}{\begin{tabular}{lcrrrrrrrrr}\toprule",
     r"Model & $k$ & AIC & AICc & BIC & $\Delta$AIC & $\Delta$BIC & $\hat\sigma$ & LB(6) & LB(12) & LB(24)\\\midrule"]
for _, r in S.iterrows():
    nm = r.label.replace(" + drift", r"$+$drift").replace("ARIMA", "ARIMA")
    star = r"$^\star$" if (r.p, r.q, r.drift) == (1, 2, False) else ""
    L.append(f"{nm}{star} & {r.k} & {r.AIC:.1f} & {r.AICc:.1f} & {r.BIC:.1f} & {r.dAIC:.1f} & {r.dBIC:.1f} & {r.sigma:.2f} & "
             f"{lbcell(r.LB6)} & {lbcell(r.LB12)} & {lbcell(r.LB24)}\\\\")
L += [r"\bottomrule", r"\end{tabular}}",
      r"\par\smallskip\footnotesize $k$ counts all estimated parameters including the innovation variance. LB($m$): Ljung--Box p-value using $m$ lags and $m-p-q$ degrees of freedom. $^\star$ recommended. "
      r"ARIMA(0,1,0) has no estimated mean, so $k=1$ (the variance) and LB uses no degree-of-freedom correction.",
      r"\end{table}"]
tex_write("candidates", "\n".join(L))

# top-15 by BIC across the full grid (appendix)
top = G.sort_values("BIC").head(15)
L = [r"\begin{longtable}{lcrrrrrr}\caption{Top 15 of the 50 grid models ranked by BIC.}\label{tab:top15}\\\toprule",
     r"Model & $k$ & AIC & AICc & BIC & LB(12) & LB(24) & Converged\\\midrule\endfirsthead"]
for _, r in top.iterrows():
    L.append(f"{r.label.replace(' + drift', '$+$drift')} & {int(r.k)} & {r.AIC:.1f} & {r.AICc:.1f} & {r.BIC:.1f} & {lbcell(r.LB12)} & {lbcell(r.LB24)} & {'yes' if r.converged else 'no'}\\\\")
L += [r"\bottomrule\end{longtable}"]
tex_write("top15", "\n".join(L))

# ---- heatmap of delta BIC (no drift)
Gn = G[~G.drift].pivot(index="p", columns="q", values="BIC")
dB = Gn - G.BIC.min()
fig, ax = plt.subplots(figsize=(4.6, 3.6))
im = ax.imshow(np.minimum(dB.values, 40), cmap="Blues_r", vmin=0, vmax=40, origin="upper")
for i in range(dB.shape[0]):
    for j in range(dB.shape[1]):
        v = dB.values[i, j]
        ax.text(j, i, f"{v:.0f}" if v < 100 else ">99", ha="center", va="center", fontsize=8,
                color="white" if v < 14 else "#111827", fontweight="bold" if v == 0 else "normal")
ax.set_xticks(range(5)); ax.set_yticks(range(5)); ax.set_xlabel("MA order q"); ax.set_ylabel("AR order p")
ax.grid(False); cb = fig.colorbar(im, ax=ax, shrink=0.85); cb.set_label("$\\Delta$BIC vs best (capped at 40)")
ax.set_title("BIC of ARIMA(p,1,q), no drift")
fig.tight_layout(); fig.savefig(f"{OUT}/figures/fig4_bic_heatmap.png"); plt.close(fig)

# ---- recommended model: estimates and diagnostics
REC = (1, 2, False)
rec = fits[REC]
names = ["ar.L1", "ma.L1", "ma.L2", "sigma2"]
ci = rec.conf_int()
est = pd.DataFrame({"coef": rec.params, "se": rec.bse, "z": rec.tvalues, "p": rec.pvalues,
                    "lo": ci[:, 0], "hi": ci[:, 1]}, index=names)
print(est.round(4))
ma_roots = np.roots([est.loc["ma.L2", "coef"], est.loc["ma.L1", "coef"], 1.0])
print("MA roots", ma_roots, np.abs(ma_roots))
L = [r"\begin{table}[H]\centering\small", r"\caption{Maximum-likelihood estimates, ARIMA(1,1,2), no drift, training sample.}\label{tab:params}",
     r"\begin{tabular}{lrrrrr}\toprule Parameter & Estimate & Std.\ error & $z$ & p-value & 95\% CI\\\midrule"]
pn = {"ar.L1": r"$\phi_1$ (AR 1)", "ma.L1": r"$\theta_1$ (MA 1)", "ma.L2": r"$\theta_2$ (MA 2)", "sigma2": r"$\sigma^2$"}
for n_, r in est.iterrows():
    L.append(f"{pn[n_]} & {r.coef:.3f} & {r.se:.3f} & {r.z:.2f} & {pfmt(r.p)} & [{r.lo:.3f}, {r.hi:.3f}]\\\\")
L += [r"\bottomrule\end{tabular}", r"\end{table}"]
tex_write("params", "\n".join(L))

resid = pd.Series(rec.resid, index=train.index).iloc[1:]
sd_all = resid.std()
z = resid / sd_all
cov = (resid.index >= pd.Period("2020-03", "M")) & (resid.index <= pd.Period("2021-12", "M"))
pre_ = resid.index < pd.Period("2020-03", "M"); post_ = resid.index > pd.Period("2021-12", "M")
reg_sd = {"pre": float(resid[pre_].std()), "covid": float(resid[cov].std()), "post": float(resid[post_].std()),
          "n_pre": int(pre_.sum()), "n_cov": int(cov.sum()), "n_post": int(post_.sum())}
lbt = acorr_ljungbox(resid, lags=[6, 12, 18, 24, 36], model_df=3, return_df=True)
lbsq = acorr_ljungbox(resid ** 2, lags=[12], return_df=True)
arch = het_arch(resid, nlags=6)
jb = stats.jarque_bera(resid)
jb_nc = stats.jarque_bera(resid[~cov])
a_res = acf(resid, nlags=36, fft=False)
bandr = 1.96 / np.sqrt(len(resid))
sig_lags = [(i, float(a_res[i])) for i in range(1, 37) if abs(a_res[i]) > bandr]
dw = float(np.sum(np.diff(resid) ** 2) / np.sum(resid ** 2))
big = [(str(resid.index[i]), float(z.iloc[i])) for i in np.where(np.abs(z) > 3)[0]]
diag = dict(n=len(resid), mean=float(resid.mean()), sd=float(sd_all), skew=float(stats.skew(resid)),
            exkurt=float(stats.kurtosis(resid)), jb=float(jb[0]), jb_p=float(jb[1]), jb_nc=float(jb_nc[0]), jb_nc_p=float(jb_nc[1]),
            skew_nc=float(stats.skew(resid[~cov])), exkurt_nc=float(stats.kurtosis(resid[~cov])),
            arch_lm=float(arch[0]), arch_p=float(arch[1]), lbsq12_p=float(lbsq["lb_pvalue"].iloc[0]),
            lb={int(k): [float(v1), float(v2)] for k, v1, v2 in zip(lbt.index, lbt["lb_stat"], lbt["lb_pvalue"])},
            sig_lags=sig_lags, band=float(bandr), dw=dw, big=big, reg_sd=reg_sd,
            ma_root_mod=[float(x) for x in np.abs(ma_roots)], ma_roots=[complex(x).__repr__() for x in ma_roots])
print(json.dumps(diag, indent=1))
json.dump({"summary": summary, "overdiff": overdiff, "diag": diag, "est": est.to_dict()}, open(f"{OUT}/tables/results.json", "w"), indent=1, default=str)
resid.to_csv(f"{OUT}/tables/recommended_residuals.csv")

L = [r"\begin{table}[H]\centering\small", r"\caption{Residual diagnostics for ARIMA(1,1,2) (279 residuals; the first, a diffuse start-up value, is dropped).}\label{tab:diag}",
     r"\begin{tabular}{lrr}\toprule Test & Statistic & p-value\\\midrule"]
for m in [6, 12, 18, 24, 36]:
    L.append(f"Ljung--Box, {m} lags (df $={m}-3$) & {diag['lb'][m][0]:.2f} & {lbcell(diag['lb'][m][1])}\\\\")
L.append(f"Ljung--Box on squared residuals, 12 lags & {lbsq['lb_stat'].iloc[0]:.2f} & {lbcell(float(lbsq['lb_pvalue'].iloc[0]))}\\\\")
L.append(f"Engle ARCH-LM, 6 lags & {arch[0]:.2f} & {lbcell(float(arch[1]))}\\\\")
L.append(f"Jarque--Bera, all residuals & {jb[0]:.1f} & {lbcell(float(jb[1]))}\\\\")
L.append(f"Jarque--Bera, excluding Mar 2020--Dec 2021 & {jb_nc[0]:.1f} & {lbcell(float(jb_nc[1]))}\\\\")
L += [r"\bottomrule\end{tabular}", r"\end{table}"]
tex_write("diag", "\n".join(L))

# ---- figure: residual diagnostics (4 panels)
fig, ax = plt.subplots(2, 2, figsize=(7.4, 5.6))
ax[0, 0].plot(fmt_dates(z.index), z, color=BLUE, lw=1)
ax[0, 0].axhline(0, color="black", lw=0.6)
for h_ in (-3, 3): ax[0, 0].axhline(h_, color=RED, lw=0.7, ls="--")
ax[0, 0].axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2021-12-31"), color=ORANGE, alpha=0.12, lw=0)
ax[0, 0].annotate("Apr 2020", (pd.Period("2020-04", "M").to_timestamp(), big[0][1]), fontsize=7, xytext=(-8, 0), textcoords="offset points", ha="right", va="center")
ax[0, 0].text(pd.Timestamp("2022-03-01"), 4.4, "3 more |z|>3\n(Jul 2020, May and Aug 2021)", fontsize=6.5, va="center", ha="left")
ax[0, 0].set_title("Standardised residuals"); ax[0, 0].set_ylabel("z")
plot_acf(resid, lags=36, ax=ax[0, 1], color=BLUE, vlines_kwargs={"colors": BLUE}, title="Residual ACF")
ax[0, 1].set_xlabel("Lag")
ax[1, 0].hist(z, bins=40, density=True, color=BLUE, alpha=0.8, edgecolor="white", linewidth=0.4)
xs = np.linspace(-6, 6, 300); ax[1, 0].plot(xs, stats.norm.pdf(xs), color=ORANGE, lw=1.4, label="N(0,1)")
ax[1, 0].set_xlim(-8, 6); ax[1, 0].set_title("Histogram of standardised residuals"); ax[1, 0].legend(frameon=False)
qqplot(z.values, line="45", ax=ax[1, 1], markerfacecolor=BLUE, markeredgecolor=BLUE, markersize=3, alpha=0.8)
ax[1, 1].get_lines()[1].set_color(ORANGE); ax[1, 1].set_title("Normal Q-Q plot")
fig.tight_layout(); fig.savefig(f"{OUT}/figures/fig5_residuals.png"); plt.close(fig)

# ---- figure: Ljung-Box p-values by lag for a few candidates
fig, ax = plt.subplots(figsize=(6.6, 3.2))
cols = {(1, 2, False): BLUE, (0, 2, False): GREY, (1, 3, False): ORANGE, (1, 4, False): RED}
for key, c in cols.items():
    res = fits[key]; p, q, _ = key
    r_ = res.resid[1:]
    lg_ = np.arange(p + q + 1, 37)
    pv = acorr_ljungbox(r_, lags=list(lg_), model_df=p + q, return_df=True)["lb_pvalue"].values
    ax.plot(lg_, pv, color=c, label=f"ARIMA({p},1,{q})", lw=1.6 if key == REC else 1.0)
ax.axhline(0.05, color="black", ls="--", lw=0.8); ax.text(36.3, 0.05, "0.05", va="center", fontsize=7.5)
ax.set_xlabel("Number of lags m"); ax.set_ylabel("Ljung--Box p-value".replace("--", "–")); ax.set_ylim(0, 1)
ax.set_title("Ljung–Box p-value by number of lags"); ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.22))
fig.tight_layout(); fig.savefig(f"{OUT}/figures/fig6_ljungbox.png"); plt.close(fig)

# ---- figure: fitted vs actual first differences, last 12 years of training
fit_d = pd.Series(rec.fittedvalues, index=train.index).diff()
fig, ax = plt.subplots(figsize=(7.0, 2.8))
sel = train.index >= pd.Period("2012-01", "M")
ax.plot(fmt_dates(train.index[sel]), d1.reindex(train.index)[sel], color=GREY, lw=1.0, label="Actual change")
ax.plot(fmt_dates(train.index[sel]), (pd.Series(rec.fittedvalues, index=train.index) - train.shift(1))[sel], color=BLUE, lw=1.2, label="One-step fitted change")
ax.axhline(0, color="black", lw=0.6); ax.legend(frameon=False); ax.set_ylabel("Thousands")
ax.set_title("ARIMA(1,1,2): one-step-ahead fitted vs actual monthly change (training sample, 2012 on)")
fig.tight_layout(); fig.savefig(f"{OUT}/figures/fig7_fitted.png"); plt.close(fig)

# ---- robustness: alternative treatments, grid p,q in 0..4, d=1, no drift
def grid_best(y_, exog=None, label=""):
    out = []
    for p in range(5):
        for q in range(5):
            try:
                res = ARIMA(np.asarray(y_, float), order=(p, 1, q), exog=exog).fit()
                kx = 0 if exog is None else exog.shape[1]
                lbp = acorr_ljungbox(res.resid[1:], lags=[12, 24], model_df=p + q, return_df=True)["lb_pvalue"].values if p + q < 12 else [np.nan] * 2
                out.append(dict(p=p, q=q, AIC=res.aic, BIC=res.bic, LB12=lbp[0], LB24=lbp[1], params=res.params.round(3).tolist()))
            except Exception:
                pass
    D = pd.DataFrame(out)
    b = D.sort_values("BIC").iloc[0]; a = D.sort_values("AIC").iloc[0]
    # best by BIC among those passing LB at 12 and 24
    ok = D[(D.LB12 > 0.05) & (D.LB24 > 0.05)].sort_values("BIC")
    return dict(label=label, bic_best=f"({int(b.p)},1,{int(b.q)})", aic_best=f"({int(a.p)},1,{int(a.q)})", n=len(y_),
                LB12=float(b.LB12), LB24=float(b.LB24), rec_in_top3=bool(((b.p, b.q) == (1, 2)) or ((1, 2) in [(int(x.p), int(x.q)) for _, x in D.sort_values('BIC').head(3).iterrows()])),
                pass_best=(f"({int(ok.iloc[0].p)},1,{int(ok.iloc[0].q)})" if len(ok) else "none"),
                dBIC_12=float(D[(D.p == 1) & (D.q == 2)].BIC.iloc[0] - D.BIC.min())), D

rob = []
r1, D1 = grid_best(train, label="Training sample, levels (baseline)"); rob.append(r1)
r2, D2 = grid_best(np.log(train), label="Training sample, log levels"); rob.append(r2)
r3, D3 = grid_best(pre, label="Pre-COVID only (Jun 2001--Dec 2019)"); rob.append(r3)
ao_dates = [d_ for d_, _ in big]
X = pd.DataFrame({d_: (train.index == pd.Period(d_, "M")).astype(float) for d_ in ao_dates}, index=train.index)
r4, D4 = grid_best(train, exog=X.values, label="Training sample + 4 outlier dummies"); rob.append(r4)
R = pd.DataFrame(rob); R.to_csv(f"{OUT}/tables/robustness.csv", index=False); print(R.to_string())
L = [r"\begin{table}[H]\centering\small", r"\caption{Does the choice of specification survive alternative treatments? Each row repeats the 25-model ARIMA$(p,1,q)$ grid ($0\le p,q\le4$, no drift).}\label{tab:robust}",
     r"\resizebox{\textwidth}{!}{\begin{tabular}{lccrrr}\toprule Treatment & Best BIC & Best AIC & $\Delta$BIC of (1,1,2) & LB(12) & LB(24)\\\midrule"]
for _, r in R.iterrows():
    L.append(f"{r.label} & {r.bic_best} & {r.aic_best} & {r.dBIC_12:.1f} & {lbcell(r.LB12)} & {lbcell(r.LB24)}\\\\")
L += [r"\bottomrule\end{tabular}}", r"\par\smallskip\footnotesize LB columns refer to the best-BIC model of that row. Dummies are single-month additive outliers at "
      + ", ".join(pd.Period(d_, "M").strftime("%b %Y") for d_ in ao_dates) + r" (months with $|z|>3$ in the baseline residuals; chosen after inspecting residuals, so this is a sensitivity check, not an independent test).", r"\end{table}"]
tex_write("robust", "\n".join(L))

# parameter stability under the dummy treatment
res_ao = ARIMA(train.values.astype(float), order=(1, 1, 2), exog=X.values).fit()
ao_est = pd.DataFrame({"coef": res_ao.params, "se": res_ao.bse}, index=res_ao.param_names)
print(ao_est.round(3))
r_ao = res_ao.resid[1:]
ao_lb = acorr_ljungbox(r_ao, lags=[6, 12, 18, 24], model_df=3, return_df=True)["lb_pvalue"].round(3).tolist()
print("AO-dummy LB", ao_lb)
json.dump({"ao_est": ao_est.round(4).to_dict(), "ao_lb": ao_lb, "ao_sd": float(r_ao.std())}, open(f"{OUT}/tables/ao_results.json", "w"), indent=1)

# drift term significance
rd = fits[(1, 2, True)]
print("drift", dict(zip(rd.param_names, rd.params.round(3))), dict(zip(rd.param_names, rd.pvalues.round(3))))
# AR(1) persistence on levels (supports unit root)
rl = ARIMA(train.values.astype(float), order=(1, 0, 0), trend="c").fit()
print("AR(1) on levels phi", rl.params.round(4), rl.bse.round(4))

# ---- levels vs log: Jacobian-adjusted comparison (valid on the scale of the original data)
ly = np.log(train.astype(float))
jac = 2 * ly.values[1:].sum()          # AIC_levels-equivalent = AIC_log + 2*sum(log y_t), t = 2..n
cmp_rows = []
for tag, s, order in [("Levels", train.astype(float), (1, 2)), ("Levels", train.astype(float), (0, 2)),
                      ("Log levels", ly, (0, 2)), ("Log levels", ly, (1, 2))]:
    p, q = order
    r = ARIMA(s.values, order=(p, 1, q)).fit()
    adj = jac if tag == "Log levels" else 0.0
    e = pd.Series(r.resid, index=train.index).iloc[1:]
    cmp_rows.append(dict(scale=tag, model=f"ARIMA({p},1,{q})", AIC=r.aic + adj, BIC=r.bic + adj,
                         LB12=float(acorr_ljungbox(e, lags=[12], model_df=p + q, return_df=True)["lb_pvalue"].iloc[0]),
                         LB24=float(acorr_ljungbox(e, lags=[24], model_df=p + q, return_df=True)["lb_pvalue"].iloc[0]),
                         ARCH=float(het_arch(e, nlags=6)[1])))
C = pd.DataFrame(cmp_rows); C["dBIC"] = C.BIC - C.BIC.min(); print(C.round(3).to_string())
C.to_csv(f"{OUT}/tables/levels_vs_log.csv", index=False)
L = [r"\begin{table}[H]\centering\small", r"\caption{Levels versus log levels. Log-scale AIC/BIC have been converted to the scale of the original data by adding $2\sum_t \log y_t$, so the four rows are directly comparable.}\label{tab:lvslog}",
     r"\begin{tabular}{llrrrrrr}\toprule Scale & Model & Adj.\ AIC & Adj.\ BIC & $\Delta$BIC & LB(12) & LB(24) & ARCH-LM\\\midrule"]
for _, r in C.iterrows():
    L.append(f"{r.scale} & {r.model} & {r.AIC:.1f} & {r.BIC:.1f} & {r.dBIC:.1f} & {lbcell(r.LB12)} & {lbcell(r.LB24)} & {lbcell(r.ARCH)}\\\\")
L += [r"\bottomrule\end{tabular}", r"\end{table}"]
tex_write("lvslog", "\n".join(L))

# ---- pre-COVID residual check for the recommended model (referenced in the report)
rp = ARIMA(pre.values.astype(float), order=(1, 1, 2)).fit()
pre_lb = acorr_ljungbox(rp.resid[1:], lags=[12, 24], model_df=3, return_df=True)["lb_pvalue"].round(4).tolist()
print("pre-COVID ARIMA(1,1,2) LB(12), LB(24):", pre_lb)
print("residual ACF lags 1-9:", np.round(a_res[1:10], 3).tolist(), "max abs", np.abs(a_res[1:10]).max().round(3))
