from __future__ import annotations

import argparse, glob, math, os, sys, yaml, copy
from pathlib import Path
from dataclasses import dataclass
from typing import Iterable, List

import numpy as np
import pandas as pd
import numpy_financial as nfin
from  scipy.optimize import brentq

# ─────────────────────── configuration ─────────────────────── #
CFG = yaml.safe_load(Path("config.yml").read_text())

# ───────────────── helper functions ────────────────────────── #
def ann_cap_cost(capex: float, disc: float, life: int, real: bool) -> float:
    """
    Annualised capital recovery (equity portion).  
    If `real=False`, converts nominal WACC → real using cfg infl_rate.
    """
    if not real:
        disc = (1 + disc)/(1 + CFG["infl_rate"]) - 1
    equity = capex * (1 - CFG["loan_ltv"])
    return -nfin.pmt(disc, life, equity)


def load_prices(folder: str) -> pd.Series:
    """Concat CAISO 15-min CSVs into a single $/MWh time-series."""
    frames: list[pd.Series] = []
    for fp in glob.glob(os.path.join(folder, "*.csv")):
        df = pd.read_csv(fp, skiprows=3, engine="python")
        ts = pd.to_datetime(df.iloc[:, 0], errors="coerce", utc=True)
        col = next(c for c in df.columns
                   if "ZP-26" in c.upper() and "LMP" in c.upper())
        price = pd.to_numeric(df[col], errors="coerce")
        ok    = ts.notna() & price.notna()
        frames.append(price[ok].set_axis(ts[ok]))
    if not frames:
        sys.exit("ERROR: no CAISO CSVs found.")
    return pd.concat(frames).sort_index()


def decline_curve(life: int) -> np.ndarray:
    c = CFG; cf0, cf_min = c["perf_cf0"], c["perf_min_cf"]
    d1, d2, t_sw = c["decl_d1"], c["decl_d2"], c["decl_t_switch"]
    vals = []
    for t in range(life):
        if t < t_sw:
            vals.append(cf0 / (1 + d1 * t))
        else:
            base = cf0 / (1 + d1 * t_sw)
            vals.append(max(base * ((1 - d2) ** (t - t_sw)), cf_min))
    return np.asarray(vals)


def draw_overruns(base: float, n: int) -> np.ndarray:
    mu, sig = CFG["capex_overrun_mu"], CFG["capex_overrun_sigma"]
    mu_ln = math.log(1 + mu) - 0.5 * sig**2
    return base * np.random.lognormal(mu_ln, sig, n)


def price_multipliers(years: int) -> np.ndarray:
    μ, σ, dt = CFG["price_vol_mu"], CFG["price_vol_sigma"], 1.0
    shocks = np.random.normal((μ - 0.5*σ**2)*dt, σ*math.sqrt(dt), years)
    return np.exp(np.cumsum(shocks))


def curt_prob(year: int) -> float:
    return max(0, min(1, CFG["curtail_p0"] + CFG["curtail_slope"]*(year-1)))


# ───────────── debt sculpting helper ───────────── #
@dataclass
class DebtSchedule:
    ltv: float; rate: float; tenor: int; life: int
    def sculpt(self, capex: float, pre_ds_cash: list[float]) -> tuple[list[dict], float]:
        min_dscr = CFG["dsc_min"]; ltv = self.ltv
        while ltv > 0:
            bal  = capex * ltv
            pmt  = -nfin.pmt(self.rate, self.tenor, bal)
            rows, ok = [], True
            for yr in range(1, self.life+1):
                if yr > self.tenor:
                    ds, intr = 0.0, bal * self.rate
                else:
                    intr  = bal * self.rate
                    princ = max(pmt - intr, 0)
                    bal   = max(bal - princ, 0)
                    ds    = pmt
                if (pre_ds_cash[yr-1] + intr) / max(ds,1e-9) < min_dscr:
                    ok = False; break
                rows.append({"DS": ds, "Int": intr})
            if ok:
                rows[self.tenor-1]["DS"] += bal * CFG["refi_fee"]
                return rows, ltv
            ltv = round(ltv - 0.05, 2)
        return [{"DS":0.0,"Int":0.0} for _ in pre_ds_cash], 0.0


# ─────────────────── core project model ─────────────────── #
def build_case(
    mw: float, mi: float, credit: str, prices: pd.Series, price_mult: np.ndarray,
    price_floor: float | None, disc: float, life: int, basis0: float,
) -> dict:

    c = CFG
    expl_cost = (1 - c["expl_success_prob"]) * c["dry_hole_cost_m"] * 1e6

    # ----- CAPEX & IDC -------------------------------------------------------- #
    gross_nom  = mw*c["plant_capex_mw"] + mi*c["line_capex_mi"] + c["interconnect_fixed"]
    draws = np.array(c["draw_weights"], float)
    draws /= draws.sum()
    idc = sum(draw * gross_nom * c["idc_rate"] * 0.5 for draw in draws)
    gross_capex = gross_nom + idc
    itc_rate  = 0.30 if credit == "itc30" else 0.0
    net_capex = gross_capex * (1 - itc_rate)
    macrs_basis = gross_capex * (0.5 if itc_rate else 1.0)

    # ----- price reference ---------------------------------------------------- #
    eff_p = prices - basis0
    if price_floor is not None:
        eff_p = eff_p.clip(lower=price_floor)
    p0 = eff_p.median() * (1 - c["merchant_penalty"])

    cf_vec = decline_curve(life)
    first_mwh = mw*8760*cf_vec[0]
    cap_y1  = c["capacity_payment_kw"]*mw*1_000
    byp_y1  = c["byproduct_base"]*mw/25
    o_fix_y1 = c["o_and_m_fixed_kw"]*mw*1_000
    o_var_y1 = c["o_and_m_var_kw"] *mw*1_000
    rec_y1  = 0.0 if credit=="ptc" else c["rec_price_y1"]*first_mwh

    # ----- operating cash pre-debt ------------------------------------------- #
    pre_ds: List[float] = []
    for yr in range(1, life+1):
        mwh = mw*8760*cf_vec[yr-1]*(1 - curt_prob(yr))
        basis_adj = basis0*(1 + c["basis_escal"])**(yr-1)
        price_real = max(p0*price_mult[yr-1] - basis_adj, price_floor or -1e9)

        rec_price = 0 if credit=="ptc" else c["rec_price_y1"]*((1+c["rec_esc"])**(yr-1))
        energy = price_real * (mwh/first_mwh)
        rec    = rec_price * mwh
        cap    = cap_y1*(yr<=c["capacity_payment_yrs"])*(1+c["ra_escal"])**(yr-1)
        ptc    = c["ptc_rate"]*mwh if credit=="ptc" and yr<=10 else 0
        gross  = energy + rec + cap + byp_y1

        o_m = (o_fix_y1*(1+c["esc_o_and_m_fixed"])**(yr-1) +
               o_var_y1*(1+c["esc_o_and_m_var"])**(yr-1))
        if yr % c["well_rework_int"] == 0:
            o_m += c["well_rework_frac"]*net_capex

        royalty = gross * (c["royalty_rate1"] if yr < c["royalty_switch"]
                           else c["royalty_rate2"])

        dep = macrs_basis*(c["macrs_5"][yr-1] if yr<=len(c["macrs_5"]) else 0)
        assess_ratio = max(c["assessed_pct"]*(1-0.05*(yr-1)), c["nv_tax_floor"])
        tax_cost_basis = gross_nom
        prop_tax = c["property_tax_rate"] * assess_ratio * tax_cost_basis
        taxable  = max(gross - royalty - o_m - dep, 0)

        pre_ds.append(gross + ptc - royalty - o_m -
                      (taxable*c["tax_rate"] + taxable*c["net_proceeds_tax"] + prop_tax))

    # ----- debt schedule ------------------------------------------------------ #
    debt_rows, ltv_used = DebtSchedule(c["loan_ltv"], c["loan_rate"],
                                       c["loan_tenor"], life).sculpt(net_capex, pre_ds)

    # ----- final cash-flow list (build yrs + ops) ----------------------------- #
    cash = [-expl_cost]*3
    itc_left = gross_capex*itc_rate
    for yr, base in enumerate(pre_ds, 1):
        ds, intr = debt_rows[yr-1]["DS"], debt_rows[yr-1]["Int"]
        if itc_left>0:
            off = min(base*c["tax_rate"], itc_left)
            base += off; itc_left -= off
        if yr == life:
            base += c["salvage_frac"]*net_capex
            base -= c["decom_frac"]*net_capex
        cash.append(base - ds - intr)

    delivered = mw*8760*cf_vec*(1 - np.array([curt_prob(y) for y in range(1,life+1)]))
    avg_mwh = delivered.mean()

    y1 = {"Energy_$":p0*first_mwh/1e6,
          "Capacity_$":cap_y1/1e6,
          "REC_$":rec_y1/1e6,
          "Byprod_$":byp_y1/1e6,
          "PTC_$":(first_mwh*c["ptc_rate"]/1e6) if credit=="ptc" else 0,
          "O&M_$":-(o_fix_y1+o_var_y1)/1e6}

    return dict(net_capex=net_capex+expl_cost, cash=cash,
                avg_mwh=avg_mwh, y1=y1, ltv=ltv_used)


# ─────────── robust IRR helper ─────────── #
def robust_irr(cf: list[float]) -> float:
    if all(x<=0 for x in cf) or all(x>=0 for x in cf): return float("nan")
    try:  return brentq(lambda r: nfin.npv(r, cf), -0.9, 1.0, maxiter=1_000)
    except ValueError: return float("nan")


# ─────────── Monte-Carlo grid ─────────── #
def run_grid(cap: Iterable[float], miles: Iterable[float], credits: Iterable[str],
             prices: pd.Series, price_floor: float, disc: float, life: int,
             basis: float, disc_mode:str="real") -> pd.DataFrame:

    real = (disc_mode=="real")
    rows = []
    for mw in cap:
        for mi in miles:
            for cr in credits:
                price_path = price_multipliers(life)
                case = build_case(mw, mi, cr, prices, price_path,
                                  price_floor, disc, life, basis)

                overruns = draw_overruns(case["net_capex"], CFG["mc_trials"])
                disc_vec = (1+disc)**np.arange(1, life+4)
                base_npv = np.sum(np.array(case["cash"])/disc_vec)
                npvs     = -overruns + base_npv

                p50_irr = robust_irr([-np.median(overruns)] + case["cash"])

                pct = lambda v,p: round(np.percentile(v,p)/1e6,1)
                rows.append({
                    "MW":mw,"Line_mi":mi,"Credit":cr,
                    "NPV_P10_M":pct(npvs,10),"NPV_P50_M":pct(npvs,50),"NPV_P90_M":pct(npvs,90),
                    "IRR_P50_pct":round(p50_irr*100,1),
                    "LCOE_$":round((ann_cap_cost(case["net_capex"],disc,life+3,real)+
                                    (CFG["o_and_m_fixed_kw"]+CFG["o_and_m_var_kw"])*mw*1_000) /
                                   case["avg_mwh"],2),
                    "Avg_CF_pct":round(case["avg_mwh"]/(mw*8_760)*100,1),
                    **case["y1"]})
    return pd.DataFrame(rows)


# ─────────── sensitivity sweep ─────────── #
def run_sensitivity(base_args: dict, prices: pd.Series):
    cases=[]
    for capex_k in [0.85,1.0,1.15]:
        for price_k in [0.8,1.0,1.2]:
            for rec_k in [0.5,1.0,1.5]:
                for d_shift in [-0.02,0.0,0.02]:
                    a = copy.deepcopy(base_args)
                    a["discount"]=max(0.0,a["discount"]+d_shift)
                    CFG["plant_capex_mw"]*=capex_k
                    CFG["rec_price_y1"]  *=rec_k
                    df=run_grid(a["capacities"],a["line_miles"],a["tax_credits"],
                                prices,a["price_floor"],a["discount"],
                                a["life"],a["basis_adj"],a["disc_mode"])
                    cases.append({"CAPEX_k":capex_k,"PRICE_k":price_k,"REC_k":rec_k,
                                  "Disc":a["discount"],
                                  "Best_P50_NPV_M":df["NPV_P50_M"].max()})
                    CFG["plant_capex_mw"]/=capex_k
                    CFG["rec_price_y1"]  /=rec_k
    pd.DataFrame(cases).to_csv("sensitivity.csv",index=False)
    print("sensitivity.csv written")


# ───────────── CLI & entry-point ───────────── #
def cli()->argparse.Namespace:
    p=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data_dir",required=True)
    p.add_argument("--capacities",nargs="+",type=float,default=[25,30,40,50])
    p.add_argument("--line_miles",nargs="+",type=float,default=[5,10,20])
    p.add_argument("--tax_credits",nargs="+",choices=["itc30","ptc"],default=["itc30","ptc"])
    p.add_argument("--price_floor",type=float,default=None)
    p.add_argument("--discount",type=float,default=0.09)
    p.add_argument("--disc_mode",choices=["real","nom"],default="real",
                   help="Is --discount a real or nominal rate?")
    p.add_argument("--life",type=int,default=30)
    p.add_argument("--basis_adj",type=float,default=CFG["basis_adjust"])
    p.add_argument("--sens",action="store_true",
                   help="Run ± sensitivity sweep and write sensitivity.csv")
    return p.parse_args()


def main()->None:
    args=cli()
    prices=load_prices(args.data_dir)

    df=run_grid(args.capacities,args.line_miles,args.tax_credits,
                prices,args.price_floor,args.discount,args.life,
                args.basis_adj,args.disc_mode)
    df.to_csv("results.csv",index=False)

    if args.sens:
        run_sensitivity(vars(args),prices)

    best=df.loc[df["NPV_P50_M"].idxmax()]
    flag="GREEN" if best["NPV_P50_M"]>0 else ("AMBER" if best["NPV_P50_M"]>-25 else "RED")
    print("\n================  INVESTOR SUMMARY  ================\n"
          f"Flag       : {flag}\n"
          f"Optimal    : {best['MW']} MW, {best['Line_mi']} mi, {best['Credit']}\n"
          f"P50 NPV    : {best['NPV_P50_M']:+.1f} M$ | IRR₅₀: {best['IRR_P50_pct']:.1f}%\n"
          f"LCOE       : {best['LCOE_$']:.2f} $/MWh | Avg CF: {best['Avg_CF_pct']:.1f}%\n")
    print(df.head(10).to_string(index=False))
    print("\nresults.csv written\n")

if __name__=="__main__":
    main()
