"""
Functions specificlly for PR1 assay and plotting from a FED device
William B. Rubio
"""

# import dependancies
from dataclasses import dataclass
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
from pathlib import Path
import tqdm
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import sem
from itertools import cycle
from IPython.display import display


# Import cousins
from fedverse.fedutils.fedlog import status
from fedverse.extracted import fed3bandit_extracted, fed3_loading, fed3_fedframe
from fedverse import fedassets
from fedverse.fedcore import core





#@@@@@@@@@@@@@@@@@@ INDIVIDUAL PLOTS @@@@@@@@@@@@@@@@@@#
def pr1_indv_plots(fed_list, metadata_df, root_path, dpi=150):
    """
    Returns (fig, filename) for UI to display/save.
    Prints a message and returns (None, None) on errors.
    """

    status.step("Creating individual PR1 Plots")

    # Safety check
    #if 'feds' not in globals() or file_index >= len(feds):
    #    print(f"Index {file_index} is out of range (max {len(feds)-1 if 'feds' in globals() else 'N/A'}).")
    #    return None, None
    
    # intitate a empty list to capture the results
    saved_paths  = []

    # Create the directory that will be writen to
    out_dir = Path(root_path, "indv_pr1_plots")
    out_dir.mkdir(parents=True, exist_ok=True)
    

    for fed in fed_list:
        df = fed.copy()
        # Keep only pellet rows, but don't mutate df
        pellet_df = df[df['Event'] == 'Pellet'].copy()

        filename = getattr(df, 'name', f"File_{fed}")

        # Match Mouse_ID from metadata, robust to path vs basename
        #filename = os.path.basename(str(raw_file))
        title_str = filename

        if 'filename' in metadata_df.columns:
            md_fn = metadata_df['filename'].astype(str).map(os.path.basename)
            hit = metadata_df.loc[md_fn == filename]
            if not hit.empty:
                match_row = hit.iloc[0]
                if 'Mouse_ID' in match_row and pd.notna(match_row['Mouse_ID']):
                    title_str = f"{match_row['Mouse_ID']}"


        if pellet_df.empty:
            status.warn(f"No pellet events for file index {title_str}.")
            return None, None

        

        # Determine x-axis
        x_series = None
        x_is_datetime = isinstance(pellet_df.index, pd.DatetimeIndex)
        if x_is_datetime:
            x_series = pellet_df.index
        else:
            for col in ['Timestamp', 'Time', 'DateTime', 'Datetime', 'datetime']:
                if col in pellet_df.columns:
                    ts = pd.to_datetime(pellet_df[col], errors='coerce')
                    if ts.notna().any():
                        pellet_df['_x'] = ts
                        x_series = pellet_df['_x']
                        x_is_datetime = True
                        break
            if x_series is None:
                x_series = pellet_df.index  # fall back to index

        # Build figure (DO NOT show here)
        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(111)

        hue_vals = pellet_df['Block_Pellet_Count'].clip(upper=40)
        sns.scatterplot(
            data=pellet_df,
            x=x_series,
            y='Block_Pellet_Count',
            hue=hue_vals,
            palette='spring',
            alpha=0.6,
            legend=False,
            ax=ax
        )

        # Night shading if datetime x
        if x_is_datetime:
            import datetime as dt
            night_start = dt.time(18, 0)  # 18:00
            night_end   = dt.time(6, 0)   # 06:00

            start_date = pd.to_datetime(pd.Series(x_series)).min().normalize()
            end_date   = pd.to_datetime(pd.Series(x_series)).max().normalize()

            # shade from each day's 18:00 to next day's 06:00
            for day in pd.date_range(start_date, end_date - pd.Timedelta(days=1)):
                start = pd.Timestamp.combine(day, night_start)
                end   = pd.Timestamp.combine(day + pd.Timedelta(days=1), night_end)
                ax.axvspan(start, end, color='gray', alpha=0.2)

            import matplotlib.dates as mdates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            ax.set_xlabel('')
        else:
            ax.set_xlabel('Index')

        # set axis values
        ax.set_ylabel('Pellet Count in Block')
        ax.set_title(title_str)
        fig.tight_layout()


        ### Save Plots ###
        # suggest a base filename for saving
        safe_title = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in title_str)
        suggested = f"{safe_title}_indvPR1"

        # assign path
        out_path = out_dir / suggested
        
        # save the figure
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")

        # Close the current figure to save memory
        plt.close(fig)

        saved_paths.append(out_path)
        status.ok(f"saved {out_path.name}")


    status.ok(f"Plots saved to: {out_dir} ")



    # Return for UI to display/save later
    return saved_paths





#@@@@@@@@@@@@@@@@@@ PR METRICS @@@@@@@@@@@@@@@@@@#

### ------ General Helper functions ------- ###

def _file_base_lower(pathlike):
    return os.path.splitext(os.path.basename(str(pathlike)))[0].lower()

def _to_num(s):
    return pd.to_numeric(s, errors="coerce")

def _count_events(df, label: str) -> int:
    if "Event" not in df.columns:
        return 0
    return int((df["Event"].astype(str) == label).sum())

def breakpoint_and_runs(df):
    """
    Returns: (median_breakpoint, max_breakpoint, runs_count)
    end-of-block: BPC > 0 now and next BPC == 0
    """
    if "Block_Pellet_Count" not in df.columns or df["Block_Pellet_Count"].empty:
        return np.nan, np.nan, 0

    bpc = _to_num(df["Block_Pellet_Count"])
    end_mask = (bpc > 0) & (bpc.shift(-1) == 0)
    end_vals = bpc[end_mask]

    med_bp = float(end_vals.median()) if len(end_vals) else np.nan
    max_bp = float(bpc.max()) if bpc.notna().any() else np.nan
    runs   = int(end_mask.sum())
    return med_bp, max_bp, runs


### ------ Compute PR Metrics ------- ###
def compute_pr_metrics(fed_list, md, min_runs_per_mice = 9):
    """
    Computes metrics for the PR task.

    On Reproducibility:
    the demand curve paramters may differ ~1e-8 across environments: this is 'trf'
    solver convergence noise (default tol 1e-8) from differing BLAS/scipy
    builds. Counts are bit-identical. 
    e.g. Different machine environments use different arithmetic programs that 
    handle rounding differently. As curve_fit approaches a near-flat 
    optimum, those tiny rounding differences cause it to stop at slightly 
    different points.

    Arguments:
        fed_list; List
        md; Dataframe
        min_runs_per_mice; Int


    """
    # copy variables
    feds = fed_list.copy()

    rows = []

    def _tqdm(x, **kwargs): return x
    
    #for idx, c_df in enumerate(_tqdm(feds, desc="Computing PR metrics")):
    for idx in tqdm.tqdm(range(len(fed_list))):
        c_df = fed_list[idx]
        file_name = core._basename(getattr(c_df, "name", f"File_{idx}"))

        pellets = _count_events(c_df, "Pellet")
        left    = _count_events(c_df, "Left")
        right   = _count_events(c_df, "Right")
        total   = left + right

        acc = (left / total * 100.0) if total > 0 else np.nan
        ppp = (total / pellets) if pellets > 0 else np.nan

        med_bp, max_bp, runs = breakpoint_and_runs(c_df)
        daily = core._estimate_daily_pellets(c_df)

        rows.append({
            "filename": file_name,
            "Left_Poke": left,
            "Right_Poke": right,
            "Total_Pokes": total,
            "Accuracy": acc,
            "PokesPerPellet": ppp,
            "MedianBreakPoint": med_bp,
            "MaxBreakPoint": max_bp,
            "Numberofblocks": runs,
            "Daily_Pellets": daily,   # use consistent name (recommended)
        })

    PRmetrics = pd.DataFrame(rows)

    # check for error in dataframe creation.
    if PRmetrics.empty:
        status.error("No files to analyze!")
        raise SystemExit
    
    #return PRmetrics
    status.ok("First set of metrics calculated")
    status.preview(PRmetrics)

    ### Calculate the demand fits ###

    if "filename" in md.columns and "Mouse_ID" in md.columns:
        mouse_map = md.dropna(subset=["filename"]).drop_duplicates("filename").set_index("filename")["Mouse_ID"]
        PRmetrics["Mouse_ID"] = PRmetrics["filename"].map(mouse_map)
    else:
        PRmetrics["Mouse_ID"] = np.nan

    # substring fallback if still missing
    if PRmetrics["Mouse_ID"].isna().any() and ("Mouse_ID" in md.columns):
        known_ids = md["Mouse_ID"].dropna().unique().tolist()
        for i, r in PRmetrics.loc[PRmetrics["Mouse_ID"].isna()].iterrows():
            base = _file_base_lower(r["filename"])
            hits = [mid for mid in known_ids if str(mid).lower() in base]
            if len(hits) == 1:
                PRmetrics.at[i, "Mouse_ID"] = hits[0]
            elif len(hits) > 1:
                longest = max(len(str(h)) for h in hits)
                best = [h for h in hits if len(str(h)) == longest]
                if len(best) == 1:
                    PRmetrics.at[i, "Mouse_ID"] = best[0]

    # build filename->df map
    file_names = [core._basename(getattr(df, "name", f"File_{i}")) for i, df in enumerate(feds)]
    file_to_df = dict(zip(file_names, feds))

    #MIN_RUNS_PER_MOUSE = 10
    runs_per_mouse = (
        PRmetrics.dropna(subset=["Mouse_ID"])
        .groupby("Mouse_ID")["Numberofblocks"]
        .sum()
        .to_dict()
    )
    keep_mice = {m for m, r in runs_per_mouse.items() if r >= min_runs_per_mice}
    for m, r in runs_per_mouse.items():
        if m not in keep_mice:
            status.warn(f"[{m}] skipped: only {r} blocks (< {min_runs_per_mice}).")

    raw_rows = []
    for fn, df in file_to_df.items():
        mouse = PRmetrics.loc[PRmetrics["filename"] == fn, "Mouse_ID"]
        mouse = mouse.iloc[0] if len(mouse) else None
        if (mouse is None) or (mouse not in keep_mice):
            continue
        if {"Event", "Block_Pellet_Count"} - set(df.columns):
            status.warn(f"[{mouse}] file {fn} missing Event/BPC columns; skipped.")
            continue

        pellets_df = df[df["Event"].astype(str) == "Pellet"].copy()
        if pellets_df.empty:
            status.warn(f"[{mouse}] file {fn} has 0 pellet rows; skipped.")
            continue

        bpc = pd.to_numeric(pellets_df["Block_Pellet_Count"], errors="coerce")
        counts = (
            pellets_df.assign(Block_Pellet_Count=bpc)
            .dropna(subset=["Block_Pellet_Count"])
            .groupby("Block_Pellet_Count")
            .size()
        )
        if counts.empty:
            status.warn(f"[{mouse}] file {fn} produced no valid price bins; skipped.")
            continue

        for price, cnt in counts.items():
            raw_rows.append({"Mouse_ID": mouse, "PricePaid": int(price), "PelletCount": int(cnt)})

    demand_raw_df = pd.DataFrame(raw_rows)

    if demand_raw_df.empty:
        demand_metrics = pd.DataFrame(columns=[
            "Mouse_ID", "Demand_Q0_raw", "Demand_alpha_raw", "Demand_beta_raw",
            "Demand_alpha_FR", "Demand_beta_FR",
        ])
    else:
        demand_mouse = (demand_raw_df
                        .groupby(["Mouse_ID", "PricePaid"], as_index=False)["PelletCount"]
                        .sum())

        def _baseline_B(sub):
            if sub.empty:
                return np.nan
            mprice = sub["PricePaid"].min()
            return sub.loc[sub["PricePaid"] == mprice, "PelletCount"].mean()

        B = (demand_mouse.groupby("Mouse_ID")
            .apply(_baseline_B).rename("B").reset_index())
        B["q"] = 100.0 / B["B"]

        normalized = (
            demand_mouse
            .merge(B[["Mouse_ID", "q"]], on="Mouse_ID", how="left")
            .assign(P_FR=lambda d: d["PricePaid"],
                    Q_norm=lambda d: d["PelletCount"] * d["q"])
        )

        def loglog3(P, Q0, a, b):
            return Q0 / (1.0 + (P / a) ** b)

        def loglogN(P, a, b):
            return 100.0 / (1.0 + (P / a) ** b)

        raw_fit_rows = []
        for mouse, sub in demand_mouse.groupby("Mouse_ID"):
            P = sub["PricePaid"].to_numpy(float)
            Q = sub["PelletCount"].to_numpy(float)
            if np.unique(P).size < 3:
                status.warn(f"[{mouse}] demand fit skipped (raw): need ≥3 unique FR levels.")
                continue
            try:
                popt, _ = curve_fit(
                    loglog3, P, Q,
                    p0=[Q.max(), float(np.median(P)), 2.0],
                    bounds=([1e-6, 1e-6, 1e-3], [np.inf, np.inf, 50.0]),
                    maxfev=20000
                )
                raw_fit_rows.append({
                    "Mouse_ID": mouse,
                    "Demand_Q0_raw": float(popt[0]),
                    "Demand_alpha_raw": float(popt[1]),
                    "Demand_beta_raw": float(popt[2]),
                })
            except Exception as e:
                status.warn(f"[{mouse}] demand curve fit failed (raw): {e}")
        demand_fit_raw = pd.DataFrame(raw_fit_rows)

        fr_fit_rows = []
        for mouse, sub in normalized.groupby("Mouse_ID"):
            P = sub["P_FR"].to_numpy(float)
            Q = sub["Q_norm"].to_numpy(float)
            if np.unique(P).size < 3:
                status.warn(f"[{mouse}] demand fit skipped (norm-Q): need ≥3 unique FR levels.")
                continue
            try:
                popt, _ = curve_fit(
                    loglogN, P, Q,
                    p0=[float(np.median(P)), 2.0],
                    bounds=([1e-6, 1e-3], [np.inf, 50.0]),
                    maxfev=20000
                )
                fr_fit_rows.append({
                    "Mouse_ID": mouse,
                    "Demand_alpha_FR": float(popt[0]),
                    "Demand_beta_FR": float(popt[1]),
                })
            except Exception as e:
                status.warn(f"[{mouse}] demand curve fit failed (norm-Q): {e}")
        demand_fit_fr = pd.DataFrame(fr_fit_rows)

        demand_metrics = demand_fit_raw.merge(demand_fit_fr, on="Mouse_ID", how="outer")

    # merge demand into PRmetrics (Mouse-level)
    PRmetrics = PRmetrics.merge(demand_metrics, on="Mouse_ID", how="left")

    status.ok("PR metric demand curves calculated")
    status.preview(PRmetrics)

    # return
    return PRmetrics



#@@@@@@@@@@@@@@@@@@ GROUPED DEMAND CURVES @@@@@@@@@@@@@@@@@@#


def plot_group_mean_demand_with_params(mapping_df, pm_grps_df, md, x_colors,
                                       group_col='XGroup', mapping_obj_name='files_to_group_both',
                                       price_grid=None, n_grid=120, P_min=1.0, fallback_max_P=128,
                                       annotate_x_frac=0.18):
    """
    Arguments:
        mapping_df; Dataframe
            Dataframe tying each filename to their XGroup and HueGroup
        pm_grps_df; Dataframe
            Wide dataframe that contains the metric data per mouse along with the xgroup and hue group 
        md; Dataframe
            cleaned metadata dataframe
    """

    # --- mapping ---
    mapping = mapping_df.copy()
    pm = pm_grps_df.copy()


    if group_col not in mapping.columns:
        raise ValueError(f"mapping does not contain group column '{group_col}'")

    # filename -> Mouse_ID

    #if 'md' in globals() and 'filename' in md.columns and 'Mouse_ID' in md.columns:
    #    fname_to_mouse = md.set_index('filename')['Mouse_ID'].to_dict()
    #elif 'pm' in globals() and 'filename' in pm.columns and 'Mouse_ID' in pm.columns:
    #    fname_to_mouse = pm.set_index('filename')['Mouse_ID'].to_dict()
    #else:
    #    raise RuntimeError("Cannot find filename -> Mouse_ID mapping (need `md` or `bm`).")

    if  'filename' in md.columns and 'Mouse_ID' in md.columns:
        fname_to_mouse = md.set_index('filename')['Mouse_ID'].to_dict()
    elif 'filename' in pm.columns and 'Mouse_ID' in pm.columns:
        fname_to_mouse = pm.set_index('filename')['Mouse_ID'].to_dict()
    else:
        status.fail("Cannot find filename -> Mouse_ID mapping (need `md` or `bm`).")

    # demand params table
    #if 'pm' not in globals():
    #    raise RuntimeError("Need `pm` (PRmetrics_merged copy) with Demand_alpha_FR & Demand_beta_FR.")
    
    metrics = pm.copy()
    alpha_col = 'Demand_alpha_FR'
    beta_col  = 'Demand_beta_FR'
    
    if alpha_col not in metrics.columns or beta_col not in metrics.columns:
        raise RuntimeError(f"`pm` must contain '{alpha_col}' and '{beta_col}'.")

    # normalize mapping file basenames
    mapping['filename'] = mapping['filename'].astype(str).apply(lambda p: os.path.basename(str(p)))

    # choose price grid if not provided
    if price_grid is None:
        maxP_candidates = []
        max_alpha = metrics[alpha_col].dropna().max()
        if pd.notna(max_alpha):
            maxP_candidates.append(float(max_alpha) * 8.0)
        if 'demand_raw_df' in globals() and 'PricePaid' in demand_raw_df.columns:
            maxP_candidates.append(int(demand_raw_df['PricePaid'].max()))
        maxP = int(max(maxP_candidates) if maxP_candidates else fallback_max_P)
        maxP = max(maxP, P_min + 1)
        price_grid = np.unique(np.logspace(np.log10(max(P_min, 1.0)), np.log10(maxP), n_grid))
    else:
        price_grid = np.asarray(price_grid, dtype=float)

    def predict_Q(P, alpha, beta):
        P = np.asarray(P, dtype=float)
        return 100.0 / (1.0 + (P / alpha) ** beta)

    
    groups = mapping[group_col].dropna().unique().tolist()

    
    # Determine colors
    #if 'group_colors' in globals() and isinstance(group_colors, dict):
    #    color_map = {g: group_colors.get(g, None) for g in groups}
    #else:
    #    color_map = {g: None for g in groups}
    color_map = {
        g: (x_colors[g].value if hasattr(x_colors[g], "value") else x_colors[g])
        for g in groups
        if g in x_colors
    }

    default_cycle = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])

    # ONE figure / axes for all groups
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.close(fig)

    stat_rows = []
    for gi, g in enumerate(groups):
        subset = mapping.loc[mapping[group_col] == g, 'filename'].unique().tolist()
        mice = [fname_to_mouse.get(os.path.basename(f), None) for f in subset]
        unique_mice = sorted({m for m in mice if pd.notna(m)})

        mice_with_params = []
        for m in unique_mice:
            row = metrics.loc[metrics['Mouse_ID'] == m]
            if row.empty:
                continue
            a = row[alpha_col].values[0]
            b = row[beta_col].values[0]
            if pd.isna(a) or pd.isna(b):
                continue
            if float(a) <= 0 or float(b) <= 0:
                continue
            mice_with_params.append((m, float(a), float(b)))

        if len(mice_with_params) == 0:
            status.warn(f"[Group '{g}'] no mice with valid alpha/beta — skipped.")
            continue

        Qs = np.vstack([predict_Q(price_grid, a, b) for (_m,a,b) in mice_with_params])
        mean_Q = np.nanmean(Qs, axis=0)
        sem_Q  = sem(Qs, axis=0, nan_policy='omit')

        color = color_map.get(g)
        if not color:
            color = next(default_cycle)

        alphas = np.array([a for (_m,a,b) in mice_with_params])
        betas  = np.array([b for (_m,a,b) in mice_with_params])
        a_mean, a_sem = alphas.mean(), sem(alphas)
        b_mean, b_sem = betas.mean(), sem(betas)

        Q_meanparams_at_alpha = predict_Q(a_mean, a_mean, b_mean)
        ix_closest = np.argmin(np.abs(price_grid - a_mean))
        P_meancurve = price_grid[ix_closest]
        Q_meancurve = mean_Q[ix_closest]

        print(f"[{g}]")
        print(f"  alpha_mean = {a_mean:.4f}, beta_mean = {b_mean:.4f}")
        print(f"  Q_meanparams(P = alpha_mean): P = {a_mean:.4f}, Q = {Q_meanparams_at_alpha:.4f}")
        print(f"  Q_meancurve (nearest alpha_mean): P = {P_meancurve:.4f}, Q = {Q_meancurve:.4f}")
        print()

        # curve + SEM
        ax.plot(price_grid, mean_Q, label=g, color=color, linewidth=2.8)
        ax.fill_between(price_grid, mean_Q - sem_Q, mean_Q + sem_Q,
                        alpha=0.22, edgecolor='none', linewidth=0, facecolor=color)

        # marker & vertical line at alpha
        x_alpha = P_meancurve
        y_alpha = Q_meancurve
        ax.vlines(x_alpha, 0, y_alpha, linestyle='--', linewidth=1.6,
                  color=color, alpha=0.8)
        ax.plot(x_alpha, y_alpha, 'o', color=color, markersize=6)

        # text, alternating above/below
        text_x = x_alpha * 1.05
        vertical_offset = 4
        direction = 1 if gi % 2 == 0 else -1
        text_y = y_alpha + direction * vertical_offset
        text = rf"α={a_mean:.2f}, β={b_mean:.3f}"
        ax.text(text_x, text_y, text, color=color, fontsize=10,
                ha='left', va='center')

        stat_rows.append({
            'Group': g,
            'N_mice': len(mice_with_params),
            'alpha_mean': a_mean, 'alpha_sem': a_sem,
            'beta_mean': b_mean,  'beta_sem': b_sem
        })

    # axes / legend
    ax.set_xscale('log')
    ax.set_xlabel('Food Price (FR)')
    ax.set_yticks(np.arange(0, 101, 20)); ax.set_ylim(0, 105)
    ax.set_xlim(1, 100)
    ticks = np.array([1,2,3,5,10,15,20,30,40,50,75,100], float)
    ax.set_xticks(ticks); ax.set_xticklabels([str(int(x)) for x in ticks])
    ax.set_ylabel('Consumption')
    ax.set_title('')
    ax.spines['left'].set_position(('data', 1))
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    ax.tick_params(axis='both', which='both', width=1.5, length=6)

    ax.legend(title="Group", frameon=False)

    plt.tight_layout()

    stat_df = pd.DataFrame(stat_rows).sort_values('Group').reset_index(drop=True)

    if not stat_df.empty:
        status.warn("Stats table emtpy")
        display(stat_df[['Group','N_mice','alpha_mean','alpha_sem','beta_mean','beta_sem']].round(3))
    
    return fig, stat_df




#@@@@@@@@@@@@@@@@@@ PR L4 @@@@@@@@@@@@@@@@@@#


### ------ L4 drawing cores ------- ###

def _plot_pr_trace_core(df, ax, *, cmap="spring", show_xlabel=True, night_shade=True):
    """
    Draw ONE mouse's PR "pellet histogram" onto a caller-supplied Axes: each earned
    pellet as a dot whose height is how many pellets deep into the current block it
    was (Block_Pellet_Count), so the rising sawtooth shows effort ramping across the
    progressive ratio before each reset. Colored by block depth and (optionally)
    night-shaded. Creates no figure and saves nothing -- the caller owns the Axes.

    This is the composable core behind pr1_indv_plots, reused by assemble_pr_l4 so
    the L4 example trace and the standalone per-mouse plots stay identical.

    Arguments:
        df : DataFrame
            A single FED session (must carry Event / Block_Pellet_Count).
        ax : matplotlib Axes
            Target axes to draw into.
        cmap : str
            Palette for the block-depth hue.
        show_xlabel : bool
            Whether to keep an x-axis label (off inside a tight composite).
        night_shade : bool
            Shade 18:00->06:00 grey when the x-axis is real datetime.
    """
    pellet_df = df[df["Event"] == "Pellet"].copy()
    if pellet_df.empty:
        status.warn("L4: example mouse has no pellet events; blank trace.")
        ax.axis("off")
        return

    # --- Resolve an x-axis: prefer a real datetime so we can night-shade ---
    x_is_dt = isinstance(pellet_df.index, pd.DatetimeIndex)
    if x_is_dt:
        x_series = pd.Series(pellet_df.index, index=pellet_df.index)
    else:
        x_series = None
        for col in ["Timestamp", "Time", "DateTime", "Datetime", "datetime"]:
            if col in pellet_df.columns:
                ts = pd.to_datetime(pellet_df[col], errors="coerce")
                if ts.notna().any():
                    x_series = ts
                    x_is_dt = True
                    break
        if x_series is None:
            # Fall back to a plain running index (no shading possible).
            x_series = pd.Series(np.arange(len(pellet_df)), index=pellet_df.index)

    hue_vals = pellet_df["Block_Pellet_Count"].clip(upper=40)
    sns.scatterplot(
        x=x_series, y=pellet_df["Block_Pellet_Count"],
        hue=hue_vals, palette=cmap, s=16, alpha=0.7,
        edgecolor="none", legend=False, ax=ax,
    )

    if x_is_dt:
        import datetime as dt
        night_start, night_end = dt.time(18, 0), dt.time(6, 0)

        start_date = pd.to_datetime(x_series).min().normalize()
        end_date   = pd.to_datetime(x_series).max().normalize()

        # --- Night shading: each night spans that day's 18:00 -> the next day's
        # 06:00. Stop one day short of the last day (end_date - 1) so no empty
        # trailing night is painted past the end of the data (matches pr1_indv_plots).
        if night_shade:
            for day in pd.date_range(start_date, end_date - pd.Timedelta(days=1)):
                start = pd.Timestamp.combine(day, night_start)
                end   = pd.Timestamp.combine(day + pd.Timedelta(days=1), night_end)
                ax.axvspan(start, end, color="gray", alpha=0.18, zorder=0)

        # --- X-axis "Day N" labels. For every day that has a following dark cycle,
        # center the label on the light->dark transition (18:00). The final day has
        # no dark cycle drawn (the shading loop stops one day early), so fall back to
        # centering on its light segment (noon). Same fallback if shading is off.
        days = pd.date_range(start_date, end_date)
        tick_locs = []
        for i, day in enumerate(days):
            has_dark = night_shade and (i < len(days) - 1)
            if has_dark:
                tick_locs.append(pd.Timestamp.combine(day, night_start))   # dark-cycle transition
            else:
                tick_locs.append(day + pd.Timedelta(hours=12))             # light-segment center
        ax.set_xticks(tick_locs)
        ax.set_xticklabels([f"Day {i + 1}" for i in range(len(days))])
        ax.tick_params(axis="x", length=0)     # keep the day labels, drop tick marks
        ax.set_xlabel("")
    else:
        ax.set_xlabel("Index" if show_xlabel else "")

    ax.set_ylabel("Pellets earned", fontsize=13)
    ax.set_title("")
    sns.despine(ax=ax)


def _predict_demand_Q(P, alpha, beta):
    """Exponentiated-demand consumption: Q = 100 / (1 + (P/alpha)**beta)."""
    P = np.asarray(P, dtype=float)
    return 100.0 / (1.0 + (P / alpha) ** beta)


def _demand_curve_core(long_df, ax, color_map, group_order, *,
                       alpha_var="Demand_alpha_FR", beta_var="Demand_beta_FR",
                       P_min=1.0, P_max=100.0, n_grid=120, annotate=True,
                       annotate_top_right=False):
    """
    Draw the grouped mean demand curve onto a caller-supplied Axes, rebuilt straight
    from the melted metric frame. For every mouse we already carry a fitted alpha and
    beta (as two `variable` rows), so we pivot those back to one row per mouse, predict
    Q over a shared log price grid, and plot the per-group mean +- SEM. A dashed marker
    and an "alpha / slope" annotation sit at each group's mean alpha, matching
    plot_group_mean_demand_with_params without needing the mapping/grouped frames.

    Assumes long_df["XGroup"] is ALREADY relabeled to the final display labels and that
    color_map / group_order are keyed by those labels.

    Arguments:
        annotate : bool
            Master toggle for drawing the alpha/slope labels at all.
        annotate_top_right : bool
            True (default) -> stack every group's alpha/slope label in the top-right
            corner, one per group, so labels never overlap the curves or each other.
            False -> place each group's label in a fixed corner ranked by mean alpha:
            lowest alpha top-left, 2nd-lowest bottom-left, 3rd top-right, highest
            bottom-right. Only as many corners as there are groups are used, so 2 and
            3 groups work too.
    """
    dsub = long_df[long_df["variable"].isin([alpha_var, beta_var])]
    if dsub.empty:
        status.warn("L4: no demand parameters in long_df; panel C left blank.")
        ax.axis("off")
        return

    wide = dsub.pivot_table(
        index=["Mouse_ID", "XGroup"], columns="variable",
        values="value", aggfunc="first",
    ).reset_index()

    if alpha_var not in wide.columns or beta_var not in wide.columns:
        status.warn("L4: demand alpha/beta missing after pivot; panel C left blank.")
        ax.axis("off")
        return

    price_grid = np.unique(np.logspace(np.log10(max(P_min, 1.0)), np.log10(P_max), n_grid))

    # Collect each drawn group's label (with its mean alpha) and place them all after
    # the loop, so the ranked-corner layout can see every group's alpha at once.
    annot_records = []                     # list of (a_mean, x_a, y_a, color, label)
    for g in group_order:
        rows = wide.loc[wide["XGroup"] == g]
        # Keep only physically meaningful fits (positive alpha and beta).
        pairs = [
            (float(a), float(b))
            for a, b in zip(rows[alpha_var], rows[beta_var])
            if pd.notna(a) and pd.notna(b) and float(a) > 0 and float(b) > 0
        ]
        if not pairs:
            status.warn(f"L4: group '{g}' has no valid demand fits; skipped in panel C.")
            continue

        Qs = np.vstack([_predict_demand_Q(price_grid, a, b) for a, b in pairs])
        mean_Q = np.nanmean(Qs, axis=0)
        sem_Q = sem(Qs, axis=0, nan_policy="omit")

        color = color_map.get(g, "tab:blue")
        ax.plot(price_grid, mean_Q, color=color, lw=2.5, label=g)
        ax.fill_between(price_grid, mean_Q - sem_Q, mean_Q + sem_Q,
                        color=color, alpha=0.2, linewidth=0)

        # Dashed drop-line + annotation at the group's mean alpha.
        a_mean = float(np.mean([a for a, _ in pairs]))
        b_mean = float(np.mean([b for _, b in pairs]))
        ix = int(np.argmin(np.abs(price_grid - a_mean)))
        x_a, y_a = price_grid[ix], mean_Q[ix]
        ax.vlines(x_a, 0, y_a, linestyle="--", linewidth=1.4, color=color, alpha=0.85)
        ax.plot(x_a, y_a, "o", color=color, markersize=5)

        if annotate:
            label = f"α = {a_mean:.2f}\nslope = {b_mean:.2f}"
            annot_records.append((a_mean, x_a, y_a, color, label))

    # --- Place the alpha/slope labels ---
    if annotate and annot_records:
        if annotate_top_right:
            # Stack every group's label in the top-right corner (axes coords), one
            # block per group in draw order, colored to match its curve.
            for slot, (_a, _x, _y, color, label) in enumerate(annot_records):
                ax.text(0.83, 0.95 - slot * 0.16, label,
                        transform=ax.transAxes, color=color, fontsize=9,
                        ha="left", va="top")
        else:
            # Rank by mean alpha (ascending) and drop each label into a fixed corner:
            # lowest -> top-left, 2nd -> bottom-left, 3rd -> top-right, 4th -> bottom-
            # right. zip stops at the number of groups, so 2 and 3 groups just fill the
            # first 2 / 3 corners.
            corners = [
                (0.20, 0.55, "left",  "top"),      # lowest alpha  -> top-left
                (0.25, 0.30, "left",  "bottom"),   # 2nd lowest    -> bottom-left
                (0.80, 0.65, "right", "top"),      # 3rd (next)    -> top-right
                (0.90, 0.40, "right", "bottom"),   # highest       -> bottom-right
            ]
            ranked = sorted(annot_records, key=lambda r: r[0])   # ascending mean alpha
            for (cx, cy, cha, cva), (_a, _x, _y, color, label) in zip(corners, ranked):
                ax.text(cx, cy, label, transform=ax.transAxes, color=color,
                        fontsize=9, ha=cha, va=cva)

    # --- Axes cosmetics: log price axis, Hursh-style ticks (matches the demand fig) ---
    ax.set_xscale("log")
    ax.set_xlabel("Food Price (FR)", fontsize=13)
    ax.set_ylabel("Consumption", fontsize=13)
    ax.set_ylim(0, 105)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_xlim(1, 100)
    ticks = np.array([1, 2, 3, 5, 10, 15, 20, 30, 40, 50, 75, 100], float)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(int(t)) for t in ticks], fontsize=8)
    ax.legend(title="", frameon=False, loc="lower left", fontsize=9)
    sns.despine(ax=ax)


def _pick_example_fed(long_df, fed_list, metadata_df, *, target_group=None,
                      min_blocks=10):
    """
    Choose one representative mouse for the panel-B example trace
    among mice in the requested genotype group that ran at least `min_blocks` blocks.
     
    Take the one with the highest MedianBreakPoint (the fullest, most visually rich sawtooth) whose FED
    session we can actually resolve.

    The group / block constraints are applied strictly first, then relaxed in steps
    (drop the block floor, then drop the group) so we always return SOME usable mouse
    rather than a blank panel.

    Arguments:
        target_group : str | None
            Final display label the example must belong to (e.g. the het label). None
            means "any group". Must match long_df["XGroup"] AFTER relabeling.
        min_blocks : int
            Minimum Numberofblocks required for the preferred candidates.

    Returns (fed_df | None, mouse_id | None).
    """
    # --- Pivot the two per-mouse metrics we filter/sort on back to wide ---
    sub = long_df[long_df["variable"].isin(["Numberofblocks", "MedianBreakPoint"])]
    wide = sub.pivot_table(
        index=["Mouse_ID", "XGroup"], columns="variable",
        values="value", aggfunc="first",
    ).reset_index()

    # Rank by break point (fullest sawtooth first); guard if the column is absent.
    sort_col = "MedianBreakPoint" if "MedianBreakPoint" in wide.columns else None

    def _ranked_ids(df):
        df = df.dropna(subset=[sort_col]) if sort_col else df
        if sort_col:
            df = df.sort_values(sort_col, ascending=False)
        return df["Mouse_ID"].astype(str).tolist()

    # Build progressively looser candidate pools, most-constrained first.
    in_group = wide if target_group is None else wide[wide["XGroup"] == target_group]
    if "Numberofblocks" in in_group.columns:
        enough_blocks = in_group[in_group["Numberofblocks"] >= min_blocks]
    else:
        enough_blocks = in_group

    for pool in (enough_blocks, in_group, wide):     # group+blocks -> group -> anything
        for mid in _ranked_ids(pool):
            fed = core._fed_for_mouse(fed_list, metadata_df, mid)
            if fed is not None and "Event" in fed.columns and (fed["Event"] == "Pellet").any():
                return fed, mid

    # Last resort: first session with pellet rows, whatever its Mouse_ID.
    for fed in fed_list:
        if "Event" in fed.columns and (fed["Event"] == "Pellet").any():
            return fed, str(os.path.basename(str(getattr(fed, "name", "example"))))
    return None, None








### ------ L4 assembly ------- ###

def assemble_pr_l4(long_df, x_colors, ordered_x, bm_md, root_path,
                   fed_list, metadata_df, *, schematic_path=None, dpi=300, 
                   figsize=(12, 9), h_space = 0.45, font_size=14
                   ):
    """
    Assemble the composite "L4" deliverable figure for one knockout model on the
    Progressive Ratio (PR1) task, mirroring the published PR figure:

        A) FED3 + PR task schematic (device image) with the gene name as the title
        B) one example mouse's pellet histogram (earned pellet vs. effort/block depth,
           night-shaded)
        C) grouped mean demand curve with per-group alpha / slope annotations
        D) Daily pellets bar
        E) Total pokes bar
        F) Median break point bar
        G) Demand alpha bar
        H) Demand slope (beta) bar

    plus a single shared Sex (Female/Male) legend and a caption. Every bar panel
    reports the genotype main effect from a two-way ANOVA, matching the caption.

    Orchestration only: the drawing lives in _plot_pr_trace_core, _demand_curve_core
    and _plot_metric_display so the layout logic here stays readable.

    Arguments:
        long_df : DataFrame
            Melted PR metrics (columns: variable / XGroup / HueGroup / value /
            Mouse_ID). Supplies BOTH the bar panels and -- via the per-mouse
            Demand_alpha_FR / Demand_beta_FR rows -- the grouped demand curve.
        x_colors : dict
            Group -> ipywidgets color widget (plain strings also tolerated).
        ordered_x : list
            X-group order hint from the aesthetics step (WT is forced first).
        bm_md : DataFrame
            Per-session metadata; supplies the gene name (bm_md["Gene"][0]).
        root_path : str | Path
            Output root; the figure is written under <root_path>/L4.
        fed_list : list
            FED session DataFrames, used to draw the example trace directly.
        metadata_df : DataFrame
            Stitched metadata, used to map Mouse_ID -> FED session for the trace.
        bandittype : str | None
            Accepted for call-site compatibility with the bandit L4; ignored here
            (PR1 always uses the pr1 schematic).
        schematic_path : str | Path | None
            Override for the panel-A image. Defaults to the packaged pr1 schematic.
        dpi : int
            Save resolution.
        figsize : tuple
            Figure width, height in inches.
        h_space : float
            Vertical space between the top row and the bottom row.
        font_size : float
            Font size for the caption text.

    Returns:
        out_path : Path
            Path to the saved composite SVG.
    """

    status.step("Assembling PR1 L4 composite figure")

    out_dir = Path(root_path, "L4")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Gene name that replaces the generic "HET" label throughout the figure.
    genename = bm_md["Gene"][0]

    long_df = long_df.copy()

    # --- Relabel zygosities to display form (identical convention to bandit L4) ---
    # Single mutant   -> bare gene name on the bar.
    # Multiple mutants -> zygosity on the bar, gene name on the x-axis.
    zygotic_order = ["HET", "HOM", "HEMI"]
    zyg_display   = {"HET": "Het", "HOM": "Hom", "HEMI": "Hemi"}

    present = long_df["XGroup"].dropna().unique().tolist()
    non_wt  = [g for g in present if str(g).upper() != "WT"]
    multi   = len(non_wt) > 1

    if not multi:
        relabel = {g: genename for g in non_wt}
    else:
        relabel = {g: zyg_display.get(str(g).upper(), str(g).title()) for g in non_wt}

    long_df["XGroup"] = long_df["XGroup"].replace(relabel)

    # --- Order: WT first, then mutants by zygotic_order (unknowns sort last) ---
    def _zygo_rank(orig_label):
        up = str(orig_label).upper()
        return zygotic_order.index(up) if up in zygotic_order else len(zygotic_order)

    non_wt_sorted = sorted(non_wt, key=_zygo_rank)
    group_order = ["WT"] + [relabel[g] for g in non_wt_sorted]

    # --- Resolve colors keyed to the FINAL display labels ---
    # x_colors keys are inconsistent upstream (raw "HOM"/"HEMI" but bare gene for
    # het), so for each final label we try several candidate source keys,
    # case-insensitively, before falling back.
    def _resolve_widget(entry):
        val = getattr(entry, "value", entry)          # widget -> .value, else itself
        val = val.strip() if isinstance(val, str) else ""
        return val or None

    xc_norm = {}                                      # UPPER(source key) -> color
    for k, v in x_colors.items():
        c = _resolve_widget(v)
        if c:
            xc_norm[str(k).upper()] = c

    final_to_orig = {new: old for old, new in relabel.items()}
    final_to_orig["WT"] = "WT"

    def _color_for(final_label):
        orig = final_to_orig.get(final_label, final_label)
        for cand in (orig, genename, final_label):
            hit = xc_norm.get(str(cand).upper())
            if hit:
                return hit
        return "tab:blue"

    color_map = {g: _color_for(g) for g in group_order}
    # One controlled Sex order shared by every bar panel so dot colors line up.
    hue_order = core._order_hue_groups(long_df["HueGroup"].dropna().unique().tolist())

    # ---------------- Figure + grid layout ----------------
    # Row 0: A (schematic) | B (example histogram) | C (demand curve, widest).
    # Row 1: five equal-width bar panels D-H.
    core.set_plot_style()   # one shared font family across every L4 figure
    #fig = plt.figure(figsize=(16, 8))
    fig_w, fig_h = figsize
    fig = plt.figure(figsize=(fig_w, fig_h))

    # --- Gridspec: two rows, top row has 3 panels, bottom row has 5 panels ---
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[1.0, 1.0], hspace=h_space)

    gs_top = gs[0].subgridspec(1, 3, width_ratios=[1.2, 1.3, 1.8], wspace=0.25)
    gs_bot = gs[1].subgridspec(1, 5, wspace=0.75)



    ##### Panel A: schematic #####
    if schematic_path is None:
        schematic_path = fedassets.get("pr1_schematic.jpg")

    ax_schem = fig.add_subplot(gs_top[0, 0])
    ax_schem.axis("off")
    core._panel_label(ax_schem, "A)", dx=0.00, dy=0.75)
    if schematic_path is not None and Path(schematic_path).exists():
        ax_schem.imshow(mpimg.imread(str(schematic_path)))
    else:
        status.warn(f"L4: schematic image not found ({schematic_path}); panel A blank.")
    ax_schem.set_title(genename, loc="left", fontsize=30, fontweight="bold", y=1.2)
    
    

    ##### Panel B: example mouse pellet histogram #####
    # Force the example to be a het mouse: recover HET's final display label (the
    # bare gene name when it's the only mutant, else "Het") to match the relabeled
    # long_df["XGroup"]. None if there's no het group, in which case any mouse is ok.
    het_raw = next((g for g in non_wt if str(g).upper() == "HET"), None)
    het_display = relabel.get(het_raw) if het_raw is not None else None

    ax_ex = fig.add_subplot(gs_top[0, 1])
    core._panel_label(ax_ex, "B)")
    fed_ex, mid_ex = _pick_example_fed(
        long_df, fed_list, metadata_df, target_group=het_display, min_blocks=10,
    )
    if fed_ex is None:
        status.warn("L4: no usable FED session for the example trace; panel B blank.")
        ax_ex.axis("off")
    else:
        status.ok(f"L4 example mouse -> {mid_ex}")
        _plot_pr_trace_core(fed_ex, ax_ex, show_xlabel=True)



    ##### Panel C: grouped demand curve #####
    ax_dem = fig.add_subplot(gs_top[0, 2])
    core._panel_label(ax_dem, "C)")
    _demand_curve_core(long_df, ax_dem, color_map, group_order)
    # In the multi-mutant case, prefix the gene name onto non-WT legend entries.
    if multi:
        leg = ax_dem.get_legend()
        if leg is not None:
            for txt in leg.get_texts():
                if txt.get_text().upper() != "WT":
                    txt.set_text(f"{genename} {txt.get_text()}")



    ##### Panels D-H: metric bars #####
    # (long_df variable name, y-axis label shown on the panel)
    metric_specs = [
        ("Daily_Pellets",   "Daily Pellets"),
        ("Total_Pokes",     "Total Pokes"),
        ("MedianBreakPoint", "Median Break Point"),
        ("Demand_alpha_FR", "alpha"),
        ("Demand_beta_FR",  "Slope"),
    ]
    bar_panel_letters = ["D)", "E)", "F)", "G)", "H)"]

    shared_handles = []
    for col, (metric, ylabel) in enumerate(metric_specs):
        ax_bar = fig.add_subplot(gs_bot[0, col])
        core._panel_label(ax_bar, bar_panel_letters[col])
        sub = long_df[long_df["variable"] == metric]
        # Guard: a model missing a metric shouldn't crash the whole composite.
        if sub["value"].dropna().empty:
            status.warn(f"L4: no data for {metric}; leaving panel blank.")
            ax_bar.axis("off")
            continue
        handles = core._plot_metric_display(
            sub, metric, ax_bar, color_map,
            group_order=group_order, hue_order=hue_order, ylabel=ylabel,
            xlabel=(genename if multi else ""),
        )
        if handles and not shared_handles:
            shared_handles = handles

    # --- One shared Sex legend for the whole figure ---
    if shared_handles:
        fig.legend(
            handles=shared_handles, title="Sex",
            loc="lower right", frameon=False,
            bbox_to_anchor=(0.125, 0.55),
        )



    # --- Caption block beneath the panels ---
    caption_raw = (
        "A) FED3 device and PR task schematic. B) Individual mouse histogram: each earned pellet "
        "lotted at how many pokes (block depth) it took to earn it. C) Grouped demand curve showing "
        "mean alpha (the price at which consumption halves) and slope of the curve. D, E, F, G, H) "
        "Bar graphs of mean daily pellets, total pokes, median break point, alpha and slope respectively. "
        "Statistics: two-way ANOVA; the reported p-value is the genotype effect (genotype x sex and sex "
        "effects are in the stats table)."
    )

    caption = core._wrap_caption(caption_raw, fig_w, fontsize=font_size)
    fig.text(0.1, 0.02, caption, ha="left", va="bottom", fontsize=font_size)

    # Reserve room at the bottom for the caption (tight_layout can't see fig.text).
    n_lines = caption.count("\n") + 1

    # line height in fig fraction: fontsize pts * ~ 1.6 leading / figure height in pts
    cap_frac = n_lines * font_size * 1.7 / (fig_h * 72)
    print(f"L4: caption {n_lines} lines, reserving {cap_frac:.3f} fig fraction at bottom.")

    # figure out padding crudely based on the longest group label, so the x-axis labels don't overlap the caption.
    longest = max((len(str(g)) for g in group_order), default=0)
    # ~0.011 fig-fraction per char at fontsize 14 on a 9in figure; tune the constant
    label_depth = longest * 0.011 * (9 / fig_h)

    fig.subplots_adjust(bottom=cap_frac + label_depth)



    ### --- Save the composite figure --- #
    out_path = out_dir / f"{genename}_PR_L4.svg"
    # bbox_inches="tight" keeps the caption and shared legend from being clipped.
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", format="svg")
    plt.close(fig)
    
    status.ok(f"PR1 L4 composite saved -> {out_path}")
    return out_path


#@@@@@@@@@@@@@@@@@@ DEFINE PIPELINE AND CLASS @@@@@@@@@@@@@@@@@@#

@dataclass
class PRResult:
    """Everything the L1->L4 run produced, so callers can inspect or re-plot
    any stage without rerunning the pipeline. Beats returning a bare tuple of
    a dozen values — attributes are self-documenting and order-independent."""
    fed_list: list
    key_df2: object
    saved_paths_indv: list
    pm_md: object          # essentailly the L3, bandit metrics and metadata
    pm_long: object        # melted version og bm_md
    l3_path: Path
    barplot_paths: list
    stats_df: object
    l4_path: Path




def run_pr_l1_l4(l1_path, key_path, root_path, *, colors=None, dpi=300):
    """Run the full PR pipeline from an L1 zip to the L4 composite figure.

    Orchestration only — every step delegates to the existing public
    functions, so this stays a readable table of contents for the pipeline.

    Args:
        l1_path, key_path, root_path : the three inputs your notebook sets by hand.
        colors; optional {group: color} override. If None, falls back to
            define_aesthetics' defaults so the function runs headless (no widget
            interaction required).
        dpi; int 
            save resolution passed through to assemble_l4.

    """
    root_path = Path(root_path)

    # ------ Ingest data ------ #
    # call to lkoad lists and ingest the data from the l1 folder
    fed_list, loaded_files, session_types = core.ingest_l1(l1_path)

    # ------ Create Meta Data Key ------ #
    # upload key file
    key_df, msg = core._read_key_from_upload(key_path)

    # build key dataframe
    key_df2 = core.build_or_rematch_key_df(loaded_files, session_types, key_df, msg_hint = f"Key status: {msg}")

    # Create Individual PR1 plots
    saved_paths_indv = pr1_indv_plots(fed_list, key_df2, root_path)

    # --- Individual plots + metrics -> L3 ---
    # Clean PR1 metadata
    meta_cols, md = core.build_metakey(key_df2, assay = "bandit")

    # Find ID columns
    id_col, other_id = core.pick_match_method(md)

    # compute PR metrics
    pm = compute_pr_metrics(fed_list, md)

    pm_md = core.attach_meta(pm, md, id_col)

    # Build the l3
    l3_path = core.output_l3(pm_md, id_col, other_id, meta_cols, root_path, assay = "pr1", )

    # Build groupings
    mapped_df = core.build_group_selections(md)

    # attach the grouping dataframe to the metrics
    pm_grps_df = core.merge_group_selections(pm_md, mapped_df)

    # --- Plot PR --- #
    # Melt the metric dataframe to long format
    pm_long = core.melt_metric(pm_grps_df, assay = "pr1")

    # Create variables to control aesthetics
    x_checks, x_colors, ordered_x = core.define_aesthetics(pm_long)

    # Actually create the bar plots 
    barplot_paths = core._run_plots(pm_long, x_checks, x_colors, ordered_x, root_path)

    # Honor a caller-supplied palette; otherwise use the aesthetics defaults.
    x_checks, x_colors, ordered_x = core.define_aesthetics(pm_long)
    if colors is not None:
        x_colors = colors
    
    # --- Plot demand curve --- #
    fig, stats = plot_group_mean_demand_with_params(mapped_df, pm_grps_df, md, x_colors)

    # --- build stats table --- #
    stats_df = core.build_stats_table(pm_long, ordered_x, root_path, assay="pr1")

    # --- Build L4 --- #
    l4_path = assemble_pr_l4(pm_long, x_colors, ordered_x, pm_md, root_path, fed_list=fed_list, metadata_df=key_df2)

    ### Return PR class ###
    return PRResult(fed_list, key_df2, saved_paths_indv, pm_md, pm_long,
                        l3_path, barplot_paths, stats_df, l4_path)
