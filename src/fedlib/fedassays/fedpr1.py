"""
Functions specificlly for PR1 assay and plotting from a FED device
William B. Rubio
"""

# import dependancies
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import tqdm
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import sem
from itertools import cycle

# Import cousins
from fedlib.fedutils.fedlog import status 
from fedlib.extracted import fed3bandit_extracted, fed3_loading, fed3_fedframe
from fedlib import fedassets
from fedlib.fedcore import core





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
def compute_pr_metrics(fed_list, md, min_runs_per_mice = 5):
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





def assemble_pr_l4(long_df, x_colors, ordered_x, bm_md, root_path,
                fed_list, metadata_df, *, bandittype = None, schematic_path=None, dpi=300):
    """
    Assemble the composite "L4" deliverable figure for one knockout model:

        A) device schematic (optional) + two example P(Left) traces
           (a WT and a gene mouse chosen for similar reward / different accuracy)
        B) reverse-learning line plot  +  Peak Accuracy bar
        C) Total pokes bar
        D) Win-stay bar
        E) Lose-shift bar

    plus a single shared Sex (Female/Male) legend and a caption. Every bar panel
    reports the genotype main effect from a two-way ANOVA, matching the caption.

    Orchestration only: the drawing lives in _plot_pleft_core, _rev_learning_core
    and _plot_metric_display so the layout logic here stays readable.

    Arguments:
        long_df : DataFrame
            Melted metrics (columns: variable / XGroup / HueGroup / value).
        rev_df : DataFrame
            Per-trial peak accuracy around the switch (Timepoint / Value /
            Display_Group).
        x_colors : dict
            Group -> ipywidgets color widget (plain strings also tolerated).
        ordered_x : list
            X-group order hint from the aesthetics step (WT is forced first).
        bm_md : DataFrame
            Per-session metadata; supplies the gene name AND feeds
            find_contrast_pair to choose the two example mice.
        root_path : str | Path
            Output root; the figure is written under <root_path>/L4.
        fed_list : list
            FED session DataFrames, used to draw the two example traces directly.
        metadata_df : DataFrame
            Stitched metadata, used to map Mouse_ID -> FED session for the traces.
        bandittype; String | None
            accepts "bandit100" & "bandit80" in order to properly create the schematics.
        schematic_path : str | Path | None
            Optional device/behaviour image for the top-left of panel A. If None,
            that corner is left blank for manual assembly.
        dpi : int
            Save resolution.

    Returns:
        out_path : Path
            Path to the saved composite PNG.
    """


    status.step("Assembling L4 composite figure")

    out_dir = Path(root_path, "L4")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Gene name that replaces the generic "HET" label throughout the figure.
    genename = bm_md["Gene"][0]

    # --- Pick the two example mice: similar reward, different performance ---
    best, ranked = find_contrast_pair(bm_md, pellet_tol=15)
    wt_id = str(best["Mouse_ID_WT"])
    het_id = str(best["Mouse_ID_HET"])
    status.ok(f"L4 example pair -> WT: {wt_id} | {genename}: {het_id} "
              f"(acc gap {best['acc_gap']:.3f}, pellet gap {best['pellet_gap']:.1f})")


    # --- Resolve colors to a plain {group: color} dict ONCE ---
    # x_colors may hold ipywidgets (with .value) or plain strings; tolerate both.
    def _resolve(g):
        c = x_colors.get(g, None)
        val = getattr(c, "value", c)                 
        val = val.strip() if isinstance(val, str) else ""
        return val or "tab:blue"
    color_map = {g: _resolve(g) for g in x_colors}


    long_df = long_df.copy()
    rev_df = rev_df.copy()

    # Preferred left-to-right order of zygosities after WT, and their pretty form.
    zygotic_order = ["HET", "HOM", "HEMI"]
    zyg_display   = {"HET": "Het", "HOM": "Hom", "HEMI": "Hemi"}

    present = long_df["XGroup"].dropna().unique().tolist()
    non_wt  = [g for g in present if str(g).upper() != "WT"]
    multi   = len(non_wt) > 1                      # drives the labeling convention

    if not multi:
        # Single mutant -> bare gene name shown on the bar itself.
        relabel = {g: genename for g in non_wt}
    else:
        # Multiple mutants -> zygosity on the bar; gene name goes on the x-axis.
        relabel = {g: zyg_display.get(str(g).upper(), str(g).title()) for g in non_wt}

    long_df["XGroup"] = long_df["XGroup"].replace(relabel)
    rev_df["Display_Group"] = rev_df["Display_Group"].replace(relabel)

    # --- Order: WT first, then mutants by zygotic_order (unknowns sort last) ---
    def _zygo_rank(orig_label):
        up = str(orig_label).upper()
        return zygotic_order.index(up) if up in zygotic_order else len(zygotic_order)

    non_wt_sorted = sorted(non_wt, key=_zygo_rank)
    group_order = ["WT"] + [relabel[g] for g in non_wt_sorted]

    # --- Resolve colors keyed to the FINAL display labels ---
    # x_colors keys are inconsistent upstream (raw "HOM"/"HEMI" but bare "FMR1"
    # for het), so for each final label we try several candidate source keys,
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

    # Recover each final label's ORIGINAL raw zygosity so we can try it as a key.
    final_to_orig = {new: old for old, new in relabel.items()}
    final_to_orig["WT"] = "WT"

    def _color_for(final_label):
        orig = final_to_orig.get(final_label, final_label)
        # try: raw zygosity ("HOM"), the bare gene ("FMR1", covers het), then the
        # display label itself, then a visible default.
        for cand in (orig, genename, final_label):
            hit = xc_norm.get(str(cand).upper())
            if hit:
                return hit
        return "tab:blue"

    color_map = {g: _color_for(g) for g in group_order}
    # One controlled Sex order shared by every bar panel so dot colors line up.
    hue_order = core._order_hue_groups(long_df["HueGroup"].dropna().unique().tolist())



    # ---------------- Figure + grid layout ----------------
    # Row 0: panel A. 
    # Row 1: line plot (wide) + four equal-width bar plots.
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(
        nrows=2, ncols=5,
        height_ratios=[1.0, 1.4],
        width_ratios=[3, 1, 1, 1, 1],
        hspace=0.35, wspace=0.45,
    )


    ##### Panel A  #####
    # schematic (left) + two stacked example traces (right)
    # Nested grid so the top row can hold both the image and the two traces.
    gs_a = gs[0, :].subgridspec(
        2, 3, 
        width_ratios=[1, 2, 0.3], 
        hspace=0.25, wspace=0.08)

    # Schematic spans both sub-rows on the left; blank if no image supplied.
    # find what bandit version
    bt = str(bandittype).strip().lower()
    bandit80_dict = {"bandit80", "80"}
    bandit100_dict = {"bandit100", "100"}
    
    if bt in bandit80_dict:
        schematic_path=fedassets.get("bandit80_schematic.jpg")
    elif bt in bandit100_dict:
        schematic_path=fedassets.get("bandit100_schematic.jpg")
    else:
        # fail loudly
        raise ValueError(
            f"Unrecognized bandittype {bandittype!r}; "
            f"expected one of {sorted(bandit80_dict | bandit100_dict)}."
        )

    # add the bandit schematic
    ax_schem = fig.add_subplot(gs_a[:, 0])
    ax_schem.axis("off")
    _panel_label(ax_schem, "A)", dx=1.2, dy=1.0)

    if schematic_path is not None and Path(schematic_path).exists():
        ax_schem.imshow(mpimg.imread(str(schematic_path)))
    # add genename label at title
    ax_schem.set_title(genename, loc="left", fontsize=20, fontweight="bold")


    ### Switching plots ###
    # Two example traces, drawn directly onto their axes (no PNG round-trip).
    # (Mouse_ID, right-margin label, genotype color, show x-label on bottom only)
    het_raw = next((g for g in non_wt if str(g).upper() == "HET"), None)
    het_display = relabel.get(het_raw, genename)
    het_color = color_map.get(het_display, "tab:blue")

    # Label reads "<gene> behaviour" (bare gene), matching the figure.
    trace_specs = [
        (wt_id,  "Wildtype behaviour",    color_map["WT"], False),
        (het_id, f"{genename} behaviour", het_color,       True),
    ]




    for row, (mid, label, col, show_x) in enumerate(trace_specs):
        ax_tr = fig.add_subplot(gs_a[row, 1])
        fed = _fed_for_mouse(fed_list, metadata_df, mid)
        
        if fed is None:
            # Missing session shouldn't kill the whole composite.
            status.warn(f"L4: no FED session found for Mouse_ID {mid}; blank trace.")
            ax_tr.axis("off")
            continue
        _plot_pleft_core(fed, ax_tr, line_color=col,
                         behaviour_label=label, show_xlabel=show_x)
        
        # turn off all axis
        ax_tr.axis("off")
        
        if row == 1:
            # add reward side text
            ax_tr.text(1.01, 1.05, "Rewarded Side", transform=ax_tr.transAxes,
                       ha="left", va="bottom", color="0.5", fontsize = 12)

            # Add arrow for both plots
            ax_tr.annotate("",
                    xy=(1.0, -0.2), xytext=(0.15, -0.2),
                    xycoords="axes fraction",
                    annotation_clip=False,  # don't clip content drawn above the axes
                    arrowprops=dict(arrowstyle="->", lw=5, color="0.6"))

            # Label for arrow independently
            ax_tr.text(0.07, -0.28, "3 days", transform=ax_tr.transAxes,
                       ha="left", va="bottom", color="0.5")



    ##### Panel B-left: reverse-learning line plot #####
    ax_line = fig.add_subplot(gs[1, 0])
    _panel_label(ax_line, "B)")          
    _rev_learning_core(rev_df, ax_line, 
                       palette_map = color_map, 
                       group_order = group_order)

    if multi:
        leg = ax_line.get_legend()
        if leg is not None:
            for txt in leg.get_texts():
                if txt.get_text().upper() != "WT":
                    txt.set_text(f"{genename} {txt.get_text()}")


    ##### Panels B-bar / C / D / E: metrics#####
    # (long_df variable name, y-axis label shown on the panel)
    metric_specs = [
        ("PeakAccuracy", "Peak Accuracy %"),
        ("Total_pokes",  "Total pokes"),
        ("Win-stay",     "Win-stay"),
        ("Lose-shift",   "Lose-shift"),
    ]

    # specify the panel letters for each bar plot metric
    bar_panel_letters = {2: "C)", 3: "D)", 4: "E)"}

    shared_handles = []
    for col, (metric, ylabel) in enumerate(metric_specs, start=1):
        ax_bar = fig.add_subplot(gs[1, col])
        # assign bar plot panels
        if col in bar_panel_letters:
            _panel_label(ax_bar, bar_panel_letters[col])
        sub = long_df[long_df["variable"] == metric]
        # Guard: a model missing a metric shouldn't crash the whole composite.
        if sub["value"].dropna().empty:
            status.warn(f"L4: no data for {metric}; leaving panel blank.")
            ax_bar.axis("off")
            continue
        handles = _plot_metric_display(
            sub, metric, ax_bar, color_map,
            group_order=group_order, hue_order=hue_order, ylabel=ylabel,
            xlabel=(genename if multi else ""),
        )
        # Keep the first non-empty proxy set for the single shared legend.
        if handles and not shared_handles:
            shared_handles = handles

    # --- One shared Sex legend for the whole figure (top-right) ---
    if shared_handles:
        fig.legend(
            handles=shared_handles, title="Sex",
            loc="lower right", frameon=False,
            bbox_to_anchor=(0.95, 0.35),
        )

    

    # --- Caption block beneath the panels ---
    # Assembled line-by-line (each string ends with a space) so the joins never
    # run words together, with explicit \n where a visible line break is wanted.
    caption = (
        "A) FED3 device and example behaviour schematic. "
        "B) Line plot and bar graph of peak accuracy in the 10 trials around a switch.\n"
        "C, D, E) Bar graphs of mean total pokes, win-stay and lose-shift respectively.\n"
        "Win-stay: after a reward, did the mouse choose the same port again. "
        "Lose-shift: after no reward, did it choose the opposite port.\n"
        "Statistics: two-way ANOVA; the reported p-value is the genotype effect "
        "(genotype x sex and sex effects are in the stats table)."
    )
    fig.text(0.1, 0.02, caption, ha="left", va="bottom", fontsize=14)


    

    # Reserve room at the bottom for the caption (tight_layout can't see fig.text).
    fig.subplots_adjust(bottom=0.22)

    out_path = out_dir / f"{genename}_L4.svg"
    # bbox_inches="tight" keeps the caption and shared legend from being clipped.
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", format = "svg")
    plt.close(fig)

    status.ok(f"L4 composite saved -> {out_path}")
    return out_path