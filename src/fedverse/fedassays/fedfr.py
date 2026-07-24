"""
Functions specificlly for Fixed ratio FR1 assay and plotting from a FED device
William B. Rubio
"""
# import dependancies
from dataclasses import dataclass
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
from IPython.display import display, HTML

# call cousins
from fedverse.fedutils.fedlog import status
from fedverse.fedcore import core
from fedverse import fedassets



#@@@@@@@@@@@@@@@@@@ FR METRICS @@@@@@@@@@@@@@@@@@#

### ------ Helpers functions ------- ###



def _pellet_times_from_df(df):
    """
    called by: 
    """
    ev_col = core._safe_col(df, ["Event", "event"])
    if ev_col is None:
        return []
    pel = df[df[ev_col] == "Pellet"]
    if isinstance(pel.index, pd.DatetimeIndex):
        ts = pel.index.to_series()
    else:
        for cand in ["MM:DD:YYYY hh:mm:ss", "DateTime", "Datetime", "Timestamp", "timestamp", "datetime"]:
            if cand in pel.columns:
                ts = pd.to_datetime(pel[cand], errors="coerce")
                break
        else:
            ts = pd.to_datetime(pel.index, errors="coerce")
    ts = ts.dropna().sort_values()
    return ts.to_list()


def _get_ts_series_from_col_or_index(df, ts_candidates=("MM:DD:YYYY hh:mm:ss","DateTime","Datetime","Timestamp","timestamp","datetime")):
    for cand in ts_candidates:
        if cand in df.columns:
            ts = pd.to_datetime(df[cand], errors="coerce")
            if not isinstance(ts, pd.Series):
                ts = pd.Series(ts, index=df.index)
            else:
                ts = ts.reindex(df.index)
            return ts
    if isinstance(df.index, pd.DatetimeIndex):
        return pd.Series(df.index, index=df.index)
    return pd.to_datetime(pd.Series(df.index, index=df.index), errors="coerce")


def _split_day_night_masks(ts):
    valid = ts.notna()
    hrs = ts.dt.hour
    day_mask = valid & (hrs >= 6) & (hrs < 18)
    night_mask = valid & ~day_mask
    return day_mask, night_mask

def _cluster_times(ts_list, max_interval_sec=60):
    if not ts_list:
        return []
    clusters, current = [], [ts_list[0]]
    for i in range(1, len(ts_list)):
        if (ts_list[i] - ts_list[i-1]).total_seconds() <= max_interval_sec:
            current.append(ts_list[i])
        else:
            clusters.append(current); current = [ts_list[i]]
    clusters.append(current)
    return clusters

def _within_meal_ipi_mean_from_pellet_times(ts_list, max_interval_sec=60, min_meal_size=3, weighted=False):
    """
    Mean inter-pellet interval (seconds) computed ONLY within meals:
      - meal = cluster of pellets where successive pellets are <= max_interval_sec apart
      - meal must have at least min_meal_size pellets (default 3)
    If weighted=False: average the per-meal means (each meal counts equally).
    If weighted=True: average all within-meal IPIs pooled (meals weighted by (meal_size-1)).
    """
    if not ts_list or len(ts_list) < 2:
        return np.nan

    clusters = _cluster_times(ts_list, max_interval_sec=max_interval_sec)
    meal_clusters = [c for c in clusters if len(c) >= min_meal_size]
    if not meal_clusters:
        return np.nan

    meal_means = []
    pooled_ipis = []

    for c in meal_clusters:
        ipis = np.diff([t.timestamp() for t in c])  # seconds
        ipis = ipis[(ipis > 0) & (ipis <= max_interval_sec)]  # safety clamp to within-meal definition
        if ipis.size == 0:
            continue
        meal_means.append(float(np.mean(ipis)))
        pooled_ipis.extend(ipis.tolist())

    if weighted:
        return float(np.mean(pooled_ipis)) if len(pooled_ipis) else np.nan
    else:
        return float(np.mean(meal_means)) if len(meal_means) else np.nan
    
def _subset_meal_metrics(df, ts_series, max_interval_sec=60):
    out = {
        "%MealPellets": np.nan,
        "%GrazingPellets": np.nan,
        "Pellets": 0,
        "NumMeals": np.nan,
        "AvgMealSize": np.nan,
        "AvgMealDuration": np.nan,
        "MealsPerHour": np.nan,
        "Accuracy": np.nan,
    }
    ev_col = core._safe_col(df, ["Event", "event"])
    if ev_col is None or df.empty:
        return out

    left_n = int((df[ev_col] == "Left").sum())
    right_n = int((df[ev_col] == "Right").sum())
    denom = left_n + right_n
    if denom > 0:
        out["Accuracy"] = 100.0 * (left_n / denom)

    pel_mask = (df[ev_col] == "Pellet")
    pel_ts = ts_series[pel_mask].dropna().sort_values()
    out["Pellets"] = int(pel_ts.size)
    if pel_ts.size == 0:
        return out

    ts_list = pel_ts.to_list()
    clusters = _cluster_times(ts_list, max_interval_sec=max_interval_sec)
    meal_clusters = [c for c in clusters if len(c) >= 3]
    grazing_clusters = [c for c in clusters if 1 <= len(c) < 3]

    meal_pellets = sum(len(c) for c in meal_clusters)
    grazing_pellets = sum(len(c) for c in grazing_clusters)
    total_pg = meal_pellets + grazing_pellets
    if total_pg > 0:
        out["%MealPellets"] = 100.0 * meal_pellets / total_pg
        out["%GrazingPellets"] = 100.0 * grazing_pellets / total_pg

    if len(meal_clusters) > 0:
        out["NumMeals"] = float(len(meal_clusters))
        out["AvgMealSize"] = float(np.mean([len(c) for c in meal_clusters]))
        out["AvgMealDuration"] = float(np.mean([(c[-1] - c[0]).total_seconds() for c in meal_clusters]))
    else:
        out["NumMeals"] = 0.0

    if len(ts_list) >= 2:
        rec_hours = (ts_list[-1] - ts_list[0]).total_seconds() / 3600.0
        if rec_hours > 0:
            out["MealsPerHour"] = (out["NumMeals"] / rec_hours) if np.isfinite(out["NumMeals"]) else np.nan
    return out


# specific metric calculations
def compute_within_meal_mode_from_sessions(sessions, max_interval_sec=60, min_samples=5):
    rows = []
    try:
        from scipy.stats import gaussian_kde
        use_kde = True
    except Exception:
        use_kde = False

    for i, df in enumerate(sessions):
        file = core._basename(getattr(df, "name", f"File_{i}"))
        ipi_col = core._safe_col(df, ["InterPelletInterval", "interpelletinterval", "inter_pellet_interval"])
        if ipi_col is None or df.empty:
            rows.append({"File": file, "Within_meal_pellet_rate": np.nan}); continue

        vals = pd.to_numeric(df[ipi_col], errors="coerce").dropna()
        vals = vals[(vals > 0) & (vals <= max_interval_sec)]
        if vals.size < min_samples:
            rows.append({"File": file, "Within_meal_pellet_rate": np.nan}); continue

        logv = np.log10(vals.values)

        if use_kde:
            kde = gaussian_kde(logv, bw_method="scott")
            xs  = np.linspace(logv.min(), logv.max(), 1000)
            ys  = kde(xs)
            peak_log10 = xs[np.argmax(ys)]
        else:
            xs = np.linspace(logv.min(), logv.max(), 256)
            hist, edges = np.histogram(logv, bins=xs)
            centers = 0.5 * (edges[:-1] + edges[1:])
            peak_log10 = centers[np.argmax(hist)]

        peak_seconds = float(10.0 ** peak_log10)
        rows.append({"File": file, "Within_meal_pellet_rate": peak_seconds})

    return pd.DataFrame(rows)




def compute_meal_bout_metrics_from_sessions(sessions, max_interval_sec=60):
    rows, recinfo = [], []
    for i, df in enumerate(sessions):
        file = core._basename(getattr(df, "name", f"File_{i}"))
        ts = _pellet_times_from_df(df)
        if len(ts) < 2:
            continue
        duration_hr = (ts[-1] - ts[0]).total_seconds() / 3600.0
        recinfo.append({"File": file, "RecordingHours": duration_hr})

        clusters = _cluster_times(ts, max_interval_sec=max_interval_sec)
        meal_id = 0
        for c in clusters:
            if len(c) >= 3:
                rows.append({
                    "File": file,
                    "MealID": meal_id,
                    "MealSize": len(c),
                    "MealDuration_sec": (c[-1] - c[0]).total_seconds(),
                })
                meal_id += 1
    meal_df = pd.DataFrame(rows)
    rec_df  = pd.DataFrame(recinfo)
    if meal_df.empty:
        out = pd.DataFrame(columns=["File","NumMeals","AvgMealSize","AvgMealDuration","RecordingHours","MealsPerHour"])
    else:
        out = (meal_df.groupby("File")
               .agg(NumMeals=("MealID","count"), AvgMealSize=("MealSize","mean"), AvgMealDuration=("MealDuration_sec","mean"))
               .reset_index())
    out = out.merge(rec_df, on="File", how="left")
    out["MealsPerHour"] = out["NumMeals"] / out["RecordingHours"]
    return out

def compute_meal_pellet_distribution_from_sessions(sessions, max_interval_sec=60):
    rows = []
    for i, df in enumerate(sessions):
        file = core._basename(getattr(df, "name", f"File_{i}"))
        ts = _pellet_times_from_df(df)
        if len(ts) == 0:
            rows.append({"File": file, "%MealPellets": np.nan, "%GrazingPellets": np.nan})
            continue
        clusters = _cluster_times(ts, max_interval_sec=max_interval_sec)
        meal_pellets    = sum(len(c) for c in clusters if len(c) >= 3)
        grazing_pellets = sum(len(c) for c in clusters if 1 <= len(c) < 3)
        total = meal_pellets + grazing_pellets
        rows.append({
            "File": file,
            "%MealPellets": (100 * meal_pellets / total) if total else np.nan,
            "%GrazingPellets": (100 * grazing_pellets / total) if total else np.nan
        })
    return pd.DataFrame(rows)



### --- compute the FR metrics for a list of fed dataframes --- ###
def compute_fr_metrics(fed_list):
    """
    Does the computation of FR metrics for a list of fed dataframes. Returns a dataframe with the metrics.
    1) Computes basic FR metrics (pellets, pokes, accuracy, retrieval time, inter-pellet interval, poke time, daily pellets)
    2) Computes meal/grazing metrics (percent meal pellets, percent grazing pellets, number of meals, average meal size, average meal duration, meals per hour)

    Arguments:
    fed_list: list 
        list of fed dataframes
    Returns:
        FR1_enriched
            pd.DataFrame

    """

    status.step("Computing FR metrics")

    rows = []
    
    for idx, c_df in enumerate(fed_list):
        file_name = core._basename(getattr(c_df, "name", f"File_{idx}"))
        d, ev = core._prep_events(c_df)
        if ev is None or d.empty:
            pellets = left = right = lwp = 0
            rt_med = ipi_med = pt_med = np.nan
        else:
            pellets = int((d[ev] == "Pellet").sum())
            left    = int((d[ev] == "Left").sum())
            right   = int((d[ev] == "Right").sum())
            lwp     = int((c_df[ev] == "LeftWithPellet").sum()) if ev in c_df.columns else 0

            rt_col  = core._safe_col(d, ["Retrieval_Time", "retrieval_time"])
            pt_col  = core._safe_col(d, ["Poke_Time", "poke_time"])

            rt_med  = pd.to_numeric(d.get(rt_col, pd.Series(dtype=float)), errors="coerce").median() if rt_col else np.nan
            pt_med  = pd.to_numeric(d.get(pt_col, pd.Series(dtype=float)), errors="coerce").median() if pt_col else np.nan

            # Mean IPI within meals only (exclude grazing clusters of <3 pellets)
            pellet_ts_list = _pellet_times_from_df(c_df)
            ipi_med = _within_meal_ipi_mean_from_pellet_times(
                pellet_ts_list,
                max_interval_sec=60,
                min_meal_size=3,
                weighted=False,   # set True if you want larger meals to contribute more
            )

        total_pokes = left + right
        acc = (left / total_pokes * 100.0) if total_pokes > 0 else np.nan
        ppp = (total_pokes / pellets) if pellets > 0 else np.nan
        daily_pel = core._estimate_daily_pellets(c_df)

        rows.append({
            "File": file_name,
            "FileIndex": idx,
            "Pellets": pellets,
            "Left_Poke": left,
            "Right_Poke": right,
            "Total_Pokes": total_pokes,
            "Accuracy": acc,
            "PokesPerPellet": ppp,
            "RetrievalTime": rt_med,
            "InterPelletInterval": ipi_med,
            "PokeTime": pt_med,
            "Daily_Pellets": daily_pel,
            "Left Poke with Pellet": lwp,
        })

    FR1metrics = pd.DataFrame(rows)
    if FR1metrics.empty:
        display(HTML("<b style='color:#b00'>No files to analyze.</b>"))
        raise SystemExit

    # 4) Meal/grazing metrics
    status.sub("Computing meal/grazing metrics")
    meal_info_df   = compute_meal_bout_metrics_from_sessions(fed_list, max_interval_sec=60)
    status.ok("meal/grazing metrics computed")

    status.sub("Computing meal/grazing pellet distribution metrics")
    pellet_dist_df = compute_meal_pellet_distribution_from_sessions(fed_list, max_interval_sec=60)
    status.ok("meal/grazing pellet distribution metrics computed")

    FR1_enriched = (
        FR1metrics
        .merge(pellet_dist_df, on="File", how="left")
        .merge(meal_info_df, on="File", how="left")
    )

    wmpr_df = compute_within_meal_mode_from_sessions(fed_list, max_interval_sec=60, min_samples=5)
    FR1_enriched = FR1_enriched.merge(wmpr_df, on="File", how="left")
    
    # calculate day/night metrics
    status.sub("Computing day/night metrics")
    _day_night_bases = ["%MealPellets","%GrazingPellets","Pellets","NumMeals","AvgMealSize","AvgMealDuration","MealsPerHour","Accuracy"]
    for base in _day_night_bases:
        FR1_enriched[f"{base}_Day"] = np.nan
        FR1_enriched[f"{base}_Night"] = np.nan

    for i, df_all in enumerate(fed_list):
        file = core._basename(getattr(df_all, "name", f"File_{i}"))
        ts_all = _get_ts_series_from_col_or_index(df_all)
        day_mask, night_mask = _split_day_night_masks(ts_all)

        day_df = df_all.loc[day_mask]
        night_df = df_all.loc[night_mask]

        day_vals = _subset_meal_metrics(day_df, ts_all.loc[day_mask])
        night_vals = _subset_meal_metrics(night_df, ts_all.loc[night_mask])

        row_mask = FR1_enriched["File"] == file
        for base in _day_night_bases:
            FR1_enriched.loc[row_mask, f"{base}_Day"] = day_vals[base]
            FR1_enriched.loc[row_mask, f"{base}_Night"] = night_vals[base]
    status.ok("day/night metrics computed")

    status.ok("metrics computed:")
    status.preview(FR1_enriched, msg="FR metrics preview:")

    return FR1_enriched




#@@@@@@@@@@@@@@@@@@ FR INTERPELLET @@@@@@@@@@@@@@@@@@#
def plot_interpellet_interval(fed_list, mapped_df, df_md,  x_colors, root_path):
    """
    Plot the interpellet interval for a list of fed dataframes. Returns a dataframe with the interpellet interval values.
    Arguments:
    fed_list: list 
        list of fed dataframes
    mapped_df: pd.DataFrame
        dataframe with the mapping of files to groups
    df_md: pd.DataFrame
        dataframe with the FR metrics and metadata (used to get the gene name for the plot title)
    x_colors: dict
        dictionary with the colors for each group
    """

    status.step("Plotting interpellet interval")

    # define the directory to save the plot
    out_dir = Path(root_path, "interpellet_interval_plot")
    out_dir.mkdir(parents=True, exist_ok=True)

    # capture the saved path for the plot at the end of the function
    saved_path = []

    # Pull out the gene name (needed by the relabel lines below).
    genename = df_md["Gene"][0]

    #file_to_df = {Path(str(fname)).name: df for fname, df in zip(loaded_files, df)}
    file_names = [core._basename(getattr(df, "name", f"File_{i}")) for i, df in enumerate(fed_list)]
    file_to_df = dict(zip(file_names, fed_list))





    rows = []    
    for fname, df in file_to_df.items():
        if df is None or df.empty:
            continue
        if "Event" in df.columns:
            df = df[df["Event"].isin(["Left","Right","Pellet"])].copy()
        if "InterPelletInterval" not in df.columns:
            continue
        vals = pd.to_numeric(df["InterPelletInterval"], errors="coerce").dropna()
        vals = vals[vals > 0]
        if vals.empty:
            continue
        base = os.path.basename(str(fname))
        rows.extend({"filename": base, "IPI_s": float(v)} for v in vals)


    ipi = pd.DataFrame(rows)
    if ipi.empty:
        raise RuntimeError("No InterPelletInterval values found.")
    

    # 2) Merge XGroup from files_to_group_both (ignore Hue)
    #if "files_to_group_both" not in globals() or files_to_group_both is None or files_to_group_both.empty:
    #s    raise RuntimeError("Missing `files_to_group_both`. Build Groups first.")

    grp = mapped_df.copy()


    src_col = "filename" if "filename" in grp.columns else ("File" if "File" in grp.columns else None)
    if src_col is None:
        raise RuntimeError("`files_to_group_both` needs 'filename' or 'File'.")
    
    grp["filename"] = grp[src_col].astype(str).map(os.path.basename)
    ipi = ipi.merge(grp[["filename","XGroup"]].drop_duplicates(), on="filename", how="left")
    ipi["XGroup"] = ipi["XGroup"].fillna("UNASSIGNED")

    # 3) Choose groups: all checked in widget, else all present
    def _selected_xgroups():
        if 'x_checks' in globals() and isinstance(x_checks, dict) and len(x_checks):
            return [g for g, cb in x_checks.items() if getattr(cb, "value", False)]
        return sorted(ipi["XGroup"].dropna().unique().tolist())

    chosen = _selected_xgroups()
    if not chosen:
        raise RuntimeError("No XGroups selected/found.")
    ipi2 = ipi[ipi["XGroup"].isin(chosen)].copy()

    # 4) KDE in log10 space (correct for log-x plotting)
    ipi2 = ipi2[(ipi2["IPI_s"] > 0) & np.isfinite(ipi2["IPI_s"])]
    ipi2["log10_IPI"] = np.log10(ipi2["IPI_s"])

    # --- Match order & colors from the main plotting cell ---

    _present = ipi2["XGroup"].dropna().unique().tolist()

    # ORDER: prefer the global ordered_x (WT/CONTROL-first); otherwise reproduce rule locally
    if "ordered_x" in globals():
        group_order = [g for g in ordered_x if g in _present] + [g for g in _present if g not in ordered_x]
    else:
        import re
        def _is_wt(g):
            toks = [t for t in re.split(r'[^A-Z0-9]+', str(g).upper()) if t]
            return any(t in {"WT", "WILDTYPE", "CONTROL", "CTRL"} for t in toks)
        def _x_levels(g): return [p.strip() for p in str(g).split("|")]
        def _key(g):
            lv = _x_levels(g)
            wt_rank = 0 if any(_is_wt(tok) for tok in lv) or _is_wt(g) else 1
            blanks = [(1 if s.strip().upper() in {"","UNASSIGNED","NONE","NA","N/A"} else 0, s.upper()) for s in lv]
            return (wt_rank, *blanks, str(g).upper())
        group_order = sorted(_present, key=_key)

    # COLORS: pull from x_colors (widgets) so bars/lines/KDE match
    palette_map = None

    def _col(g):
        try:
            v = x_colors[g].value
            return v.strip() if isinstance(v, str) and v.strip() else "tab:blue"
        except Exception:
            return "tab:blue"
    palette_map = {g: _col(g) for g in group_order}


    # ---- PLOT ONLY (no metric computation) ----
    sns.set_style("white")
    plt.figure(figsize=(9, 5))
    sns.set_style("white")
    fig, ax = plt.subplots(figsize=(10, 5))

    # convert seconds to minutes and compute log10 in minutes
    ipi2["IPI_min"] = ipi2["IPI_s"] / 60.0
    ipi2 = ipi2[(ipi2["IPI_min"] > 0) & np.isfinite(ipi2["IPI_min"])]
    ipi2["log10_IPI_min"] = np.log10(ipi2["IPI_min"])

    # plot KDE in log10(minutes)
    ax = sns.kdeplot(
        data=ipi2,
        x="log10_IPI_min",
        hue="XGroup",
        hue_order=group_order,
        palette=palette_map,
        fill=False,
        common_norm=False,
        cut=0,
        bw_adjust=0.9,
        gridsize=512,
        linewidth=2
    )

    # Nice decade ticks based on data range (in log10 minutes)
    lo = np.floor(ipi2["log10_IPI_min"].min())
    hi = np.ceil(ipi2["log10_IPI_min"].max())
    ticks_log = np.arange(lo, hi + 1)
    ax.set_xticks(ticks_log)

    def _min_label(t):
        v = 10.0 ** float(t)  # minutes
        # format: show 2 decimals if <1 min, otherwise integer minutes
        return f"{v:.2f}" if v < 1 else f"{int(round(v))}"

    ax.set_xticklabels([_min_label(t) for t in ticks_log])
    ax.set_xlabel("Interpellet Interval (min)")
    ax.set_ylabel("Density")
    ax.set_title("")
    leg = ax.get_legend()
    if leg:
        leg.set_title("")
        leg.set_frame_on(False)
    sns.despine()
    plt.tight_layout()

    # Embed TrueType fonts so text stays editable in Illustrator/Inkscape
    import matplotlib as mpl
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42

    outdir = "figures"
    os.makedirs(outdir, exist_ok=True)
    from datetime import datetime
    fname = f"interpellet_histogram_{datetime.now():%Y-%m-%d}.pdf"
    fig = ax.get_figure()
    

    # define what to save the plot as
    safe_name = genename + "_interpellet_interval"
    out_path = out_dir / f"{safe_name}.png"

    # Save at print-friendly resolution; bbox_inches="tight" trims the
    # generous whitespace left by the 2-panel layout + rotated x-labels.
    fig.savefig(out_path, dpi=300, bbox_inches="tight")

    plt.close(fig)

    # append the saved path to the list of saved paths for later use
    saved_path.append(out_path)

    status.ok(f"Interpellet interval plot generated. Saving to {saved_path}")
    
    return saved_path



#@@@@@@@@@@@@@@@@@@ FR L4 @@@@@@@@@@@@@@@@@@#


### ------ L4 drawing cores (FR-specific) ------- ###

def _plot_ipi_density_core(fed_list, ax, mapped_df, palette_map, group_order, *,
                           meal_threshold_min=1.0):
    """
    Draw the grouped inter-pellet-interval (IPI) density onto a caller-supplied Axes:
    a per-group KDE over log10(IPI in minutes), with a dashed divider at the meal /
    grazing threshold (default 1 min) separating "Pellets within a meal" (short IPIs,
    left) from "Grazing pellets" (long IPIs, right). Creates no figure and saves
    nothing -- the caller owns the Axes. This is the composable core extracted from
    plot_interpellet_interval so the standalone plot and the FR L4 composite share one
    routine.

    mapped_df["XGroup"] must ALREADY be relabeled to the final display labels so it
    matches palette_map / group_order (keyed by those labels).

    Arguments:
        fed_list : list
            FED session DataFrames (each carries a .name filename attribute and an
            InterPelletInterval column).
        ax : matplotlib Axes
            Target axes to draw into.
        mapped_df : DataFrame
            filename -> XGroup mapping (already relabeled).
        palette_map : dict
            {group -> color} keyed by the final display labels.
        group_order : list
            Groups to plot, in order.
        meal_threshold_min : float
            IPI (minutes) dividing within-meal from grazing pellets. 1.0 -> 60 s.
    """
    # --- Per-file IPI values, converted to minutes ---
    file_names = [core._basename(getattr(df, "name", f"File_{i}")) for i, df in enumerate(fed_list)]
    rows = []
    for fname, df in zip(file_names, fed_list):
        if df is None or df.empty or "InterPelletInterval" not in df.columns:
            continue
        vals = pd.to_numeric(df["InterPelletInterval"], errors="coerce").dropna()
        vals = vals[vals > 0]
        rows.extend({"filename": fname, "IPI_min": float(v) / 60.0} for v in vals)

    ipi = pd.DataFrame(rows)
    if ipi.empty:
        status.warn("L4: no InterPelletInterval values; panel F left blank.")
        ax.axis("off")
        return

    # --- Attach XGroup from the (already relabeled) mapping ---
    grp = mapped_df.copy()
    src = "filename" if "filename" in grp.columns else ("File" if "File" in grp.columns else None)
    if src is None:
        status.warn("L4: group mapping lacks a filename/File column; panel F left blank.")
        ax.axis("off")
        return
    grp["filename"] = grp[src].astype(str).map(os.path.basename)
    ipi = ipi.merge(grp[["filename", "XGroup"]].drop_duplicates(), on="filename", how="left")
    ipi = ipi[ipi["XGroup"].isin(group_order)].copy()
    ipi = ipi[(ipi["IPI_min"] > 0) & np.isfinite(ipi["IPI_min"])]
    if ipi.empty:
        status.warn("L4: no IPI values map to the plotted groups; panel F left blank.")
        ax.axis("off")
        return

    ipi["log10_IPI_min"] = np.log10(ipi["IPI_min"])

    # --- KDE in log10(minutes) so a log x-axis reads correctly ---
    sns.kdeplot(
        data=ipi, x="log10_IPI_min",
        hue="XGroup", hue_order=group_order, palette=palette_map,
        fill=False, common_norm=False, cut=0, bw_adjust=0.9,
        gridsize=512, linewidth=2, ax=ax,
    )

    # --- Decade ticks labeled in minutes ---
    lo = np.floor(ipi["log10_IPI_min"].min())
    hi = np.ceil(ipi["log10_IPI_min"].max())
    ticks_log = np.arange(lo, hi + 1)
    ax.set_xticks(ticks_log)

    def _min_label(t):
        v = 10.0 ** float(t)                       # value in minutes
        return f"{v:.2f}" if v < 1 else f"{int(round(v))}"

    ax.set_xticklabels([_min_label(t) for t in ticks_log])

    # --- Meal / grazing divider + labels at the threshold ---
    x_div = np.log10(meal_threshold_min)
    ax.axvline(x_div, color="0.5", linestyle="--", linewidth=1.25)
    ymax = ax.get_ylim()[1]
    ax.text(x_div - 0.15, ymax * 1.05, "Pellets within a meal",
            color="0.5", ha="right", va="top", fontsize=11)
    ax.text(x_div + 0.15, ymax * 1.05, "Grazing pellets",
            color="0.5", ha="left", va="top", fontsize=11)

    ax.set_xlabel("Interpellet Interval (min)", fontsize=13)
    ax.set_ylabel("Pellet Density", fontsize=13)
    ax.set_title("")
    leg = ax.get_legend()
    if leg is not None:
        leg.set_title("")
        leg.set_frame_on(False)
    sns.despine(ax=ax)


### ------ L4 assembly ------- ###

def assemble_fr_l4(long_df, x_colors, bm_md, root_path,
                   fed_list, mapped_df, *, schematic_path=None, dpi=300):
    """
    Assemble the composite "L4" deliverable figure for one knockout model on the
    Fixed Ratio (FR1) task, mirroring the published FR figure:

        A) FED3 + FR1 task schematic (device image) with the gene name as the title
        B) Daily pellets bar
        C) Accuracy bar
        D) Poke time (s) bar
        E) Retrieval time (s) bar
        F) Inter-pellet-interval density (meal vs. grazing, log x-axis)
        G) Within-meal IPI (s) bar
        H) %Pellets within a meal bar

    plus a single shared Sex (Female/Male) legend and a caption. Every bar panel
    reports the genotype main effect from a two-way ANOVA, matching the caption.

    Orchestration only: the drawing lives in _plot_ipi_density_core (this module) and
    core._plot_metric_display so the layout logic here stays readable.

    Arguments:
        long_df : DataFrame
            Melted FR metrics (columns: variable / XGroup / HueGroup / value /
            Mouse_ID). Supplies every bar panel.
        x_colors : dict
            Group -> ipywidgets color widget (plain strings also tolerated).
        bm_md : DataFrame
            Per-session metadata; supplies the gene name (bm_md["Gene"][0]).
        root_path : str | Path
            Output root; the figure is written under <root_path>/L4.
        fed_list : list
            FED session DataFrames, used to draw the IPI density (panel F).
        mapped_df : DataFrame
            filename -> XGroup/HueGroup mapping, used to color the IPI density by group.
        schematic_path : str | Path | None
            Override for the panel-A image. Defaults to the packaged fr1 schematic.
        dpi : int
            Save resolution.

    Returns:
        out_path : Path
            Path to the saved composite SVG.
    """

    status.step("Assembling FR1 L4 composite figure")

    out_dir = Path(root_path, "L4")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Gene name that replaces the generic "HET" label throughout the figure.
    genename = bm_md["Gene"][0]

    long_df = long_df.copy()

    # --- Relabel zygosities to display form (identical convention to the PR L4) ---
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

    # A copy of the mapping relabeled to the final display labels, for the IPI panel.
    mapped_relabeled = mapped_df.copy()
    if "XGroup" in mapped_relabeled.columns:
        mapped_relabeled["XGroup"] = mapped_relabeled["XGroup"].replace(relabel)

    # ---------------- Figure + grid layout ----------------
    # Row 0: A (schematic) + B, C, D, E bars.
    # Row 1: F (IPI density, wide) + G, H bars.
    core.set_plot_style()   # one shared font family across every L4 figure
    fig = plt.figure(figsize=(16, 8))
    #gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[1.0, 1.0], hspace=0.5)
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[1.0, 1.0], hspace=0.45)

    gs_top = gs[0, :].subgridspec(1, 5, width_ratios=[1.71, 1, 1, 1, 1], wspace=0.5)
    gs_bot = gs[1, :].subgridspec(1, 3, width_ratios=[4.15, 1, 1], wspace=0.45)

    ##### Panel A: schematic #####
    if schematic_path is None:
        schematic_path = fedassets.get("fr1_schematic.jpg")

    ax_schem = fig.add_subplot(gs_top[0, 0])
    ax_schem.axis("off")
    # imshow forces aspect="equal", which shrinks this axes and centers it vertically
    # in its cell -> its top drops below the bar panels. Pin it to the top ("N") of
    # the cell so its top edge lines up with the bar axes, making axes-fraction
    # heights comparable across the row.
    #ax_schem.set_anchor("N")

    core._panel_label(ax_schem, "A)", dx=0.05, dy=0.75)

    if schematic_path is not None and Path(schematic_path).exists():
        ax_schem.imshow(mpimg.imread(str(schematic_path)))
    else:
        status.warn(f"L4: schematic image not found ({schematic_path}); panel A blank.")
    # y=1.08 matches _panel_label's default dy, so the gene title sits at the same
    # height as the B)/C)/... panel letters.
    ax_schem.set_title(genename, loc="left", fontsize=30, fontweight="bold", y=1.0)

    ##### Panels B-E: top-row metric bars #####
    top_specs = [
        ("Daily_Pellets", "Daily Pellets"),
        ("Accuracy",      "Accuracy"),
        ("PokeTime",      "Poke Time (s)"),
        ("RetrievalTime", "Retrieval Time (s)"),
    ]
    top_letters = ["B)", "C)", "D)", "E)"]

    shared_handles = []

    def _draw_bar(ax, metric, ylabel, letter):
        """Draw one metric bar panel; returns the sex-legend proxy handles."""
        core._panel_label(ax, letter)
        sub = long_df[long_df["variable"] == metric]
        if sub["value"].dropna().empty:
            status.warn(f"L4: no data for {metric}; leaving panel blank.")
            ax.axis("off")
            return []
        return core._plot_metric_display(
            sub, metric, ax, color_map,
            group_order=group_order, hue_order=hue_order, ylabel=ylabel,
            xlabel=(genename if multi else ""),
        )

    for col, (metric, ylabel) in enumerate(top_specs, start=1):   # cols 1..4 (col 0 = schematic)
        ax_bar = fig.add_subplot(gs_top[0, col])
        handles = _draw_bar(ax_bar, metric, ylabel, top_letters[col - 1])
        if handles and not shared_handles:
            shared_handles = handles

    ##### Panel F: inter-pellet-interval density #####
    ax_ipi = fig.add_subplot(gs_bot[0, 0])
    core._panel_label(ax_ipi, "F)", dx=-0.05, dy=1.05)
    _plot_ipi_density_core(fed_list, ax_ipi, mapped_relabeled, color_map, group_order)
    # In the multi-mutant case, prefix the gene name onto non-WT legend entries.
    if multi:
        leg = ax_ipi.get_legend()
        if leg is not None:
            for txt in leg.get_texts():
                if txt.get_text().upper() != "WT":
                    txt.set_text(f"{genename} {txt.get_text()}")

    ##### Panels G-H: bottom-row metric bars #####
    bot_specs = [
        ("InterPelletInterval", "Within-meal IPI (s)"),
        ("%MealPellets",        "%Pellets Within a Meal"),
    ]
    bot_letters = ["G)", "H)"]
    
    for i, (metric, ylabel) in enumerate(bot_specs):
        ax_bar = fig.add_subplot(gs_bot[0, i + 1])          # cols 2,3 (col 1 is a spacer)
        handles = _draw_bar(ax_bar, metric, ylabel, bot_letters[i])
        if handles and not shared_handles:
            shared_handles = handles

    # --- One shared Sex legend for the whole figure ---
    if shared_handles:
        fig.legend(
            handles=shared_handles, title="Sex",
            loc="upper right", frameon=False,
            bbox_to_anchor=(0.125, 0.66),
        )

    # --- Caption block beneath the panels ---
    caption = (
        "A) Schematic of the FED3 device and FR1 task. B, C, D, E) Bar graphs of the mean daily pellets, accuracy, poke time and retrieval\n" 
        "time respectively. F) Histogram of the distribution of inter-pellet intervals: pellets eaten less than 60 s apart (within a meal)\n"
        "vs. more than 60 s apart (grazing). G, H) Bar graphs of the mean inter-pellet interval (IPI) within a meal and the percentage of\n"
        "pellets within a meal.\n"
        "Statistics: two-way ANOVA; the reported p-value is the genotype effect (genotype x sex and sex effects are in the stats table)."
    )
    fig.text(0.1, 0.02, caption, ha="left", va="bottom", fontsize=14)

    # Reserve room at the bottom for the caption (tight_layout can't see fig.text).
    fig.subplots_adjust(bottom=0.24)

    out_path = out_dir / f"{genename}_FR_L4.svg"
    # bbox_inches="tight" keeps the caption and shared legend from being clipped.
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", format="svg")
    plt.close(fig)

    status.ok(f"FR1 L4 composite saved -> {out_path}")
    return out_path




#@@@@@@@@@@@@@@@@@@ DEFINE PIPELINE AND CLASS @@@@@@@@@@@@@@@@@@#
@dataclass
class FRResult:
    """Everything the L1->L4 run produced, so callers can inspect or re-plot
    any stage without rerunning the pipeline. Beats returning a bare tuple of
    a dozen values — attributes are self-documenting and order-independent."""
    fed_list: list
    key_df2: object
    saved_paths_indv: list
    fm_md: object          # essentailly the L3, bandit metrics and metadata
    fm_long: object        # melted version og bm_md
    l3_path: Path
    barplot_paths: list
    l4_path: Path




def run_fr_l1_l4(l1_path, key_path, root_path, *, bandittype = None, colors=None, dpi=300):
    """Run the full FR pipeline from an L1 zip to the L4 composite figure.

    Orchestration only — every step delegates to the existing public
    functions, so this stays a readable table of contents for the pipeline.

    Args:
        l1_path, key_path, root_path : the three inputs your notebook sets by hand.
        bandittype; String | None
            accepts "bandit100" & "bandit80" in order to properly create the schematics.
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

    # build a metakey
    meta_cols, md = core.build_metakey(key_df2, assay = "fr1")

    # Find ID columns
    id_col, other_id = core.pick_match_method(md)

    # ------- compute FR metrics ------- #
    fm = compute_fr_metrics(fed_list)

    fm_md = core.attach_meta(fm, md, id_col)

    # Build the l3
    fm_l3 = core.output_l3(fm_md, id_col, other_id, meta_cols, root_path, assay = "fr1")

    # ------ Plot the fr metrics ------ #
    # Build groupings
    mapped_df = core.build_group_selections(md)

    # attach the grouping dataframe to the metrics
    fm_grps_df = core.merge_group_selections(fm_md, mapped_df)

    # Melt the metric dataframe to long format
    fm_long = core.melt_metric(fm_grps_df, assay = "fr1")

    # define the aesthetics for plotting
    x_checks, x_colors, ordered_x = core.define_aesthetics(fm_long)

    # plot the metrics
    barplot_paths = core._run_plots(fm_long, x_checks, x_colors, ordered_x, root_path)

    # ------ Plot the inter-pellet interval ------ #
    interpellet_path = plot_interpellet_interval(fed_list, mapped_df, fm_md, x_colors, root_path)

    # ------ Assemble the L4 composite figure ------ #
    l4_path = assemble_fr_l4(fm_long, x_colors, fm_md, root_path, fed_list, mapped_df)