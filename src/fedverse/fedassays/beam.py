"""
Functions specificlly for BEAM assays and plotting from a BEAM device
William B. Rubio
Based off code from Chantelle Murrell
"""

# import dependancies
import pandas as pd
from pathlib import Path
import os
import tqdm
import numpy as np
from scipy.optimize import curve_fit
import re
from dataclasses import dataclass

# plotting
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import matplotlib.image as mpimg

# import cousins
from fedverse.fedutils.fedlog import status 
from fedverse.fedcore import core
from fedverse import fedassets

# SET GLOBAL VARIABLES
OMEGA = 2 * np.pi / 24.0
LIGHTS_OFF_HOUR = 18


#@@@@@@@@@@@@@@@@@@ INDIVIDUAL BEAM PLOTTING @@@@@@@@@@@@@@@@@@#

def plot_individual_beams(beam_list, out_path, dpi=150):
    """
    Plot activity_percent vs datetime for file at index idx.
    
    Arguments: 
        beam_list : List
            list of dataframes 
        out_path : Path
            Where the plots will be written to
        dpi : int
            resolution
    """
    #df = dataframes[idx].copy()
    #fname = files_list[idx]

    status.step("Beginning to construct BEAM plots")
    
    # intitate a empty list to capture the results
    saved_paths  = []
    
    # Create the directory that will be writen to
    out_dir = Path(out_path, "indv_bandit_plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    # iterate over all the BEAM dataframes
    for beam in beam_list:
        df = beam.copy()
        fname = getattr(df, 'name', f"File_{beam}")
        file_basename = os.path.basename(str(fname))

        # Attempt to recover datetime index
        if 'datetime' not in df.columns:
            df = df.reset_index()

        # sanity checks for datetime
        if 'datetime' not in df.columns:
            raise KeyError(f"'datetime' column not found in file {fname}")
        if 'activity_percent' not in df.columns:
            raise KeyError(f"'activity_percent' column not found in file {fname}")

        # ensure datetime dtype
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
        df = df.dropna(subset=['datetime'])
        df = df.sort_values('datetime')

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(df['datetime'], df['activity_percent'])
        ax.set_xlabel('Time of day')
        ax.set_ylabel('activity_percent')
        ax.set_title(f"{fname}")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # X axis: only 12am, 6 am, 12pm, 6 pm
        locator = mdates.HourLocator(byhour=[0, 6, 12, 18])
        ax.xaxis.set_major_locator(locator)

        def _fmt_lower_ampm(x, pos=None):
            dt = mdates.num2date(x)
            h = dt.hour
            if h == 0:
                return "12am"
            elif h == 6:
                return "6 am"
            elif h == 12:
                return "12pm"
            elif h == 18:
                return "6pm"
            else:
                return ""  # hide any stray ticks

        ax.xaxis.set_major_formatter(FuncFormatter(_fmt_lower_ampm))

        # Title = Mouse_ID (fallback to filename)
        title_text = str(df['Mouse_ID'].iloc[0]) if (df is not None and 'Mouse_ID' in df.columns and pd.notna(df['Mouse_ID'].iloc[0])) else file_basename
        ax.set_title(title_text)
        
        # ouput plots to directory
        fig.autofmt_xdate()
        plt.tight_layout()
        
        # suggest a base filename for saving
        safe_title = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in title_text)
        suggested = f"{safe_title}_indv_BEAM"
        
        write_path = out_dir / suggested
                
        fig.savefig(write_path, dpi=dpi, bbox_inches="tight")
        
        # Close the current figure to save memory
        plt.close(fig)
        
        saved_paths.append(write_path)
        status.ok(f"saved {write_path.name}")

    return saved_paths



#@@@@@@@@@@@@@@@@@@ ANALYZE BEAM METRICS @@@@@@@@@@@@@@@@@@#

# ---------- helpers ---------- #
def cosinor(t, mesor, amplitude, acrophase):
    t = np.asarray(t, dtype=float)
    return mesor + amplitude * np.cos(OMEGA * t + acrophase)

def clock_to_zt_unsigned(hours, lights_off_hour=LIGHTS_OFF_HOUR):
    """Clock hour(s) [0..23] -> ZT [0..24), ZT0 at lights-off."""
    return (np.asarray(hours, dtype=float) - lights_off_hour) % 24

def zt_unsigned_to_signed(zt_hours):
    """ZT [0..24) -> signed ZT [-12, +12)."""
    zt = np.asarray(zt_hours, dtype=float)
    return ((zt + 12) % 24) - 12

def find_col(df, cands):
    """Case/format-insensitive column finder."""
    norm = lambda s: "".join(ch for ch in str(s).lower() if ch.isalnum())
    cols_norm = {norm(c): c for c in df.columns}
    for cand in cands:
        k = norm(cand)
        if k in cols_norm:
            return cols_norm[k]
    return None

def _inner_basename(s):
    # "zip.zip::INNER.csv" or ".../INNER.csv" -> "INNER.csv"
    p = re.split(r"::|:", str(s))[-1]
    return os.path.basename(p.replace("\\", "/"))

# Simple genotype normalization: Wt, Het, Hom, Hemi
def norm_genotype(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower().replace(" ", "")
    if s == "wt":
        return "WT"
    if s == "WT; -VE":
        return "WT"
    if s == "het":
        return "Het"
    if s == "hom":
        return "Hom"
    if s == "hemi":
        return "Hemi"
    #if s == "hom/hemi":
    #   return "Hom/Hemi"
    # fallback: keep original string so you can see odd entries later
    return str(x)



def cosinor_per_file_on_zscore(data, metric_col="z_WTref", min_points=4, lights_off_hr = LIGHTS_OFF_HOUR):
    """
    Cosinor per file_key, using Mouse_ID from Key_Df (already merged).
    Fits in ZT where ZT0 = LIGHTS_OFF_HOUR.
    Uses the 'datetime' column explicitly.

    Amplitude_z in the output is **peak-to-trough range**
    (max fitted z – min fitted z) for the cosinor curve.
    """
    if "file_key" not in data.columns:
        raise RuntimeError("BEAM_data must contain 'file_key' (built from filename).")
    if "datetime" not in data.columns:
        raise RuntimeError("BEAM_data must contain a 'datetime' column (not just as index).")

    label_col = "file_label" if "file_label" in data.columns else "file_key"

    df_all = data.copy()
    # enforce datetime column type
    df_all["datetime"] = pd.to_datetime(df_all["datetime"], errors="coerce")
    df_all = df_all[df_all["datetime"].notna()].copy()
    df_all = df_all.sort_values("datetime")

    results = []
    global fit_storage
    fit_storage = []  # clear every run

    for file_key, df_file in df_all.groupby("file_key"):
        df_file = df_file.copy()
        file_label = df_file[label_col].iloc[0]

        # ---- Mouse_ID from merged key ----
        if "Mouse_ID" in df_file.columns:
            mid_series = (
                df_file["Mouse_ID"]
                .astype(str).str.strip()
                .replace("", np.nan)
                .dropna()
            )
        else:
            mid_series = pd.Series([], dtype=object)

        if len(mid_series) == 0:
            mouse_id = ""
            id_source = "unknown"
            mixed_ids = ""
        else:
            counts = mid_series.value_counts()
            mouse_id = counts.index[0]
            id_source = "from_Key_Df"
            mixed_ids = ";".join(counts.index.tolist()) if len(counts) > 1 else mouse_id

        # ---- ZT hour from datetime column ----
        dt_series = df_file["datetime"]
        zt_hr = ((dt_series.dt.hour - lights_off_hr) % 24).astype(float)

        df_file = df_file.assign(ZT_hr=zt_hr)
        s = (
            df_file.loc[df_file["ZT_hr"].notna()]
                   .groupby("ZT_hr", dropna=True)[metric_col]
                   .mean()
                   .astype(float)
        )

        t_zt = s.index.to_numpy(dtype=float)  # ZT hours [0..23]
        y = s.values
        m = np.isfinite(y)
        t_zt = t_zt[m]
        y = y[m]

        mesor = amp_param = amp_range = acrophase = r2 = np.nan
        status = "insufficient_points"
        if t_zt.size >= min_points and np.nanstd(y) > 0:
            guess = [np.nanmean(y), (np.nanmax(y) - np.nanmin(y)) / 2, 0.0]
            try:
                params, _ = curve_fit(
                    cosinor, t_zt, y, p0=guess, maxfev=20000
                )
                mesor, amp_param, acrophase = map(float, params)
                if amp_param < 0:
                    amp_param = -amp_param
                    acrophase += np.pi
                # fitted curve at observed times
                y_fit = cosinor(t_zt, mesor, amp_param, acrophase)
                # amplitude as peak-to-trough range on the fitted curve
                amp_range = float(np.nanmax(y_fit) - np.nanmin(y_fit))

                ss_res = float(np.nansum((y - y_fit) ** 2))
                ss_tot = float(np.nansum((y - np.nanmean(y)) ** 2))
                r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
                status = "ok"
            except Exception:
                status = "fit_failed"

        # Peak in ZT
        peak_zt_unsigned = float(((-acrophase) % (2 * np.pi)) / OMEGA) if np.isfinite(acrophase) else np.nan
        peak_zt_signed   = float(zt_unsigned_to_signed(peak_zt_unsigned)) if np.isfinite(peak_zt_unsigned) else np.nan
        peak_clock_hr    = float((peak_zt_unsigned + lights_off_hr) % 24) if np.isfinite(peak_zt_unsigned) else np.nan

        results.append({
            "file": file_label,
            "file_key": file_key,
            "Mouse_ID": mouse_id,
            "ID_source": id_source,
            "Mouse_IDs_seen_in_file": mixed_ids if id_source == "from_Key_Df" else "",
            "MESOR_z": mesor,
            "Amplitude_z": amp_range,
            "Acrophase_ZT_signed":   peak_zt_signed,    # [-12, +12)
            "Cosinor_R2": r2,
            "N_hours_used": int(t_zt.size),
            "Fit_Status": status,
        })

        fit_storage.append({
            "file": file_label,
            "file_key": file_key,
            "Mouse_ID": mouse_id,
            "t_zt_unsigned": t_zt,
            "y_z": y,
            "fit_params": (mesor, amp_param, acrophase),
            "fit_peak_zt_unsigned": peak_zt_unsigned,
            "r_squared": r2,
            "status": status,
            "id_source": id_source,
            "lights_off_hr": lights_off_hr,
        })

    return pd.DataFrame(results)



# ---------- plotting ----------
def plot_file_fit(index):
    d = fit_storage[index]
    file_label = d["file"]
    t_zt_unsigned = d["t_zt_unsigned"]
    y = d["y_z"]
    mesor, amplitude, acrophase = d["fit_params"]
    peak_zt_unsigned = d["fit_peak_zt_unsigned"]
    r2 = d["r_squared"]
    status = d["status"]
    lights_off_hr = d["lights_off_hr"]

    # Convert x-values to signed ZT for a centered plot
    x_obs = zt_unsigned_to_signed(t_zt_unsigned)
    order_obs = np.argsort(x_obs)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x_obs[order_obs], y[order_obs], "o", label="Observed hourly z")

    if np.isfinite(mesor) and np.isfinite(amplitude) and np.isfinite(acrophase):
        t_fit_zt = np.linspace(0, 23.999, 1000)
        y_fit = cosinor(t_fit_zt, mesor, amplitude, acrophase)
        x_fit = zt_unsigned_to_signed(t_fit_zt)
        order_fit = np.argsort(x_fit)
        ax.plot(x_fit[order_fit], y_fit[order_fit], "-", label=f"Cosinor ({status})")

        if np.isfinite(peak_zt_unsigned):
            ax.axvline(
                zt_unsigned_to_signed(peak_zt_unsigned),
                linestyle="--",
                label="Acrophase (ZT)",
            )

    ax.set_title(f"{file_label}  |  ZT0=lights off ({lights_off_hr:02d}:00)")
    ax.set_xlabel("ZT (hours)   [−12 … +12; ZT0 at lights off]")
    ax.set_ylabel("Z of Activity")
    ax.set_xlim(-12, 12)
    ax.set_xticks([-12, -6, 0, 6, 12])
    sns.despine()
    ax.legend(frameon=False)
    plt.show()




def output_beam_l3(beam_list, md, out_path, id_col="Mouse_ID", lights_off_hr = LIGHTS_OFF_HOUR ):
    """
    BEAM workflow is different enough that the output L3 for beam cannot be replicated through the 
    core l3.
    
    Arguments:
        beam_list; List
            List contating multiple BEAM Dataframes
        md; Dataframe
            meta dataframe that contatins cropped bandit key created by build_metakey   
        id_col; String, default "Mouse_ID"
            String detailing what column will serve as the key column to link both metadata and metric files
        lights_off:
            When the lights turn off
    Returns:
        beammetrics; Dataframe
            Dataframe containing aggragate metrics from all mice for Bandit assays
    """

    # create the directory for L3
    status.step("Preparing BEAM L3")
    
    # Create the directory that will be writen to
    out_dir = Path(out_path, "L3")
    out_dir.mkdir(parents=True, exist_ok=True)


    rows = []
    
    for idx in tqdm.tqdm(range(len(beam_list))):
        c_df = beam_list[idx]
        fname = core._basename(getattr(c_df, "name", f"File_{idx}"))
        c_df["filename"]   = fname
        c_df["file_key"]   = fname   # use filename as file_key
        c_df["file_label"] = fname
        rows.append(c_df)

    raw = pd.concat(rows, ignore_index=True)

    # Attempt to recover datetime index
    if 'datetime' not in raw.columns:
        raw = raw.reset_index()
    
    # ensure datetime remains a COLUMN
    if "datetime" not in raw.columns:
        raise RuntimeError("BEAM CSVs must contain a 'datetime' column.")
    raw["datetime"] = pd.to_datetime(raw["datetime"], errors="coerce")
    raw = raw[raw["datetime"].notna()].copy()
    raw = raw.sort_values("datetime")

    # prepare key for merge (use basename of filename)
    md = md.copy()
    if "filename" not in md.columns:
        raise RuntimeError("Key_Df must contain a 'filename' column.")
    md["filename"] = md["filename"].astype(str).map(os.path.basename)

    # merge all key metadata directly; no Mouse_ID remapping here
    beam_data = raw.merge(md, on="filename", how="left", suffixes=("", "_key"))

    # basic sanity: Mouse_ID and Gene should exist after merge
    if id_col not in beam_data.columns:
        raise RuntimeError("Key_Df must contain 'Mouse_ID' column (and it must match filenames).")
    if find_col(beam_data, ("Gene", "gene", "GENE")) is None:
        raise RuntimeError("Key_Df must contain a 'Gene' column for per-Gene WT normalization.")


    # --- find activity column & compute WT-referenced z (per Gene) --- #
    # get clumns we are interested in
    activity_candidates = core.get_cols("beam", "activity", status)
    activity_col = find_col(beam_data, activity_candidates)

    # Coerce numeric; strip '%' if present
    if (
        beam_data[activity_col].dtype == object
        and beam_data[activity_col].astype(str).str.contains("%").any()
    ):
        beam_data[activity_col] = (
            beam_data[activity_col]
            .astype(str)
            .str.replace("%", "", regex=False)
        )
    beam_data[activity_col] = pd.to_numeric(beam_data[activity_col], errors="coerce")

    # Gene column from merged key
    gene_col = find_col(beam_data, ("Gene", "gene", "GENE"))
    if gene_col != "Gene":
        beam_data.rename(columns={gene_col: "Gene"}, inplace=True)

    # Genotype column from merged key (includes case like 'Genotype$')
    geno_col = find_col(
        beam_data,
        ("Genotype$", "Genotype", "genotype", "GENOTYPE", "Gt", "GT", "geno")
    )
    if geno_col is None:
        raise RuntimeError("Key_Df / BEAM_data must contain a Genotype column (e.g. 'Genotype$').")
    
    beam_data["Genotype_raw"] = beam_data[geno_col]
    beam_data["Genotype_norm"] = beam_data["Genotype_raw"].map(norm_genotype)

    # WT rows per Gene (Wt is the only reference)
    wt_df = beam_data[beam_data["Genotype_norm"] == "WT"].copy()
    if wt_df.empty:
        raise RuntimeError("No Wt rows found in BEAM_data; cannot form per-Gene WT references.")

    # WT mean and SD within each Gene
    gene_stats = (
        wt_df
        .groupby("Gene")[activity_col]
        .agg(["mean", "std"])
        .rename(columns={"mean": "mu_geneWT", "std": "sd_geneWT"})
    )

    # Attach Gene-specific WT stats back to all rows
    beam_data = beam_data.merge(gene_stats, on="Gene", how="left")

    # Warn for Genes with no WT (they'll get NaN z-scores)
    missing_ref_genes = beam_data.loc[beam_data["mu_geneWT"].isna(), "Gene"].dropna().unique()
    if len(missing_ref_genes) > 0:
        status.warn(
            "Warning: these Genes have no Wt animals and will have NaN z_WTref:",
            list(missing_ref_genes),
        )

    # Avoid divide-by-zero
    beam_data.loc[beam_data["sd_geneWT"] == 0, "sd_geneWT"] = np.nan

    # Final per-Gene WT-referenced z-score (per-row)
    beam_data["z_WTref"] = (
        beam_data[activity_col] - beam_data["mu_geneWT"]
    ) / beam_data["sd_geneWT"]


    # --- per-file raw mean activity (no z-score) --- #
    mean_activity_df = (
        beam_data
        .dropna(subset=[activity_col])
        .groupby("filename")[activity_col]
        .mean()
        .reset_index()
        .rename(columns={activity_col: "Mean_activity"})
    )

    # --- Night_z and Day_z per file (before plotting) --- #
    # night and day should be
    # Night: ZT in [0, 12)
    # Day:   ZT in [12, 24)

    tmp = beam_data[["file_key", "filename", "datetime", "z_WTref"]].copy()
    tmp["datetime"] = pd.to_datetime(tmp["datetime"], errors="coerce")
    tmp = tmp.dropna(subset=["datetime", "z_WTref"])

    clock_hour = tmp["datetime"].dt.hour.astype(float)
    zt_hour = (clock_hour - lights_off_hr) % 24

    tmp["phase"] = np.where(zt_hour < 12, "Night", "Day")

    phase_agg = (
        tmp.groupby(["file_key", "filename", "phase"])["z_WTref"]
        .mean()
        .unstack("phase")
        .rename(columns={"Night": "Night_z", "Day": "Day_z"})
        .reset_index()
    )


    # --- run cosinor --- #
    cosinor_df = cosinor_per_file_on_zscore(beam_data, metric_col="z_WTref")
    status.ok(cosinor_df["Fit_Status"].value_counts(dropna=False))


    # --- Build beam_l3 (Key_Df + cosinor + Night/Day z + raw activity) --- #
    # Make a copy of cosinor_df and align filename column name with Key_Df
    metrics_df = cosinor_df.copy()
    if "file" not in metrics_df.columns:
        raise RuntimeError("cosinor_df does not have a 'file' column; check the cosinor code.")
    metrics_df["filename"] = metrics_df["file"].astype(str).map(os.path.basename)

    # 3. Merge: one row per file with all key metadata + cosinor metrics
    beam_l3 = md.merge(
        metrics_df.drop(columns=["file"]),  # drop old 'file' to avoid confusion
        on="filename",
        how="left",                         # keep all files from Key_Df
        suffixes=("", "_cosinor")
    )

    # 4. Merge Night_z and Day_z
    beam_l3 = beam_l3.merge(
        phase_agg[["filename", "Night_z", "Day_z"]],
        on="filename",
        how="left"
    )

    # 5. Merge raw mean activity (no z-score)
    beam_l3 = beam_l3.merge(
        mean_activity_df,
        on="filename",
        how="left"
    )

    # Drop plumbing columns we don't want in the output
    cols_to_drop = [
        "file_key",
        "ID_source",
        "Mouse_IDs_seen_in_file",
        "Fit_Status",
        "N_hours_used",
    ]
    drop_existing = [c for c in cols_to_drop if c in beam_l3.columns]
    beam_l3 = beam_l3.drop(columns=drop_existing)




    ### get name of file
    example = beam_l3.iloc[0]

    gene_name = str(example.get("Gene", "BEAM")).replace(" ", "_")
    gene_id_raw = example.get("Gene_ID", "000")
    try:
        gene_id = f"{int(gene_id_raw):03d}"
    except Exception:
        gene_id = str(gene_id_raw).replace(" ", "_")

    fname = f"{gene_name}_{gene_id}_BEAM_L3.csv"

    # Return 
    status.ok("L3 prepared")
    status.preview(beam_l3, msg="L3 Dataframe")

    write_path = out_dir / fname
    beam_l3.to_csv(write_path, index=False)

    return beam_data, beam_l3

    

#@@@@@@@@@@@@@@@@@@ COSINOR BEAM PLOT @@@@@@@@@@@@@@@@@@#


# ---------- helpers ---------- #

def zt_unsigned_to_signed(zt_hours):
    """ZT [0..24) -> signed ZT [-12, +12)."""
    zt = np.asarray(zt_hours, dtype=float)
    return ((zt + 12) % 24) - 12

def cosinor_func(t, mesor, amplitude, acrophase):
    t = np.asarray(t, dtype=float)
    return mesor + amplitude * np.cos(OMEGA * t + acrophase)

def _safe_filename(s):
    s = str(s).strip()
    s = re.sub(r"[^\w\-\.]+", "_", s)   # keep letters/numbers/_-. ; replace others with _
    s = re.sub(r"_+", "_", s).strip("_")
    return s if s else "Gene"


# ---------- Create the cosinor plot ---------- #

def plot_xgroup_cosinor(beam_data, files_to_group_x, x_colors, ordered_x, out_path = None,
    lights_off_hour=LIGHTS_OFF_HOUR, metric_col="z_WTref", show_points = True, write = True,
    ax = None):

    
    status.step("creating cosinor plot")

    
    if write:
        # ensure that if write there is a path set
        if out_path == None:
            status.fail("Writing with no destined directory: Please specifcy directory or set write to 'False'")

        # make directory to write to
        out_dir = Path(out_path, "x_cosinor")
        out_dir.mkdir(parents=True, exist_ok=True)

    # line styles per genotype
    ls_map = {
        "WT":   "-",
        "HET":  "--",
        "HOM/HEMI":  ":",
    }

    # Pull out the gene name (needed by the relabel lines below).
    genename = beam_data["Gene"][0]

    needed = ["datetime", "filename", "Mouse_ID", metric_col]
    missing = [c for c in needed if c not in beam_data.columns]
    if missing:
        raise RuntimeError(f"beam_data is missing: {missing}")

    if files_to_group_x is None or files_to_group_x.empty:
        raise RuntimeError("files_to_group_x is missing or empty. Run the X grouping cell and click Build Groups first.")

    df = beam_data[needed].copy()
    df["filename"] = df["filename"].apply(lambda p: os.path.basename(str(p)))

    xmap = files_to_group_x.rename(columns={"Group": "XGroup"}).copy()
    xmap["filename"] = xmap["filename"].apply(lambda p: os.path.basename(str(p)))

    df = df.merge(xmap[["filename", "XGroup"]], on="filename", how="inner")

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df[df["datetime"].notna()].copy()
    df = df[np.isfinite(df[metric_col])].copy()

    df["ZT_hr"] = ((df["datetime"].dt.hour - lights_off_hour) % 24).astype(int)

    per_mouse_hour = (
        df.groupby(["XGroup", "Mouse_ID", "ZT_hr"], observed=True)[metric_col]
          .mean()
          .reset_index()
    )

    grp = (
        per_mouse_hour.groupby(["XGroup", "ZT_hr"], observed=True)[metric_col]
                      .mean()
                      .reset_index()
                      .rename(columns={metric_col: "group_mean"})
    )

    fit_rows = []

    for xgroup, sub in grp.groupby("XGroup", observed=True):
        t = sub["ZT_hr"].to_numpy(dtype=float)
        y = sub["group_mean"].to_numpy(dtype=float)

        if len(t) < 4 or np.nanstd(y) == 0:
            fit_rows.append({
                "XGroup": xgroup,
                "MESOR": np.nan,
                "Amplitude": np.nan,
                "Acrophase_ZT_unsigned": np.nan,
                "Acrophase_ZT_signed": np.nan,
                "Cosinor_R2": np.nan,
                "N_hours_used": len(t),
            })
            continue

        guess = [np.nanmean(y), (np.nanmax(y) - np.nanmin(y)) / 2.0, 0.0]

        try:
            params, _ = curve_fit(cosinor_func, t, y, p0=guess, maxfev=20000)
            mesor, amplitude, phi = map(float, params)

            if amplitude < 0:
                amplitude = -amplitude
                phi += np.pi

            y_fit = cosinor_func(t, mesor, amplitude, phi)
            ss_res = float(np.nansum((y - y_fit) ** 2))
            ss_tot = float(np.nansum((y - np.nanmean(y)) ** 2))
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

            peak_zt_u = ((-phi) % (2 * np.pi)) / OMEGA
            peak_zt_s = zt_unsigned_to_signed(peak_zt_u)

        except Exception:
            mesor = amplitude = r2 = peak_zt_u = peak_zt_s = np.nan

        fit_rows.append({
            "XGroup": xgroup,
            "MESOR": mesor,
            "Amplitude": amplitude,
            "Acrophase_ZT_unsigned": peak_zt_u,
            "Acrophase_ZT_signed": peak_zt_s,
            "Cosinor_R2": r2,
            "N_hours_used": len(t),
        })

    # --- Resolve plotting order: honor ordered_x, append any leftover groups ---
    present_groups = grp["XGroup"].unique().tolist()
    if ordered_x:
        group_order = [g for g in ordered_x if g in present_groups]
        group_order += [g for g in present_groups if g not in group_order]
    else:
        group_order = sorted(present_groups, key=str)

    fits_xgroup = (
        pd.DataFrame(fit_rows)
        .assign(XGroup=lambda d: pd.Categorical(d["XGroup"], categories=group_order, ordered=True))
        .sort_values("XGroup")
        .reset_index(drop=True)
    )

    # --- Resolve colors: x_colors may hold ipywidgets (.value) or plain strings ---
    def _resolve_color(g, default="tab:blue"):
        c = x_colors.get(g, None) if x_colors else None
        val = getattr(c, "value", c)
        val = val.strip() if isinstance(val, str) else ""
        return val or default

    sns.set_style("white")
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.axvspan(0, 12, color="0.9", zorder=-1)

    color_map = {g: _resolve_color(g) for g in group_order}


    for xgroup in group_order:
        sub = grp[grp["XGroup"] == xgroup].sort_values("ZT_hr")

        x_u = sub["ZT_hr"].to_numpy(dtype=float)
        x_s = zt_unsigned_to_signed(x_u)
        y_mean = sub["group_mean"].to_numpy(dtype=float)

        order = np.argsort(x_s)
        x_s = x_s[order]
        y_mean = y_mean[order]

        color = color_map[xgroup]
        ls = ls_map.get(xgroup, "-")

        if show_points:
            ax.plot(x_s, y_mean, "o", color=color, markersize=6, alpha=0.8)

        row_fit = fits_xgroup[fits_xgroup["XGroup"] == xgroup]

        if not row_fit.empty and np.isfinite(row_fit["MESOR"].iloc[0]):
            mesor = float(row_fit["MESOR"].iloc[0])
            amp = float(row_fit["Amplitude"].iloc[0])
            peak_u = float(row_fit["Acrophase_ZT_unsigned"].iloc[0])
            peak_s = float(row_fit["Acrophase_ZT_signed"].iloc[0])

            phi = ((-peak_u * OMEGA) % (2 * np.pi))

            t_fit_u = np.linspace(0, 23.999, 400)
            y_fit = cosinor_func(t_fit_u, mesor, amp, phi)
            x_fit_s = zt_unsigned_to_signed(t_fit_u)

            m_neg = x_fit_s < 0
            m_pos = ~m_neg
            label_done = False

            if m_neg.any():
                idx = np.argsort(x_fit_s[m_neg])
                ax.plot(
                    x_fit_s[m_neg][idx],
                    y_fit[m_neg][idx],
                    color=color,
                    linestyle=ls,
                    linewidth=2,
                    label=xgroup,
                )
                label_done = True

            if m_pos.any():
                idx = np.argsort(x_fit_s[m_pos])
                ax.plot(
                    x_fit_s[m_pos][idx],
                    y_fit[m_pos][idx],
                    color=color,
                    linestyle=ls,
                    linewidth=2,
                    label=None if label_done else xgroup,
                )

            ax.axvline(
                peak_s,
                color=color,
                linestyle=":",
                linewidth=1.5,
                alpha=0.9,
            )

        else:
            ax.plot(x_s, y_mean, color=color, linestyle=ls, linewidth=2, label=xgroup)

    ax.set_xlim(-12, 12)
    ax.set_xticks([-12, -6, 0, 6, 12])
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel("ZT", fontsize = 18)
    ax.set_ylabel("Activity (Z)", fontsize = 18)
    #ax.set_title("Cosinor by X grouping")
    sns.despine(ax=ax)

    ax.legend(frameon=False, bbox_to_anchor=(0.0, 1), loc="upper left")

    fig = ax.get_figure()  # or plt.gcf() if you prefer

    # --- only manage the figure lifecycle when we created it ourselves; ---
    # --- an ax passed in (e.g. from assemble_beam_l4) belongs to the caller ---
    if own_fig:
        plt.tight_layout()

        # --- save the plot --- #
        if write:
            safe_name = genename + "_xgrouped_cosinor"
            out_path = out_dir / f"{safe_name}.png"

            # Save at print-friendly resolution; bbox_inches="tight" trims the
            # generous whitespace left by the 2-panel layout + rotated x-labels.
            fig.savefig(out_path, dpi=300, bbox_inches="tight")

        plt.close(fig)

    status.ok("X grouped cosinor plot created")

    # return figure and other metrics
    return fits_xgroup, grp, fig




#@@@@@@@@@@@@@@@@@@ ASSEMBLE BEAM L4 @@@@@@@@@@@@@@@@@@#


def assemble_beam_l4(long_df, beam_data, mapped_df, x_colors, out_path,
                schematic_path=None, dpi=300, figsize=(12, 10), h_space = 0.3, 
                font_size=14, leading = 1.4):
    """
    Assemble the composite "L4" deliverable figure for one knockout model:

        A) device schematic
        B) Activite inactive phase bar
        C) Activite active phase bar
        D) Activity z timeplot
        E) Amplitude z bar
        F) MESOR z bar
        G) acrophase ZT bar

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
        out_path : str | Path
            Output root; the figure is written under <out_path>/L4.
        schematic_path : str | Path | None
            Optional device/behaviour image for the top-left of panel A. If None,
            that corner is left blank for manual assembly.
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
            Path to the saved composite PNG.
    """


    status.step("Assembling L4 composite figure")

    # --- create a directory to write to --- #
    out_dir = Path(out_path, "L4")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Gene name that replaces the generic "HET" label throughout the figure.
    genename = beam_data["Gene"][0]



    # --- Resolve colors to a plain {group: color} dict ONCE ---
    # x_colors may hold ipywidgets (with .value) or plain strings; tolerate both.
    def _resolve(g):
        c = x_colors.get(g, None)
        val = getattr(c, "value", c)                 
        val = val.strip() if isinstance(val, str) else ""
        return val or "tab:blue"
    color_map = {g: _resolve(g) for g in x_colors}

    # mnake copies
    long_df = long_df.copy()
    beam_data = beam_data.copy()


        # --- Relabel zygosities to display form ---
    # Single mutant   -> bare gene name on the bar.
    # Multiple mutants -> zygosity on the bar, gene name on the x-axis.
    zygotic_order = ["HET", "HOM", "HOM/HEMI", "HEMI"]
    zyg_display   = {"HET": "Het", "HOM": "Hom", "HOM/HEMI": "Hom/Hemi", "HEMI": "Hemi"}

    # find all XGroups present in the data (ignore HueGroup)
    present = long_df["XGroup"].dropna().unique().tolist()


    # Isolate all non-wt group names
    non_wt = [g for g in present if str(g).upper() != "WT"]
    multi  = len(non_wt) > 1

    if not multi:
        relabel = {g: genename for g in non_wt}
    else:
        relabel = {g: zyg_display.get(str(g).upper(), str(g).title()) for g in non_wt}

    long_df["XGroup"] = long_df["XGroup"].replace(relabel)



    # --- Order: WT first, then mutants by zygotic_order (unknowns sort last) ---
    def _zygo_rank(orig_label):
        up = str(orig_label).upper()
        return zygotic_order.index(up) if up in zygotic_order else len(zygotic_order)

    # conduct the actual ordering of the mutant groups (WT is always first)
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
    # Row 0: A (schematic) + B, C bars.
    # Row 1: D (Cosinor plot) + E, F, G bars.
    core.set_plot_style()   # one shared font family across every L4 figure
    #fig = plt.figure(figsize=(16, 8))
    fig_w, fig_h = figsize
    fig = plt.figure(figsize=(fig_w, fig_h))
    

    # Determine head space based on groups and if hemi is present. 
    #   If hemi is present, we need more space for the legend. add padding
    if any("HEMI" in str(g).upper() for g in present):
        print("L4: HEMI present - adding padding to hspace")
        h_space = h_space + 0.3
        cap_space = 0.29
    else:
        h_space = h_space
        cap_space = 0.25

    # define the gridspec for the figure
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[1.0, 1.0], hspace = h_space)

    # define the subgridspec for the top and bottom rows
    gs_top = gs[0, :].subgridspec(1, 3, width_ratios=[2.5, 1, 1], wspace=0.5)
    gs_bot = gs[1, :].subgridspec(1, 4, width_ratios=[4.15, 1, 1, 1], wspace=0.45)



    ##### Panel A: schematic #####
    if schematic_path is None:
        schematic_path = fedassets.get("beam_schematic.jpg")

    ax_schem = fig.add_subplot(gs_top[0, 0])
    ax_schem.axis("off")
    #ax_schem.set_anchor("N")

    core._panel_label(ax_schem, "A)", dx=0.00, dy=0.75)

    if schematic_path is not None and Path(schematic_path).exists():
        ax_schem.imshow(mpimg.imread(str(schematic_path)))
    else:
        status.warn(f"L4: schematic image not found ({schematic_path}); panel A blank.")
    # y=1.08 matches _panel_label's default dy, so the gene title sits at the same
    # height as the B)/C)/... panel letters.
    ax_schem.set_title(genename, loc="left", fontsize=30, fontweight="bold", y=1.0)


    ##### Panels B-E: top-row metric bars #####
    top_specs = [
        ("Day_z", "Activity Z (inactive phase)"),
        ("Night_z", "Activity Z (active phase)"),
    ]
    top_letters = ["B)", "C)"]

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

    ##### Panel D: cosinor grouped by X #####
    ax_cos = fig.add_subplot(gs_bot[0, 0])
    core._panel_label(ax_cos, "D)", dx=-0.05, dy=1.05)

    # mapped_df already carries raw "filename"/"XGroup" columns (pre-relabel),
    # matching what plot_xgroup_cosinor and ls_map/x_colors expect as group keys.
    raw_group_order = ["WT"] + non_wt_sorted
    plot_xgroup_cosinor(
        beam_data, mapped_df, x_colors, raw_group_order,
        ax=ax_cos, write=False,
    )

    ##### Panels G-H: bottom-row metric bars #####
    bot_specs = [
        ("Amplitude_z", "Amplitude Z"),
        ("MESOR_z", "MESOR Z"),
        ("Acrophase_ZT_signed", "Acrophase ZT")
    ]
    bot_letters = ["E)", "F)", "G)"]
    
    for i, (metric, ylabel) in enumerate(bot_specs):
        ax_bar = fig.add_subplot(gs_bot[0, i + 1])          # cols 2,3 (col 1 is a spacer)
        handles = _draw_bar(ax_bar, metric, ylabel, bot_letters[i])
        if handles and not shared_handles:
            shared_handles = handles

    ##### Caption block beneath the panels #####
    # Assembled line-by-line (each string ends with a space) so the joins never
    # run words together, with explicit \n where a visible line break is wanted.
    caption_raw = (
        "A) BEAM device and example activity trace. B, C) Bar graphs showing the mean Z scored "  
        "activity in the inactive phase and the active phase respectively D) The average z score " 
        "activity per hour for each group (dots) and the cosinor fit of the activity (lines). E, "
        "F, G Bar graphs showing the mean amplitude, MESOR and acrophase of the z scored activity. "
        "The wildtype acitivty is Z scored and the gene of interest is compared to the WT hence a 0 "
        "MESOR for WT mice. For statistics a two-way ANOVA was run, here we report the genotype effect "
        "for the genotype x sex and sex effects refer to the stats table."
    )

    caption = core._wrap_caption(caption_raw, fig_w, fontsize=font_size)
    fig.text(0.1, 0.02, caption, ha="left", va="bottom", fontsize=font_size)
    
    # Reserve room at the bottom for the caption (tight_layout can't see fig.text).
    n_lines = caption.count("\n") + 1
    
    # line height in fig fraction: fontsize pts * ~ 1.6 leading / figure height in pts
    cap_frac = n_lines * font_size * leading / (fig_h * 72)
    print(f"L4: caption {n_lines} lines, reserving {cap_frac:.3f} fig fraction at bottom.")
    
    # figure out padding crudely based on the longest group label, so the x-axis labels don't overlap the caption.
    longest = max((len(str(g)) for g in group_order), default=0)
    # ~0.011 fig-fraction per char at fontsize 14 on a 9in figure; tune the constant
    label_depth = longest * 0.011 * (9 / fig_h)
    
    fig.subplots_adjust(bottom=cap_frac + label_depth)


    
    ##### Save the figure ##### 
    # save as an SVG for vector graphics and future editing.
    out_path = out_dir / f"{genename}_BEAM_L4.svg"
    # bbox_inches="tight" keeps the caption and shared legend from being clipped.
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", format="svg")
    plt.close(fig)

    status.ok(f"FR1 L4 composite saved -> {out_path}")
    return out_path


#@@@@@@@@@@@@@@@@@@ DEFINE PIPELINE AND CLASS @@@@@@@@@@@@@@@@@@#
@dataclass
class BEAMResult:
    """Everything the L1->L4 run produced, so callers can inspect or re-plot
    any stage without rerunning the pipeline. Beats returning a bare tuple of
    a dozen values — attributes are self-documenting and order-independent."""
    beam_list: list
    key_df2: object
    beam_data: object
    beam_l3: object
    beam_long: object 
    barplot_paths: list
    cos_fig: object
    stats_df: object
    l4_path: Path


def run_beam_l1_l4(l1_path, key_path, root_path, *, colors=None, dpi=300):
    """Run the full FR pipeline from an L1 zip to the L4 composite figure.

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
    beam_list, loaded_files, session_types = core.ingest_l1(l1_path)

    # ------ Create Meta Data Key ------ #
    # upload key file
    key_df, msg = core._read_key_from_upload(key_path)

    # build key dataframe
    key_df2 = core.build_or_rematch_key_df(loaded_files, session_types, key_df, msg_hint = f"Key status: {msg}")

    # build a metakey
    meta_cols, md = core.build_metakey(key_df2, assay = "beam")

    # ------- compute BEAM metrics & L3------- #
    beam_data, beam_l3 = output_beam_l3(beam_list, md, root_path)

    # ------ Plot the fr metrics ------ #
    # Build groupings
    mapped_df = core.build_group_selections(md)

    # attach the grouping dataframe to the metrics
    beam_grps_df = core.merge_group_selections(beam_l3, mapped_df)

    # Melt the metric dataframe to long format
    beam_long = core.melt_metric(beam_grps_df, assay = "beam")

    # define the aesthetics for plotting
    x_checks, x_colors, ordered_x = core.define_aesthetics(beam_long)

    # plot the metrics
    barplot_paths = core._run_plots(beam_long, x_checks, x_colors, ordered_x, root_path)

    # ------ Plot the grouped cosiner plots ------ #
    fits_xgroup, grp, cos_fig = plot_xgroup_cosinor(beam_data, beam_grps_df, x_colors, ordered_x, root_path)

    # ------ Build the stats table ------ #
    stats_df = core.build_stats_table(beam_long, ordered_x, root_path, assay="beam")

    # ------ Assemble the L4 composite figure ------ #
    l4_path = assemble_beam_l4(beam_long, beam_data, mapped_df, x_colors, root_path)


    ### ------ Return the frclass ------ ###
    return BEAMResult(beam_list, key_df2, beam_data, beam_l3, beam_long, barplot_paths, cos_fig, stats_df, l4_path)