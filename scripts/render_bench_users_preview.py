#!/usr/bin/env python3
"""Render inspection artifacts from already generated bench-user preview cases.

This script does no sampling, tracing, channel computation, or inference. It only
visualizes the case JSON, per-frame labels, and optional map geometry supplied by
the preview generator.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.collections import PolyCollection
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# v1 names. "no-los" = no clear direct segment (reflections are not traced here);
# the reflection class is a bounded one-bounce candidate flag, not a traced path.
LINK_COLORS = {"los": "#16834a", "no-los-reflection-candidate": "#e08c13", "no-los": "#c83d4d", "no-los-unresolved": "#7a8492", "unknown": "#7a8492"}
LEGACY_LINKS = {"blocked": "no-los", "obstructed": "no-los", "nlos-blocked": "no-los",
                "reflection-only": "no-los-reflection-candidate", "reflection": "no-los-reflection-candidate",
                "nlos-reflection": "no-los-reflection-candidate", "reflected": "no-los-reflection-candidate",
                "reflection-only-candidate": "no-los-reflection-candidate"}
RIS_COLOR = "#7a2b91"
HEIGHT_MARKERS = {"pedestrian": "o", "elevated": "^", "drone": "D", "unknown": "s"}


SOURCE_CREDITS = {
    "munich": "Geobasisdaten: Bayerische Vermessungsverwaltung – geodaten.bayern.de (Daten verändert), CC BY 4.0",
    "rotterdam": "3DBAG by 3D geoinformation research group and 3DGI (CC BY 4.0); terrain: AHN4 via PDOK",
    "maastricht": "3DBAG by 3D geoinformation research group and 3DGI (CC BY 4.0); terrain: AHN4 via PDOK",
}

def read_table(path: Path):
    """Read parquet when pyarrow/fastparquet exists, otherwise CSV for fixtures."""
    import pandas as pd
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    try:
        return pd.read_parquet(path)
    except Exception as exc:
        csv = path.with_suffix(".csv")
        if csv.exists():
            return pd.read_csv(csv)
        raise RuntimeError(f"cannot read {path}; install a parquet engine (pyarrow) or provide CSV") from exc


def _link(v: Any) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "unknown"
    s = str(v).strip().lower().replace("_", "-")
    if s in ("los", "line-of-sight", "direct-los"):
        return "los"
    s = LEGACY_LINKS.get(s, s)
    if s in ("unresolved",):
        return "no-los-unresolved"
    return s if s in LINK_COLORS else "unknown"


def load_geometry(root: Path, map_id: str):
    p = root / "geometry" / f"{map_id}.json"
    if not p.exists():
        return None, f"No geometry file: geometry/{map_id}.json"
    g = json.loads(p.read_text())
    terrain = g.get("terrain")
    if isinstance(terrain, dict) and terrain.get("file"):
        tp = (p.parent / terrain["file"]).resolve()
        if tp.exists():
            with np.load(tp) as a:
                terrain = {**terrain, **{k: a[k] for k in a.files}}
        else:
            terrain = None
    g["_terrain"] = terrain
    return g, None


def _terrain_arrays(t):
    if not isinstance(t, dict) or "z" not in t:
        return None
    z = np.asarray(t["z"], dtype=float)
    x = np.asarray(t.get("x", np.arange(z.shape[1])), dtype=float)
    y = np.asarray(t.get("y", np.arange(z.shape[0])), dtype=float)
    valid = np.asarray(t.get("valid", np.isfinite(z)), dtype=bool) & np.isfinite(z)
    zz = np.where(valid, z, np.nan)
    if z.ndim == 1:
        return None
    xx, yy = np.meshgrid(x, y)
    return xx, yy, zz


def render_case(root: Path, case_dir: Path, out_dir: Path):
    meta = json.loads((case_dir / "case.json").read_text())
    frame_path = next((p for p in (case_dir / "frames.parquet", case_dir / "frames.csv") if p.exists()), None)
    if frame_path is None:
        raise FileNotFoundError(f"no frames.parquet/csv in {case_dir}")
    df = read_table(frame_path)
    if "frame" in df.columns and "user_id" in df.columns:
        df = df.sort_values(["user_id", "frame"])
    g, warning = load_geometry(root, str(meta.get("map_id", "unknown")))
    out_dir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(15, 7.6), constrained_layout=True)
    plan = fig.add_subplot(121)
    ax = fig.add_subplot(122, projection="3d")
    terrain = _terrain_arrays(g.get("_terrain")) if g else None
    if terrain:
        xx, yy, zz = terrain
        stride = max(1, max(zz.shape) // 110)
        ax.plot_surface(xx, yy, zz, cmap="terrain", alpha=.42, linewidth=0, antialiased=False,
                        rstride=stride, cstride=stride, zorder=0)
        shade=plan.pcolormesh(xx[::stride,::stride],yy[::stride,::stride],
                            zz[::stride,::stride],cmap="terrain",shading="auto",alpha=.42,zorder=0)
        bar=fig.colorbar(shade,ax=plan,orientation="horizontal",fraction=.042,pad=.06)
        bar.set_label("Valid bare-earth DTM elevation (m)",fontsize=9)
    if g:
        outlines=[]
        for b in g.get("buildings", []):
            poly = b.get("polygon_xy", b.get("footprint_xy", b.get("polygon")))
            h = float(b.get("height_m", b.get("height", 8)) or 8)
            base = float(b.get("ground_m", 0) or 0)
            if not poly or len(poly) < 3:
                continue
            outlines.append(poly)
            top = [(float(p[0]), float(p[1]), base+h) for p in poly]
            bot = [(float(p[0]), float(p[1]), base) for p in poly]
            faces = [top, bot]
            for i in range(len(poly)):
                j = (i+1) % len(poly)
                faces.append([bot[i], bot[j], top[j], top[i]])
            pc = Poly3DCollection(faces, facecolors="#b8bec8", edgecolors="#58616d", linewidths=.35, alpha=.34)
            ax.add_collection3d(pc)
        plan.add_collection(PolyCollection(outlines,facecolors="#bbc2cd",edgecolors="#738094",
                                           linewidths=.35,alpha=.72,zorder=1))
    # User tracks are lightly drawn, then current/latest user positions are coded by link and height.
    for uid, rows in df.groupby("user_id", sort=False):
        rows = rows.sort_values("frame") if "frame" in rows else rows
        if len(rows) > 1:
            ax.plot(rows.x, rows.y, rows.z, color="#364152", alpha=.22, linewidth=.7)
            plan.plot(rows.x,rows.y,color="#253246",alpha=.33,linewidth=.75,zorder=2)
    latest = df.sort_values("frame").groupby("user_id", as_index=False).tail(1) if "frame" in df else df.drop_duplicates("user_id", keep="last")
    if "ris_los" in latest:
        reach = latest[latest.ris_los.astype(bool)]
        if len(reach):
            plan.scatter(reach.x, reach.y, s=150, facecolors="none", edgecolors=RIS_COLOR, linewidths=1.3, zorder=3)
    for (link, hc, served), rows in latest.assign(_link=latest.bs_link.map(_link) if "bs_link" in latest else "unknown",
              _hc=latest.height_class.fillna("unknown").astype(str) if "height_class" in latest else "unknown",
              _served=latest.served.fillna(False).astype(bool) if "served" in latest else False).groupby(["_link", "_hc", "_served"]):
        ax.scatter(rows.x, rows.y, rows.z, c=LINK_COLORS.get(link, LINK_COLORS["unknown"]),
                   marker=HEIGHT_MARKERS.get(hc.lower(), "s"), s=74 if served else 42,
                   edgecolors="black" if served else "white", linewidths=1.0 if served else .5,
                   label=f"{link}; {hc}; {'served' if served else 'crowd'}", depthshade=True)
        plan.scatter(rows.x,rows.y,c=LINK_COLORS.get(link,LINK_COLORS["unknown"]),
                     marker=HEIGHT_MARKERS.get(hc.lower(),"s"),s=54 if served else 25,
                     edgecolors="#111111" if served else "white",linewidths=1.5 if served else .65,
                     zorder=5 if served else 4)
    for key, label, color, marker in (("bs_xyz", "BS", "#192b88", "*") , ("ris_xyz", "RIS", "#7a2b91", "P")):
        xyz = meta.get(key)
        if xyz and len(xyz) >= 3:
            ax.scatter([xyz[0]], [xyz[1]], [xyz[2]], marker=marker, s=210, c=color, edgecolors="white", linewidths=1.2, label=label)
            ax.text(xyz[0], xyz[1], xyz[2], f" {label}", color=color, weight="bold")
            plan.scatter([xyz[0]],[xyz[1]],marker=marker,s=170,c=color,
                         edgecolors="white",linewidths=1.2,zorder=7)
            plan.annotate(label,(xyz[0],xyz[1]),xytext=(6,5),textcoords="offset points",
                          weight="bold",color=color,zorder=8,
                          bbox={"facecolor":"white","alpha":.78,"edgecolor":"none","pad":1.0})
    title = f"{meta.get('id', case_dir.name)}  |  {meta.get('kind','natural').title()}"
    n_ris = int(latest.ris_los.astype(bool).sum()) if "ris_los" in latest else 0
    n_srv_ris = int(latest[latest.served.astype(bool)].ris_los.astype(bool).sum()) if {"ris_los","served"}<=set(latest.columns) else 0
    fig.suptitle(title+f"  •  {len(latest)} people ({n_ris} with RIS LOS)  •  {int(meta.get('served_count',0))} served ({n_srv_ris} with RIS LOS)"
                 f"  •  BS-RIS {_fmt(meta.get('distance_bs_ris_m'))} m",
                 fontsize=14,weight="bold")
    plan.set(title="Street plan · final positions, walked paths, RIS-LOS rings",
             xlabel="Projected easting (m)",ylabel="Projected northing (m)",aspect="equal")
    plan.grid(color="#b5becd",alpha=.3)
    ax.set(title="3D context · building model heights",
           xlabel="Projected easting (m)", ylabel="Projected northing (m)", zlabel="Elevation (m)")
    if g and len(g.get("bounds",[])) == 4:
        xmin, xmax, ymin, ymax = g["bounds"]
        ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
        plan.set_xlim(xmin,xmax); plan.set_ylim(ymin,ymax)
        lows = [float(x) for x in df.z if np.isfinite(x)]
        highs = [float(b["ground_m"] or 0)+float(b["height_m"] or 0)
                 for b in g.get("buildings",[]) if "ground_m" in b and "height_m" in b]
        highs += [float(meta[k][2]) for k in ("bs_xyz","ris_xyz") if meta.get(k)]
        if lows:
            ax.set_zlim(min(lows)-5, max([*highs,max(lows)])+5)
        ax.set_box_aspect((1, 1, .48))
    ax.view_init(elev=32, azim=-57)
    ax.set_proj_type("ortho")
    point = lambda label,color,marker="o",edge=None: Line2D(
        [],[],marker=marker,linestyle="None",markersize=7,markerfacecolor=color,
        markeredgecolor=edge or color,label=label)
    observed = set(latest.bs_link.map(_link)) if "bs_link" in latest else set()
    encodings = [point(k,LINK_COLORS[k]) for k in LINK_COLORS if k in observed]
    encodings += [point("BS","#192b88","*"),point("RIS","#7a2b91","P"),
                  point("Served: black outline","white","o","black"),
                  Line2D([],[],marker="o",linestyle="None",markersize=11,markerfacecolor="none",
                         markeredgecolor=RIS_COLOR,label="RIS LOS (ring)")]
    if "height_class" in latest:
        encodings += [point(k,"#596879",HEIGHT_MARKERS[k]) for k in HEIGHT_MARKERS
                      if k in set(latest.height_class)]
    plan.legend(handles=encodings,loc="upper right",fontsize=8,framealpha=.96,ncol=2)
    if warning:
        ax.text2D(.01, .01, warning, transform=ax.transAxes, color="#9a3d32", fontsize=9)
    credit = SOURCE_CREDITS.get(str(meta.get("map_id","")), "Synthetic example")
    fig.text(.03,.006,credit+" • © OpenStreetMap contributors (openstreetmap.org/copyright)",
             fontsize=7,color="#526172")
    fig.savefig(out_dir / "map-3d.png", dpi=170)
    plt.close(fig)
    return meta, df, frame_path


def _fmt(v):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    if isinstance(v, (float, np.floating)):
        return f"{v:.2f}"
    return str(v)


def write_card(out_dir: Path, meta: dict, df, frame_path: Path):
    # One row per person, aggregating categorical states but retaining an observed sample for numeric values.
    rows = []
    for uid, u in df.groupby("user_id", sort=True):
        r = u.iloc[len(u)//2]
        def value(col): return _fmt(r[col]) if col in u.columns else "—"
        def cats(col): return ", ".join(sorted(set(u[col].dropna().astype(str)))) if col in u.columns else "—"
        position = "(" + ", ".join(value(c) for c in ("x", "y", "z")) + ")"
        row = [uid, cats("group_id"), position, value("height_agl_m"), cats("height_class"), value("slope_deg"), value("speed_mps"),
               cats("bs_link"), cats("ris_los"), cats("blocker_id"), cats("blocker_height_source"),
               value("bs_distance_m"), value("ris_distance_m"), value("bs_az_deg"), value("bs_el_deg"),
               value("ris_az_deg"), value("ris_el_deg"), cats("served")]
        rows.append(row)
    cols = ["user_id", "group_id", "position XYZ (m)", "height AGL (m)", "height class", "DTM slope (deg; null=elevated/drone)", "speed (m/s)", "BS link (no_los = no direct LOS)", "RIS LOS",
            "blocker ID", "blocker height source", "BS distance (m)", "RIS distance (m)", "BS az (deg)",
            "BS el (deg)", "RIS az (deg)", "RIS el (deg)", "served frames"]
    md = [f"# Case {meta.get('id', out_dir.name)}", "", f"- Map: `{meta.get('map_id','?')}`; kind: **{meta.get('kind','?')}**; challenge: `{meta.get('scenario')}`",
          f"- Duration: {_fmt(meta.get('duration_s'))} s; frame interval: {_fmt(meta.get('dt_s'))} s; crowd: {meta.get('crowd_size', df.user_id.nunique())}",
          f"- BS XYZ: `{meta.get('bs_xyz','?')}`; RIS XYZ: `{meta.get('ris_xyz','?')}`; BS-RIS distance {_fmt(meta.get('distance_bs_ris_m'))} m (clear segment)",
          f"- RIS LOS (first frame): {meta.get('ris_los_users_first_frame','?')} of {meta.get('crowd_size','?')} users; served users with RIS LOS: {meta.get('served_ris_los_first_frame','?')} of {meta.get('served_count','?')}. Walkable slope cap {_fmt(meta.get('walk_slope_cap_deg'))} deg.",
          "- Link names: `no_los` = no clear direct segment (reflections are not traced in this preview); `no_los_reflection_candidate` = a bounded one-bounce geometry candidate exists, not a traced path.", "",
          "![Rotatable-source snapshot (static rendering)](map-3d.png)", "", f"All frame labels: [`{frame_path.name}`]({frame_path.name})", "",
          "The figure is a static PNG. The plotted footprints are extruded for readability; native source surfaces are used for the geometry checks.", "",
          "## Per-user observed labels", "", "|"+"|".join(cols)+"|", "|"+"|".join(["---"]*len(cols))+"|"]
    for row in rows:
        md.append("|"+"|".join(str(x).replace("|", "\\|") for x in row)+"|")
    (out_dir / "README.md").write_text("\n".join(md)+"\n")


def render_distributions(root: Path, out: Path, case_dfs: list, draws_path: Path | None):
    import pandas as pd
    parts = [d for d in case_dfs if len(d)]
    frames = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    draws = read_table(draws_path) if draws_path else pd.DataFrame()
    # Include common compact column aliases while clearly labelling the source.
    fig, axs = plt.subplots(2, 4, figsize=(17, 9), constrained_layout=True)
    specs = [("group_size", "Group size (per cluster)", True), ("crowd_size", "Crowd size (per case)", True),
             ("bs_distance_m", "BS distance (m)", True), ("ris_distance_m", "RIS distance (m)", True),
             ("height_agl_m", "Height above ground (m)", True), ("speed_mps", "Speed (m/s)", True),
             ("slope_deg", "DTM slope (deg), pedestrians only", False),
             ("bs_ris_distance_m", "BS-RIS distance per case (m)", False)]
    for ax, (col, label, logx) in zip(axs.flat, specs):
        sources = []
        if col in draws:
            if "distribution" in draws:
                for name, sub in draws.groupby("distribution"):
                    if col in sub: sources.append((str(name), sub[col].to_numpy(dtype=float)))
            else:
                sources.append(("larger draw", draws[col].to_numpy(dtype=float)))
        if col in frames:
            if "_case_kind" in frames:
                for name, sub in frames.groupby("_case_kind"):
                    sources.append((f"preview {name}", sub[col].to_numpy(dtype=float)))
            else:
                sources.append(("preview frames", frames[col].to_numpy(dtype=float)))
        prepared=[]
        for name, raw in sources:
            vals=np.asarray(raw,dtype=float); vals=vals[np.isfinite(vals)]
            zero_share=float(np.mean(vals<=.001)) if len(vals) else 0.
            if logx:
                vals=vals[vals>(.001 if col=="speed_mps" else 0.)]
            if len(vals):
                prepared.append((name,vals,zero_share))
        if prepared:
            lo=min(v.min() for _,v,_ in prepared)
            hi=max(v.max() for _,v,_ in prepared)
            if hi<=lo:
                lo,hi=lo*.99,hi*1.01
                if lo==hi: lo,hi=-.01,.01
            bins=np.geomspace(lo,hi,42) if logx else np.linspace(lo,hi,42)
            for name,vals,zero_share in prepared:
                extra=f"; stopped {zero_share:.0%}" if col=="speed_mps" else ""
                ax.hist(vals,bins=bins,weights=np.full(len(vals),1./len(vals)),
                        histtype="step",linewidth=1.6,label=f"{name} (n={len(vals)}{extra})",
                        log=logx)
            if logx: ax.set_xscale("log")
            ax.legend(fontsize=7)
        else:
            ax.text(.5,.5,f"No `{col}` data",ha="center",va="center",transform=ax.transAxes)
        if col == "slope_deg":
            ax.axvline(15., color="k", linestyle=":", linewidth=1); ax.set_yscale("log")
        if col == "bs_ris_distance_m":
            ax.axvline(20., color="k", linestyle=":", linewidth=1); ax.axvline(150., color="k", linestyle=":", linewidth=1)
        ax.set_xlabel(label); ax.set_ylabel("Share per bin"+(" (log)" if logx or col == "slope_deg" else "")); ax.grid(alpha=.2)
    fig.suptitle("Preview distributions — frame labels and larger unsaved draw (v1)", fontsize=15)
    fig.savefig(out / "distributions.png", dpi=175); plt.close(fig)
    # Link classes are categorical; counts across frames and a split out RIS LOS summary.
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    if "bs_link" in frames:
        frames = frames.assign(_link=frames.bs_link.map(_link))
        first = frames[frames["frame"] == 0]
        cat = [k for k in LINK_COLORS if k != "unknown"]
        names = list(first["_case_kind"].unique())
        panels = [(k, first[first["_case_kind"] == k]) for k in names]
        if "bs_link" in draws:
            panels.append(("bulk natural", draws.assign(_link=draws.bs_link.map(_link))))
        width = .8/max(1,len(panels))
        for k,(label,part) in enumerate(panels):
            shares = part._link.value_counts(normalize=True).reindex(cat,fill_value=0)
            axs[0].bar(np.arange(len(cat))-.4+width*(k+.5),shares,width=width,label=f"{label} (n={len(part)})")
            if "ris_los" in part:
                ris = part.ris_los.astype(bool)
                vals = [float(ris.mean()), float(ris[part.served.astype(bool)].mean()) if "served" in part and part.served.astype(bool).any() else 0.]
                bars = axs[1].bar(np.arange(2)-.4+width*(k+.5),vals,width=width,label=label)
                for b_,v_ in zip(bars,vals):
                    axs[1].text(b_.get_x()+b_.get_width()/2,v_+.01,f"{v_:.0%}",ha="center",fontsize=8)
        axs[0].set_xticks(range(len(cat)),cat,rotation=23,ha="right")
        axs[0].set_title("BS geometry class at initial frame (no_los = no direct LOS)"); axs[0].legend(fontsize=8)
        axs[1].set_xticks(range(2),["all users","served users"])
        axs[1].set_ylim(0,1.08)
        axs[1].set_title("Share of users with RIS LOS (geometry only)"); axs[1].legend(fontsize=8)
        for ax in axs: ax.set_ylabel("User share")
    else:
        axs[0].text(.5,.5,"No BS labels",ha="center",transform=axs[0].transAxes)
        axs[1].text(.5,.5,"No RIS labels",ha="center",transform=axs[1].transAxes)
    fig.savefig(out / "link-classes.png", dpi=175); plt.close(fig)
    return draws


def render_strips(out: Path, metas: list[dict], dfs: list, limit: int = 5):
    candidates = []
    for meta, df in zip(metas, dfs):
        if not {"user_id", "frame", "x", "y", "bs_link"}.issubset(df.columns):
            continue
        for uid, u in df.groupby("user_id"):
            states = u.sort_values("frame").bs_link.map(_link).tolist()
            changes = sum((a == "los") != (b == "los") for a, b in zip(states, states[1:]))
            if changes and len(u) > 2:
                slope = float(u.slope_deg.abs().median()) if "slope_deg" in u else 0
                slope = 0. if not np.isfinite(slope) else slope
                candidates.append((slope, changes, meta, uid, u.sort_values("frame")))
    candidates.sort(key=lambda x: (
        x[2].get("scenario") == "shadow_edge_sloped" and x[3] == x[2].get("shadow_user_id"),
        str(x[2].get("scenario","")).startswith("shadow") and x[3] == x[2].get("shadow_user_id"),
        x[0] > 1, x[1]), reverse=True)
    # Prefer one transition strip per independent case before using spare slots.
    unique, seen = [], set()
    for item in candidates:
        case_id = str(item[2].get("id", ""))
        if case_id not in seen:
            unique.append(item); seen.add(case_id)
        if len(unique) >= limit: break
    result = []
    for n, (slope, changes, meta, uid, u) in enumerate(unique[:limit], 1):
        has_ground = {"ground_z_m", "time_s"}.issubset(u.columns)
        if has_ground:
            fig, (ax, profile) = plt.subplots(2, 1, figsize=(9, 4.7), constrained_layout=True)
        else:
            fig, ax = plt.subplots(figsize=(9, 2.5), constrained_layout=True)
        x, y = u.x.to_numpy(), u.y.to_numpy()
        ax.plot(x, y, color="#34465d", linewidth=2, zorder=1)
        for state, sub in u.assign(_link=u.bs_link.map(_link)).groupby("_link"):
            ax.scatter(sub.x, sub.y, c=LINK_COLORS.get(state, "#7a8492"), s=30, label=state, zorder=2)
        ax.set_aspect("equal", adjustable="datalim"); ax.grid(alpha=.2); ax.legend(ncol=3, fontsize=8)
        ax.set(xlabel="Projected X (m)", ylabel="Projected Y (m)",
               title=f"{meta.get('id')} user {uid}: link transitions={changes}, median |slope|={slope:.1f}°")
        if has_ground:
            z0 = u.ground_z_m.iloc[0]
            profile.plot(u.time_s, u.ground_z_m-z0, color="#34465d", linewidth=1.5)
            for state, sub in u.assign(_link=u.bs_link.map(_link)).groupby("_link"):
                profile.scatter(sub.time_s, sub.ground_z_m-z0, s=8,
                                c=LINK_COLORS.get(state, "#7a8492"))
            profile.set(xlabel="Time (s)", ylabel="Ground change (m)",
                        title="Native terrain under the moving user")
            profile.grid(alpha=.25)
        name = f"movement-strip-{n:02d}.png"; fig.savefig(out / name, dpi=170); plt.close(fig)
        result.append((name, meta.get("id"), uid, changes, slope))
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True, help="bench-users-v1 root containing cases/ and geometry/")
    ap.add_argument("--draws", type=Path, help="optional compact CSV/parquet from the larger unsaved draw")
    ap.add_argument("--strip-count", type=int, default=5)
    args = ap.parse_args()
    root = args.root.resolve(); out = root / "figures"; out.mkdir(parents=True, exist_ok=True)
    case_dirs = sorted(p.parent for p in (root / "cases").glob("*/case.json"))
    if not case_dirs:
        raise SystemExit(f"No cases found under {root / 'cases'}")
    metas, dfs, cards, errors = [], [], [], []
    for case_dir in case_dirs:
        try:
            meta, df, frame_path = render_case(root, case_dir, case_dir)
            write_card(case_dir, meta, df, frame_path)
            df = df.copy(); df["_case_kind"] = meta.get("kind", "unknown"); df["_map_id"] = meta.get("map_id", "unknown")
            df["crowd_size"] = np.nan
            df["group_size"] = np.nan
            df["bs_ris_distance_m"] = np.nan
            initial = df[df["frame"] == df["frame"].min()]
            if len(initial):
                df.loc[initial.index[0], "bs_ris_distance_m"] = meta.get("distance_bs_ris_m")
                df.loc[initial.index[0], "crowd_size"] = len(initial)
                for _, group in initial.groupby("group_id"):
                    df.loc[group.index[0], "group_size"] = len(group)
            metas.append(meta); dfs.append(df); cards.append(case_dir)
        except Exception as exc:
            errors.append({"case": case_dir.name, "error": f"{type(exc).__name__}: {exc}"})
    if not dfs:
        raise SystemExit(f"No cases rendered successfully: {errors[:3]}")
    render_distributions(root, out, dfs, args.draws)
    strips = render_strips(out, metas, dfs, args.strip_count)
    lines = ["# Bench users v1 preview index", "", f"Rendered {len(dfs)} / {len(case_dirs)} cases. Figures are static PNGs; each case README links to the exact all-frame table.", "",
             "## Aggregate figures", "", "- [Distributions](figures/distributions.png)", "- [Link-class counts](figures/link-classes.png)", "", "## Movement strips", ""]
    lines += [f"- [{name}](figures/{name}) — case `{case}`, user `{uid}`, {changes} BS-link transitions; median |slope| {slope:.2f}°" for name, case, uid, changes, slope in strips]
    lines += ["", "## Cases", "", "|Case|Map|Kind|Crowd|Card|Map figure|", "|---|---|---|---:|---|---|"]
    for meta, d in zip(metas, cards):
        lines.append(f"|{meta.get('id',d.name)}|{meta.get('map_id','?')}|{meta.get('kind','?')}|{meta.get('crowd_size','?')}|[README](cases/{d.name}/README.md)|[3D map](cases/{d.name}/map-3d.png)|")
    if errors:
        lines += ["", "## Render errors", ""] + [f"- `{e['case']}`: {e['error']}" for e in errors]
    lines += ["", "Street data: [© OpenStreetMap contributors](https://www.openstreetmap.org/copyright). "
              "Building and terrain credits appear on the individual maps.", ""]
    (root / "index.md").write_text("\n".join(lines)+"\n")
    (root / "render_manifest.json").write_text(json.dumps({"cases_found":len(case_dirs),"rendered":len(dfs),"errors":errors,"movement_strips":strips,"draws":str(args.draws) if args.draws else None},indent=2)+"\n")
    print(f"Rendered {len(dfs)}/{len(case_dirs)} cases; index: {root / 'index.md'}")


if __name__ == "__main__":
    main()
