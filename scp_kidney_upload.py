# ----------------------------------------------------------------------------------------
# Written by Qi Wu, Department of Biomedicine, Aarhus University, with the aid of ChatGPT
# Date: 01/September/2025
# Contact: qi.wu@biomed.au.dk
# ----------------------------------------------------------------------------------------

import os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.sparse as sp
from matplotlib.collections import PathCollection
from matplotlib import colors as mcolors

np.random.seed(42)

# ----------------------------
# I/O paths (from your template)
# ----------------------------
csv_file_path = r"C:\Users\au520563\OneDrive - Aarhus universitet\DATA\20250203\20250201_180815_Kidney_singleCell_768cells_Report.csv"
marker_genes_file = r"C:\Users\au520563\OneDrive - Aarhus universitet\DATA\marker_genes_short.xlsx"

data = pd.read_csv(csv_file_path)
data_folder = os.path.dirname(csv_file_path)
print(f"📂 Data folder detected: {data_folder}")

# ----------------------------
# Marker list (Excel: cell_type, comma-separated genes)
# ----------------------------
marker_df = pd.read_excel(marker_genes_file)
marker_genes = {row[0]: [g.strip().lower().capitalize() for g in str(row[1]).split(",")] for row in marker_df.itertuples(index=False)}
# read abbreviations from 3rd column
abbr_map = {}
if marker_df.shape[1] >= 3:
    names = marker_df.iloc[:, 0].astype(str).str.strip()
    abbrs = marker_df.iloc[:, 2].astype(str).str.strip()
    abbr_map = dict(zip(names, abbrs))

# ----------------------------
# Build AnnData
# ----------------------------
metadata = data.iloc[:, :3]
X = data.iloc[:, 3:].fillna(0).astype(np.float32).T
adata = sc.AnnData(X)

if 'PG.Genes' in metadata.columns:
    genes = metadata['PG.Genes'].astype(str).str.split(';').str[0].str.strip().fillna('Unknown')
else:
    print("⚠️ Column 'PG.Genes' not found; filling as 'Unknown'.")
    genes = pd.Series(['Unknown'] * adata.n_vars)

adata.var['gene_name'] = genes[:adata.n_vars].values
adata.var_names = adata.var['gene_name'].astype(str).str.lower().str.capitalize()

if adata.var_names.duplicated().any():
    print("⚠️ Duplicate gene names found. Making them unique…")
    new_names = []
    seen = {}
    for name in adata.var_names:
        if name in seen:
            seen[name] += 1
            new_names.append(f"{name}_{seen[name]}")
        else:
            seen[name] = 0
            new_names.append(name)
    adata.var_names = pd.Index(new_names)

# QC / norm
sc.pp.filter_cells(adata, min_genes=100)
sc.pp.filter_genes(adata, min_cells=2)
adata.raw = adata
sc.pp.log1p(adata)
sc.pp.scale(adata, max_value=10)

# neighbors + Leiden
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=20)
leiden_res = 2
leiden_key = f"leiden_res_{leiden_res:.2f}"
sc.tl.leiden(adata, key_added=leiden_key, resolution=leiden_res, flavor="igraph")

# ----------------------------
# Marker scoring → initial labels
# ----------------------------
matching_marker_genes = {ct: [g for g in gl if g in adata.var_names] for ct, gl in marker_genes.items()}
matching_marker_genes = {ct: gl for ct, gl in matching_marker_genes.items() if gl}

if matching_marker_genes:
    for ct, gl in matching_marker_genes.items():
        sc.tl.score_genes(adata, gene_list=gl, score_name=f"{ct}_score")
    scores_df = pd.DataFrame({ct: adata.obs[f"{ct}_score"] for ct in matching_marker_genes})
    adata.obs['predicted_cell_type'] = scores_df.idxmax(axis=1)
    adata.obs['max_score'] = scores_df.max(axis=1)

    clust_mean = adata.obs.groupby(leiden_key, observed=True)['max_score'].mean()
    clust_div = adata.obs.groupby(leiden_key, observed=True)['predicted_cell_type'].nunique()

    score_threshold = clust_mean.quantile(0.01)  # bottom 1%
    diversity_threshold = clust_div.quantile(1)  # most mixed

    unknown_clusters = clust_mean[clust_mean < score_threshold].index.union(
        clust_div[clust_div > diversity_threshold].index
    )

    cluster_to_celltype = (
        adata.obs.groupby(leiden_key)['predicted_cell_type']
        .agg(lambda x: x.mode()[0] if not x.mode().empty else 'Unknown')
        .to_dict()
    )
    adata.obs[f"{leiden_key}_named"] = adata.obs[leiden_key].map(cluster_to_celltype)
    adata.obs.loc[adata.obs[leiden_key].isin(unknown_clusters), f"{leiden_key}_named"] = 'Unknown'
    adata.obs[f"{leiden_key}_named"] = adata.obs[f"{leiden_key}_named"].astype('category')
    print(f"\nFinal Cluster Names for Leiden {leiden_res:.2f}:")
    print(adata.obs[f"{leiden_key}_named"].value_counts())

# ----------------------------
# Re-annotate: generalized penalized scoring for ALL cell types with shared markers
# ----------------------------
KEEP_UNKNOWN = True

marker_pos = {ct: [g for g in gl if g in adata.var_names]
              for ct, gl in marker_genes.items()}
marker_pos = {ct: gl for ct, gl in marker_pos.items() if gl}

from collections import defaultdict
gene_to_cts = defaultdict(set)
for ct, gl in marker_genes.items():
    for g in gl:
        gene_to_cts[g].add(ct)

marker_neg = {}
for ct, gl in marker_genes.items():
    competitors = set()
    for g in gl:
        competitors |= (gene_to_cts[g] - {ct})
    neg = set()
    for other in competitors:
        neg |= set(marker_genes.get(other, []))
    neg -= set(gl)
    marker_neg[ct] = [g for g in neg if g in adata.var_names]

alpha = 0.7
scores = {}
for ct in marker_genes.keys():
    pos_list = marker_pos.get(ct, [])
    neg_list = marker_neg.get(ct, [])
    if pos_list:
        sc.tl.score_genes(adata, gene_list=pos_list, score_name=f"_pos_{ct}")
        pos = adata.obs[f"_pos_{ct}"].values
    else:
        pos = np.zeros(adata.n_obs, dtype=float)
    if neg_list:
        sc.tl.score_genes(adata, gene_list=neg_list, score_name=f"_neg_{ct}")
        neg = adata.obs[f"_neg_{ct}"].values
    else:
        neg = np.zeros(adata.n_obs, dtype=float)
    scores[ct] = pos - alpha * neg

extra_df = pd.DataFrame(scores, index=adata.obs_names)
best_new = extra_df.idxmax(axis=1)
best_val = extra_df.max(axis=1)
second   = extra_df.apply(lambda r: np.partition(r.values, -2)[-2] if r.size > 1 else 0, axis=1)
margin   = best_val - second

import numpy as _np
import scipy.sparse as _sp

base_labels = adata.obs[f"{leiden_key}_named"].astype(str).copy()

if 'connectivities' not in adata.obsp:
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=20)
A_ = adata.obsp['connectivities']

def _neighbor_agreement(labels, A):
    if isinstance(labels, pd.Series):
        labels = labels.astype(str).values
    if _sp.issparse(A):
        A = A.tocsr()
    agree = 0; total = 0
    indptr, indices = A.indptr, A.indices
    for i in range(A.shape[0]):
        nbrs = indices[indptr[i]:indptr[i+1]]
        if nbrs.size == 0:
            continue
        total += nbrs.size
        agree += (labels[nbrs] == labels[i]).sum()
    return agree / max(total, 1)

ABS_GRID    = _np.round(_np.arange(0.10, 0.90, 0.05), 2)
MARGIN_GRID = _np.round(_np.arange(0.05, 0.50, 0.05), 2)

results = []
for abs_min in ABS_GRID:
    for margin_min in MARGIN_GRID:
        mask = (best_val > abs_min) & (margin > margin_min)
        trial = base_labels.copy()
        trial.loc[mask] = best_new.loc[mask]
        nar = _neighbor_agreement(trial, A_)
        frac_changed = (trial != base_labels).mean()
        conf_strength = best_val[mask].mean() if mask.any() else 0.0
        unknown_frac = (trial.str.lower() == 'unknown').mean()
        UNKNOWN_WEIGHT = 0.0 if KEEP_UNKNOWN else 0.2
        composite = (1.0 * nar) + (0.3 * conf_strength) - (0.5 * frac_changed) - (UNKNOWN_WEIGHT * unknown_frac)
        results.append((composite, abs_min, margin_min, nar, conf_strength, frac_changed, unknown_frac))

results.sort(reverse=True, key=lambda x: x[0])
_best = results[0]
ABS_MIN, MARGIN_MIN = _best[1], _best[2]
print(f"✅ Tuned thresholds: ABS_MIN={ABS_MIN}, MARGIN_MIN={MARGIN_MIN}")
print(f"   NAR={_best[3]:.3f}  mean_conf={_best[4]:.3f}  changed={_best[5]:.3f}  unknown={_best[6]:.3f}")

_res_df = pd.DataFrame(results, columns=['composite','ABS_MIN','MARGIN_MIN','NAR','mean_conf','changed','unknown'])
_tune_csv = os.path.join(data_folder, 'Threshold_tuning_results.csv')
_res_df.to_csv(_tune_csv, index=False)

try:
    import matplotlib.pyplot as _plt
    pivot = _res_df.pivot(index='MARGIN_MIN', columns='ABS_MIN', values='composite')
    _plt.figure(figsize=(8, 5))
    im = _plt.imshow(pivot.values, aspect='auto')
    _plt.colorbar(im, label='Composite score')
    _plt.xticks(range(pivot.shape[1]), [f"{v:.2f}" for v in pivot.columns], rotation=45, ha='right')
    _plt.yticks(range(pivot.shape[0]), [f"{v:.2f}" for v in pivot.index])
    _plt.xlabel('ABS_MIN')
    _plt.ylabel('MARGIN_MIN')
    _plt.title('Threshold tuning grid — composite score')
    _plt.tight_layout()
    _heatmap_path = os.path.join(data_folder, 'Threshold_tuning_heatmap.png')
    _plt.savefig(_heatmap_path, dpi=300, bbox_inches='tight')
    _plt.close()
    print(f"✅ Saved threshold tuning heatmap: {_heatmap_path}")
except Exception as e:
    print(f"(heatmap skipped) {e}")

base = base_labels.copy()
confident = (best_val > ABS_MIN) & (margin > MARGIN_MIN)
base.loc[confident] = best_new.loc[confident]

if (not KEEP_UNKNOWN) and ('connectivities' in adata.obsp):
    A = adata.obsp['connectivities']
    if sp.issparse(A):
        A = A.tocsr()
    unk = np.where(base.str.lower() == 'unknown')[0]
    if len(unk) > 0:
        print(f"🔄 Resolving {len(unk)} 'Unknown' cells by neighbor voting…")
        for i in unk:
            row = A.getrow(i)
            nn = row.indices
            if nn.size == 0:
                continue
            neigh = base.iloc[nn]
            neigh = neigh[neigh.str.lower() != 'unknown']
            if neigh.size > 0:
                base.iloc[i] = neigh.mode().iat[0]

if not KEEP_UNKNOWN:
    base = base.replace({'Unknown': 'Connecting tubule cell'})

adata.obs['cell_type_v2'] = pd.Categorical(base)
if KEEP_UNKNOWN and 'Unknown' in adata.obs['cell_type_v2'].cat.categories:
    _cats = [c for c in adata.obs['cell_type_v2'].cat.categories if c != 'Unknown'] + ['Unknown']
    adata.obs['cell_type_v2'] = adata.obs['cell_type_v2'].cat.reorder_categories(_cats, ordered=False)

print("✅ Re-annotation (generalized shared-marker penalization) complete. Counts:", adata.obs['cell_type_v2'].value_counts())

# ----------------------------
# OPTIONAL: Glasbey palette for many categories (keeps 'Unknown' grey)
# Default OFF to keep your original palette.
# ----------------------------
USE_GLASBEY = False           # <-- set True to use Glasbey
GLASBEY_VARIANT = 'glasbey'   # options if colorcet installed: 'glasbey', 'glasbey_light', 'glasbey_dark'

def _get_glasbey_colors(n: int):
    try:
        import colorcet as cc
        base = list(getattr(cc, GLASBEY_VARIANT))
        if not base:
            raise AttributeError
        cols = [base[i % len(base)] for i in range(n)]      # hex strings
        return [mcolors.to_rgb(c) for c in cols]
    except Exception:
        print("ℹ️ Glasbey palette requires 'colorcet' (pip install colorcet). Falling back to seaborn 'husl'.")
        return sns.color_palette("husl", n)

# ----------------------------
# Palette for MANY elements (original behavior) + optional Glasbey override
# ----------------------------
try:
    _cats_all = list(adata.obs.get('cell_type_v2', adata.obs.get(f"{leiden_key}_named")).cat.categories)
except Exception:
    _cats_all = sorted(pd.unique(adata.obs.get('cell_type_v2', adata.obs.get(f"{leiden_key}_named"))))

def _even_odd(colors):
    colors = list(colors)
    return colors[0::2] + colors[1::2]  # reorder for more contrast

# Original large palette (~60 colors) from Tab20/Tab20b/Tab20c
_tab20  = _even_odd(plt.get_cmap("tab20").colors)
_tab20b = _even_odd(plt.get_cmap("tab20b").colors)
_tab20c = _even_odd(plt.get_cmap("tab20c").colors)
_base_colors_original = list(_tab20) + list(_tab20b) + list(_tab20c)  # ≈60 colors

# Count non-Unknown categories
_needed = sum(1 for c in _cats_all if not (isinstance(c, str) and c.lower() == 'unknown'))

# Choose palette source
if USE_GLASBEY:
    _base_colors = _get_glasbey_colors(_needed)
else:
    _base_colors = _base_colors_original
    # If somehow more than ~60 categories, extend gracefully (keeps original default intent)
    if _needed > len(_base_colors):
        extra = sns.color_palette("husl", _needed - len(_base_colors))
        _base_colors = _base_colors + list(extra)

# Build mapping and sync to Scanpy
palette_map = {}
_i = 0
for c in _cats_all:
    if isinstance(c, str) and c.lower() == 'unknown':
        palette_map[c] = (0.7, 0.7, 0.7)  # light grey for Unknown
    else:
        palette_map[c] = _base_colors[_i]
        _i += 1

_hex_colors = [mcolors.to_hex(palette_map[c]) for c in _cats_all]
adata.uns['cell_type_v2_colors'] = _hex_colors
adata.uns[f'{leiden_key}_named_colors'] = _hex_colors
adata.uns['predicted_cell_type_colors'] = _hex_colors

# ----------------------------
# UMAP
# ----------------------------
sc.tl.pca(adata, svd_solver='arpack')
sc.tl.umap(adata)

umap_df = pd.DataFrame(adata.obsm['X_umap'], columns=['UMAP1', 'UMAP2'])
umap_df['Cell Type'] = adata.obs.get('cell_type_v2', adata.obs[f"{leiden_key}_named"]).values

umap_fig_path = os.path.join(data_folder, f"Leiden_{leiden_res}_UMAP_plot.png")
fig, ax = plt.subplots(figsize=(10, 8))

sns.scatterplot(
    x=umap_df['UMAP1'], y=umap_df['UMAP2'],
    hue=umap_df['Cell Type'],
    palette=palette_map, edgecolor='black', alpha=0.8, s=50, ax=ax
)

# axes labels + ticks
ax.set_xlabel('UMAP1', fontsize=12, fontweight='bold', labelpad=10)
ax.set_ylabel('UMAP2', fontsize=12, fontweight='bold', labelpad=10)
ax.tick_params(axis='both', labelsize=12)

# clean frame
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# legend with bold title and size 12
handles, labels = ax.get_legend_handles_labels()
leg = ax.legend(
    handles, labels,
    loc='center left', bbox_to_anchor=(1, 0.5),
    fontsize=12, title='Cell Type', frameon=False
)
# ensure bold legend title
leg.get_title().set_fontweight('bold')
leg.get_title().set_fontsize(12)

plt.subplots_adjust(right=0.75, bottom=0.15)
plt.savefig(umap_fig_path, dpi=600, bbox_inches='tight')
plt.close()
print(f"✅ UMAP figure saved to: {umap_fig_path}")

# ---------------------------------------------------------------------------------------
# UMAP for a user list of selected genes — one figure per gene
# ---------------------------------------------------------------------------------------
genes_to_plot = ['Slc12a3']   # <-- edit this list

_use_raw = True if getattr(adata, "raw", None) is not None else False
present = [g for g in genes_to_plot if g in (adata.raw.var_names if _use_raw else adata.var_names)]
missing = [g for g in genes_to_plot if g not in (adata.raw.var_names if _use_raw else adata.var_names)]
if missing:
    print(f"⚠️ Skipping missing genes: {missing}")

# Pre-fetch coordinates once
_umap = adata.obsm['X_umap']

for gene in present:
    # fetch expression vector
    if _use_raw:
        vec = adata.raw[:, gene].X
    else:
        vec = adata[:, gene].X
    vec = vec.toarray().ravel() if sp.issparse(vec) else np.asarray(vec).ravel()

    fig, ax = plt.subplots(figsize=(6, 5), dpi=600)
    pts = ax.scatter(
        _umap[:, 0], _umap[:, 1],
        c=vec,
        cmap='Reds',
        s=20,
        linewidth=0,
        alpha=0.9
    )
    ax.set_xlabel('UMAP1', fontsize=10)
    ax.set_ylabel('UMAP2', fontsize=10)
    ax.set_title(gene, fontsize=11)

    cbar = plt.colorbar(pts, ax=ax, pad=0.01)
    cbar.set_label('Expression abundance', rotation=90, labelpad=12, fontsize=9)

    plt.tight_layout()
    out = os.path.join(data_folder, f"UMAP_expr_{gene}.png")
    plt.savefig(out, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Saved UMAP for {gene}: {out}")

# ----------------------------
# Differential expression → small DE UMAP panel
# ----------------------------
sc.pp.filter_genes(adata, min_cells=1)
if 'rank_genes_groups' not in adata.uns:
    print("⚠️ Running differential expression analysis (rank_genes_groups)…")
    sc.tl.rank_genes_groups(adata, groupby=leiden_key, method='wilcoxon')

cell_type_of_interest = 'Distal convoluted tubule cell'
cell_to_cluster_map = adata.obs.groupby(leiden_key)[f"{leiden_key}_named"].apply(lambda x: x.mode()[0]).to_dict()

def _plot_de_umap_for(ct_name: str):
    if ct_name not in cell_to_cluster_map.values():
        print(f"⚠️ No cluster found for cell type '{ct_name}'. Skipping DE analysis.")
        return
    cluster_id = str([k for k, v in cell_to_cluster_map.items() if v == ct_name][0])
    df = sc.get.rank_genes_groups_df(adata, group=cluster_id).head(11)
    valid = [g for g in df['names'] if g in adata.var_names]
    if not valid:
        print(f"⚠️ No valid DE genes for '{ct_name}'.")
        return
    out = os.path.join(data_folder, f"UMAP_DE_{ct_name.replace(' ', '_')}_Leiden_{leiden_res}.png")
    sc.pl.umap(adata, color=valid + [leiden_key], legend_loc='on data', ncols=3, show=False)
    plt.savefig(out, dpi=600, bbox_inches='tight'); plt.close()
    print(f"✅ DE UMAP figure saved to: {out}")

_plot_de_umap_for(cell_type_of_interest)

# ----- Stacked violin: Top 10 DE genes across current re-annotated cell types -----
groupby_key_sv = 'cell_type_v2' if 'cell_type_v2' in adata.obs.columns else 'predicted_cell_type'

# Run DE for the chosen grouping without overwriting earlier DE results
de_key = 'rank_genes_celltype_v2'
if de_key not in adata.uns or adata.uns[de_key].get('params', {}).get('groupby') != groupby_key_sv:
    sc.tl.rank_genes_groups(adata, groupby=groupby_key_sv, method='wilcoxon', key_added=de_key)

# Collect all DE results and pick the overall top 10 unique genes (highest score/logFC)
de_df = sc.get.rank_genes_groups_df(adata, group=None, key=de_key)
sort_col = 'scores' if 'scores' in de_df.columns else (
    'score' if 'score' in de_df.columns else (
    'logfoldchanges' if 'logfoldchanges' in de_df.columns else None))
if sort_col:
    de_df = de_df.sort_values(by=sort_col, ascending=False)

# Keep first 10 unique genes that are present in the data
_use_raw = getattr(adata, 'raw', None) is not None
valid_names = (adata.raw.var_names if _use_raw else adata.var_names)
top_genes = []
for g in de_df['names'].tolist():
    if g in valid_names and g not in top_genes:
        top_genes.append(g)
    if len(top_genes) == 10:
        break

if not top_genes:
    print("⚠️ No valid DE genes found for stacked violin.")
else:
    # Size scales with number of groups and genes; swap_axes=True to match your example
    n_groups = len(adata.obs[groupby_key_sv].cat.categories) if pd.api.types.is_categorical_dtype(adata.obs[groupby_key_sv]) else adata.obs[groupby_key_sv].nunique()
    fig_w = max(8, 0.8 * len(top_genes))
    fig_h = max(4, 0.5 * n_groups)

    # --- build a view that excludes 'Unknown' for plotting ---
    mask_sv = adata.obs[groupby_key_sv].astype(str).str.lower() != 'unknown'
    adata_sv = adata[mask_sv].copy()

    # Size scales with number of groups and genes; no swap_axes
    n_groups = adata_sv.obs[groupby_key_sv].nunique()
    fig_w = max(8, 0.8 * len(top_genes))
    fig_h = max(4, 0.5 * n_groups)

    sc.pl.stacked_violin(
        adata_sv,
        var_names=top_genes,
        groupby=groupby_key_sv,
        use_raw=_use_raw,
        standard_scale='var',
        swap_axes=False,
        figsize=(fig_w, fig_h),
        show=False,
    )

    # rotate gene labels
    fig = plt.gcf()
    for ax in fig.axes:
        labs = ax.get_xticklabels()
        if labs and any(lbl.get_text() for lbl in labs):
            plt.setp(labs, rotation=45, ha="right")

    stacked_out = os.path.join(data_folder, f"StackedViolin_top10_{groupby_key_sv}_Leiden_{leiden_res}.png")
    plt.tight_layout()
    plt.savefig(stacked_out, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"✅ Stacked violin (top 10 across {groupby_key_sv}, excl. Unknown) saved to: {stacked_out}")


# ----------------------------
# Dot plot with forced marker display (SAFE: no adata shape mutation)
# ----------------------------
panel_order = []
for ct, gl in marker_genes.items():
    for g in gl:
        gn = g.strip().lower().capitalize()
        if gn not in panel_order:
            panel_order.append(gn)

missing = [g for g in panel_order if g not in adata.var_names]

# Build a *temporary* AnnData for plotting
if sp.issparse(adata.X):
    zeros = sp.csr_matrix((adata.n_obs, len(missing)), dtype=adata.X.dtype) if missing else None
    X_ext = sp.hstack([adata.X, zeros], format='csr') if missing else adata.X.copy()
else:
    zeros = np.zeros((adata.n_obs, len(missing)), dtype=adata.X.dtype) if missing else None
    X_ext = np.hstack([adata.X, zeros]) if missing else adata.X.copy()

var_ext = adata.var.copy()
if missing:
    add_var = pd.DataFrame(index=missing)
    add_var['forced_missing'] = True
    var_ext = pd.concat([var_ext, add_var], axis=0)
var_ext.index = var_ext.index.astype(str)

data_dp = sc.AnnData(X_ext, obs=adata.obs.copy(), var=var_ext)

plot_markers = {ct: [g.strip().lower().capitalize() for g in gl] for ct, gl in marker_genes.items()}
num_markers = sum(len(gl) for gl in plot_markers.values())
fig_height = min(max(num_markers * 0.15, 4), 10)

_groupby_key = 'cell_type_v2' if 'cell_type_v2' in data_dp.obs.columns else 'predicted_cell_type'

# --- Use abbreviations as the dict KEYS so Scanpy shows them on the upper x-axis ---
ct_order = list(plot_markers.keys())
abbr_order = [abbr_map.get(ct, ct) for ct in ct_order]

# ensure unique group names if abbreviations collide
seen = {}
abbr_unique = []
for a in abbr_order:
    if a in seen:
        seen[a] += 1
        abbr_unique.append(f"{a}_{seen[a]}")
    else:
        seen[a] = 0
        abbr_unique.append(a)

plot_markers_abbr = {a: plot_markers[ct] for a, ct in zip(abbr_unique, ct_order)}

# --- make a plotting view that excludes 'Unknown' groups ---
mask_dp = data_dp.obs[_groupby_key].astype(str).str.lower() != 'unknown'
data_dp_plot = data_dp[mask_dp, :].copy()

sc.pl.dotplot(
    data_dp_plot,
    plot_markers_abbr,   # ← abbreviations appear as upper x-axis headers
    groupby=_groupby_key,
    use_raw=False,
    figsize=(16, fig_height),
    color_map='Reds',
    dot_max=0.8,
    dot_min=0.0,
    standard_scale='var',
    dendrogram=False,
    show=False,
)

# make sure we have a handle to the current figure (needed below)
fig = plt.gcf()

# Make abbreviated cell-type headers smaller
fig = plt.gcf()
targets = set(abbr_unique)   # from the dict you built for abbreviations

# de-emphasize forced-missing markers
forced = set(missing)
for ax in fig.axes:
    for lab in ax.get_xticklabels():
        gname = lab.get_text()
        if gname in forced:
            lab.set_color('0.45')
            lab.set_alpha(0.9)
            lab.set_fontstyle('italic')

# keep bottom gene tick labels readable (unchanged)
for ax in fig.axes:
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

# save
dotplot_fig_path = os.path.join(data_folder, f"Leiden_{leiden_res}_Dotplot.png")
fig.savefig(dotplot_fig_path, dpi=600, bbox_inches='tight')
plt.close(fig)
print(f"✅ Dotplot figure saved to: {dotplot_fig_path}")
