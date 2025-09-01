# ----------------------------------------------------------------------------------------
# Written by Qi Wu, Department of Biomedicine, Aarhus University, with the aid of ChatGPT
# Date: 01/September/2025
# Contact Qi Wu (qi.wu@biomed.au.dk) for information and comments
# ----------------------------------------------------------------------------------------

import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker  # Ensure proper axis formatting
import seaborn as sns  # Optional: Improves plot aesthetic
import os  # For handling file paths
import scipy.sparse as sp
from matplotlib.collections import PathCollection

# Set random seed for reproducibility
np.random.seed(42)

# Load your data
csv_file_path = r"C:\Users\au520563\OneDrive - Aarhus universitet\DATA\20241126\20250106_140344_DCT_SingleCell_758cells_Report.csv"
data = pd.read_csv(csv_file_path)

# 📌 **Get the folder path automatically from the CSV file**
data_folder = os.path.dirname(csv_file_path)  # Extract the directory

print(f"📂 Data folder detected: {data_folder}")
# 📌 **Load Marker Genes from Excel File**
marker_genes_file = r"C:\Users\au520563\OneDrive - Aarhus universitet\DATA\marker_genes_DCT.xlsx"
marker_df = pd.read_excel(marker_genes_file)

# Convert marker genes from a comma-separated string to a list
marker_genes = {
    row[0]: row[1].split(",") for row in marker_df.itertuples(index=False)
}

# Convert gene names to uppercase
marker_genes = {cell_type: [gene.strip().lower().capitalize() for gene in genes] for cell_type, genes in marker_genes.items()}

# Separate metadata and expression data
metadata = data.iloc[:, :3]  # First three columns as metadata
expression_data = data.iloc[:, 3:]  # Remaining columns are expression values

# Ensure all values are float32 (to avoid integer overflow)
expression_data = expression_data.fillna(0).astype(np.float32)

# Transpose if necessary (e.g., proteins as rows, cells as columns)
expression_data = expression_data.T

# Convert to AnnData format
adata = sc.AnnData(expression_data)

# Ensure gene names are properly extracted from metadata['PG.Genes']
if 'PG.Genes' in metadata.columns:
    genes = metadata['PG.Genes'].astype(str).str.split(";").str[0].str.strip().fillna("Unknown")
else:
    print("⚠️ Column 'PG.Genes' not found in metadata. Check column names.")
    genes = pd.Series(["Unknown"] * adata.shape[1])  # Fallback

# Assign gene names to adata.var
adata.var['gene_name'] = genes[:adata.n_vars].values

# Ensure proper format of gene names
adata.var_names = adata.var["gene_name"].astype(str).str.lower().str.capitalize()

# Fix duplicate gene names by making them unique
if adata.var_names.duplicated().any():
    print("⚠️ Duplicate gene names found. Making them unique...")
    adata.var_names = pd.Index([f"{name}_{i}" if name in adata.var_names[:i] else name 
                                for i, name in enumerate(adata.var_names)])

# Filter out lowly expressed cells/proteins
sc.pp.filter_cells(adata, min_genes=100)
sc.pp.filter_genes(adata, min_cells=2)

adata.raw = adata

# Log-normalize the data
sc.pp.log1p(adata)

# Scale data
sc.pp.scale(adata, max_value=10)

#Ensure sc.pp.neighbors() is Run Before Leiden Clustering
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=20)

# Use only Leiden resolution X
leiden_res = 3
leiden_key = f"leiden_res_{leiden_res:.2f}"
sc.tl.leiden(adata, key_added=leiden_key, resolution=leiden_res, flavor="igraph")

# Find overlapping marker genes
matching_marker_genes = {
    cell_type: [gene for gene in genes if gene in adata.var_names]
    for cell_type, genes in marker_genes.items()
}

# Remove cell types with no matching genes
matching_marker_genes = {cell_type: genes for cell_type, genes in matching_marker_genes.items() if genes}

if matching_marker_genes:
    # Compute cell scores using sc.tl.score_genes()
    for cell_type, genes in matching_marker_genes.items():
        sc.tl.score_genes(adata, gene_list=genes, score_name=f"{cell_type}_score")

    # Create a DataFrame of scores
    scores_df = pd.DataFrame({cell_type: adata.obs[f"{cell_type}_score"] for cell_type in matching_marker_genes.keys()})

    # Assign the highest scoring cell type to each cell
    adata.obs["predicted_cell_type"] = scores_df.idxmax(axis=1)

    # Compute max marker score per cell
    adata.obs["max_score"] = scores_df.max(axis=1)

    # Compute mean score per cluster
    cluster_scores = adata.obs.groupby(leiden_key, observed=True)["max_score"].mean()
    cluster_diversity = adata.obs.groupby(leiden_key, observed=True)["predicted_cell_type"].nunique()

    # Set refined threshold for "Unknown" clusters
    score_threshold = cluster_scores.quantile(0.01)  # Bottom 1%
    diversity_threshold = cluster_diversity.quantile(1)  # Most mixed clusters

    unknown_clusters = cluster_scores[cluster_scores < score_threshold].index.union(
        cluster_diversity[cluster_diversity > diversity_threshold].index
    )

    # Assign most common cell type per cluster, keeping "Unknown" for ambiguous clusters
    cluster_to_celltype = (
        adata.obs.groupby(leiden_key)["predicted_cell_type"]
        .agg(lambda x: x.mode()[0] if not x.mode().empty else "Unknown")
        .to_dict()
    )

    # Initialize all clusters as their most common cell type
    adata.obs[f"{leiden_key}_named"] = adata.obs[leiden_key].map(cluster_to_celltype)

    # Ensure "Unknown" clusters are correctly labeled and not overwritten
    adata.obs.loc[adata.obs[leiden_key].isin(unknown_clusters), f"{leiden_key}_named"] = "Unknown"

    # Convert to categorical format
    adata.obs[f"{leiden_key}_named"] = adata.obs[f"{leiden_key}_named"].astype("category")

    # Debug: Print final cluster names
    print(f"\nFinal Cluster Names for Leiden {leiden_res:.2f}:")
    print(adata.obs[f"{leiden_key}_named"].value_counts())

# Now perform PCA and UMAP AFTER cell types are assigned
sc.tl.pca(adata, svd_solver='arpack')
sc.tl.umap(adata)

# ---------------------------------------------------------------------------------------
# UMAP for a user list of selected genes — one figure per gene
# ---------------------------------------------------------------------------------------
genes_to_plot = ['Slc12a3', 'Mki67', 'Pvalb']   # <-- edit this list

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
        s=16,
        linewidth=0,
        alpha=0.9
    )
    # axis labels & ticks @12
    ax.set_xlabel('UMAP1', fontsize=12)
    ax.set_ylabel('UMAP2', fontsize=12)
    ax.tick_params(axis='both', labelsize=12)

    # gene title @12 and **bold**
    ax.set_title(gene, fontsize=12, fontweight='bold')

    cbar = plt.colorbar(pts, ax=ax, pad=0.01)
    cbar.set_label('Expression abundance', rotation=90, labelpad=12, fontsize=12)
    cbar.ax.tick_params(labelsize=12)

    plt.tight_layout()
    out = os.path.join(data_folder, f"UMAP_expr_{gene}.png")
    plt.savefig(out, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Saved UMAP for {gene}: {out}")

# -----------------------------------------
# Trajectory Analysis using Diffusion Pseudotime (DPT)
# -----------------------------------------
# Set "DCT1" as the root cell type for trajectory analysis.
# Here we choose the first cell with predicted cell type "DCT1".
prolif_indices = np.where(adata.obs["predicted_cell_type"] == "DCT1")[0]
if len(prolif_indices) > 0:
    adata.uns['iroot'] = int(prolif_indices[0])
    print(f"Setting root cell as 'DCT1' with index: {adata.uns['iroot']}")
else:
    print("No cell with predicted cell type 'DCT1' found. Using default root cell index 0.")
    adata.uns['iroot'] = 0

# Compute diffusion pseudotime
sc.tl.dpt(adata)

# Plot UMAP with diffusion pseudotime coloring
# output path
dpt_fig_path = os.path.join(data_folder, "UMAP_DPT_Pseudotime.png")

# extract coords & pseudotime
coords = adata.obsm['X_umap']
pseudotime = adata.obs['dpt_pseudotime'].values

# set up figure
fig, ax = plt.subplots(figsize=(5, 4), dpi=600)

# scatter the cells, colored by pseudotime
scat = ax.scatter(
    coords[:, 0], coords[:, 1],
    c=pseudotime, cmap='viridis',
    s=30, alpha=0.8, linewidth=0, rasterized=True,
)

# axes labels + ticks @12
ax.set_xlabel('UMAP1', fontsize=12)
ax.set_ylabel('UMAP2', fontsize=12)
ax.tick_params(axis='both', labelsize=12)

# no title
ax.set_title('')

# colorbar with label & tick size @12
cbar = fig.colorbar(scat, ax=ax, pad=0.01)
cbar.set_label('Diffusion Pseudotime', rotation=90, labelpad=15, fontsize=12)
cbar.ax.tick_params(labelsize=12)

# tidy up and save
plt.tight_layout()
plt.savefig(dpt_fig_path, dpi=600, bbox_inches='tight')
plt.close(fig)

print(f"✅ Diffusion Pseudotime UMAP saved to: {dpt_fig_path}")

# ----------------------------------------------------------------------------
# PAGA / graph-based trajectory inference + edge scores
# ----------------------------------------------------------------------------
# 1) Run PAGA on cell-type groups
sc.tl.paga(adata, groups='predicted_cell_type')

# 2) Extract connectivity matrix (group × group) and group order
groups = adata.obs['predicted_cell_type'].cat.categories.tolist()
conn_mat = adata.uns['paga']['connectivities']
conn = conn_mat.toarray() if sp.issparse(conn_mat) else np.asarray(conn_mat)

# 3) Build an edge table (upper triangle, no self-edges)
src_idx, tgt_idx = np.triu_indices(len(groups), k=1)
edges = []
for i, j in zip(src_idx, tgt_idx):
    w = float(conn[i, j])
    edges.append({"source": groups[i], "target": groups[j], "connectivity": w})

edge_df = pd.DataFrame(edges)
edge_df = edge_df[edge_df["connectivity"] > 0].sort_values("connectivity", ascending=False)

# 4) Save edge scores to CSV
paga_scores_csv = os.path.join(data_folder, f"Leiden_{leiden_res}_PAGA_connectivities.csv")
edge_df.to_csv(paga_scores_csv, index=False)
print(f"✅ PAGA edge scores saved to: {paga_scores_csv}")

# 5) Report the 3 connections of interest, if present
def _lookup(a, b):
    if a in groups and b in groups:
        ia, ib = groups.index(a), groups.index(b)
        return float(conn[min(ia, ib), max(ia, ib)])
    return np.nan

for a, b in [("DCT1", "DCT2"), ("DCT1", "ProLIF"), ("DCT2", "ProLIF")]:
    val = _lookup(a, b)
    if not np.isnan(val):
        print(f"   • Connectivity {a}–{b}: {val:.3f}")

# 6) Plot the abstract PAGA graph and annotate edges with weights
paga_fig = os.path.join(data_folder, f"Leiden_{leiden_res}_PAGA.png")
plt.figure(figsize=(6, 6))
ax = sc.pl.paga(
    adata,
    show=False,
    frameon=False
)

# Enlarge nodes a bit
node_colls = [c for c in ax.collections if isinstance(c, PathCollection)]
if node_colls:
    node_coll = node_colls[0]
    node_coll.set_sizes(node_coll.get_sizes() * 3)

# Nudge labels outward and bump font
cx, cy = np.mean(ax.get_xlim()), np.mean(ax.get_ylim())
for txt in ax.texts:
    txt.set_fontsize(15)
    x, y = txt.get_position()
    dx, dy = x - cx, y - cy
    d = np.hypot(dx, dy)
    if d > 0:
        txt.set_position((x + 0.10 * dx / d, y + 0.10 * dy / d))

# If PAGA stored node positions, use them to place edge weights
pos = adata.uns.get('paga', {}).get('pos', None)
if pos is not None and len(pos) == len(groups):
    for i, j in zip(src_idx, tgt_idx):
        w = float(conn[i, j])
        if w <= 0:
            continue
        x_mid = 0.5 * (pos[i, 0] + pos[j, 0])
        y_mid = 0.5 * (pos[i, 1] + pos[j, 1])
        ax.text(
            x_mid, y_mid, f"{w:.2f}",
            ha="center", va="center",
            fontsize=18, color="black",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
            zorder=5
        )

plt.tight_layout()
plt.savefig(paga_fig, dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ PAGA graph saved to: {paga_fig}")

# ----------------------------------------------------------------------------
# Pseudotime distribution by cell type (boxplot) + median overlays
# ----------------------------------------------------------------------------
cell_types   = adata.obs['predicted_cell_type']
pseudotimes  = adata.obs['dpt_pseudotime']

# Keep plotting order consistent with the categorical order
order = cell_types.cat.categories.tolist()

dpt_group_fig = os.path.join(data_folder, f"predicted_cell_type_pseudotime_boxplot.png")
plt.figure(figsize=(8, 6), dpi=600)

# draw box + individual points (use the same order for both)
sns.boxplot(x=cell_types, y=pseudotimes, palette='Set2', order=order)
sns.stripplot(x=cell_types, y=pseudotimes, color='black', size=3, alpha=0.4, jitter=True, order=order)

# labels & ticks @12
ax = plt.gca()
ax.set_xlabel('', fontsize=15)  # keep empty label, but set size consistently
ax.set_ylabel('Diffusion Pseudotime', fontsize=15)
ax.tick_params(axis='both', labelsize=15)
plt.xticks(rotation=45, ha='right', fontsize=15)

plt.tight_layout()
plt.savefig(dpt_group_fig, dpi=600, bbox_inches='tight')
plt.close()
print(f"✅ Pseudotime distribution boxplot saved to: {dpt_group_fig}")

# -----------------------------------------------------------------------------
# Gene Expression vs. Diffusion Pseudotime (robust version)
# -----------------------------------------------------------------------------
if "dpt_pseudotime" not in adata.obs:
    print("⚠️  No DPT results found; skipping expression‑vs‑pseudotime plots.")
else:
    # Make sure scanpy is still the scanpy module
    import scanpy as sc  
    
    genes_pt = ['Slc12a3','Mki67','Top2a','Pcna']  # or any list, e.g. ['Slc12a3','Mki67']
    display_names = {
        "Mki67": "Ki-67",
        # add more if you have other special cases...
    }
    
    present = [g for g in genes_pt if g in adata.var_names]
    missing = [g for g in genes_pt if g not in adata.var_names]
    if missing:
        print(f"⚠️  Skipping {missing}; not in adata.var_names")
    dpt = adata.obs["dpt_pseudotime"].values

    for gene in present:
        disp = display_names.get(gene, gene)
        expr = adata[:, gene].X
        expr = expr.toarray().flatten() if hasattr(expr, "toarray") else expr.flatten()

        plt.figure(figsize=(5, 4), dpi=600)
        pts = plt.scatter(
            dpt, expr,
            c=dpt, cmap="viridis", s=20, edgecolors="none"
        )
        cbar = plt.colorbar(pts)
        cbar.set_label("Diffusion Pseudotime", fontsize=12)
        cbar.ax.tick_params(labelsize=12)

        plt.xlabel("Diffusion pseudotime", fontsize=12)
        plt.ylabel(f"{disp} expression", fontsize=12)
        plt.tick_params(axis='both', labelsize=12)

        out = os.path.join(data_folder, f"{gene}_expr_vs_pseudotime.png")
        plt.tight_layout()
        plt.savefig(out, dpi=600, bbox_inches="tight")
        plt.close()

        print(f"✅ Saved {disp} expression‑vs‑pseudotime: {out}")

# 📌 **Extract UMAP coordinates manually**
umap_df = pd.DataFrame(adata.obsm["X_umap"], columns=["UMAP1", "UMAP2"])
umap_df["Cell Type"] = adata.obs[f"{leiden_key}_named"].values  # Assign predicted cell types

# This section creates a file with each cell's idenpngier, its UMAP coordinates, and its predicted cell type.
umap_celltype_df = pd.DataFrame(adata.obsm["X_umap"], columns=["UMAP1", "UMAP2"], index=adata.obs_names)
umap_celltype_df["Cell ID"] = umap_celltype_df.index
umap_celltype_df["Predicted Cell Type"] = adata.obs[f"{leiden_key}_named"].values
umap_celltype_df = umap_celltype_df.reset_index(drop=False)  # Optionally keep the original index as a column
detailed_umap_csv_path = os.path.join(data_folder, "Detailed_UMAP_CellType_Info.csv")
umap_celltype_df.to_csv(detailed_umap_csv_path, index=False)
print(f"✅ Detailed UMAP cell type information saved to: {detailed_umap_csv_path}")

# Build a palette that is identical to seaborn's default ordering,
# except map 'Unknown' explicitly to grey.
cats = list(pd.Series(umap_df["Cell Type"]).astype(str).unique())
base_colors = sns.color_palette(None, n_colors=len(cats))
palette_map = {}
i = 0
for c in cats:
    if isinstance(c, str) and c.lower() == 'unknown':
        palette_map[c] = (0.7, 0.7, 0.7)  # light grey for Unknown
    else:
        palette_map[c] = base_colors[i % len(base_colors)]
        i += 1

# 📌 **Save UMAP Plot**
umap_fig_path = os.path.join(data_folder, f"Leiden_{leiden_res}_UMAP_plot.png")

fig, ax = plt.subplots(figsize=(10, 8))

sns.scatterplot(
    x=umap_df["UMAP1"], y=umap_df["UMAP2"], hue=umap_df["Cell Type"],   
    palette=palette_map, 
    edgecolor="black", alpha=0.8, s=50, ax=ax
)

# axes labels + ticks @12; keep bold axis labels
ax.set_xlabel("UMAP1", fontsize=12, fontweight="bold", labelpad=10)
ax.set_ylabel("UMAP2", fontsize=12, fontweight="bold", labelpad=10)
ax.tick_params(axis="both", labelsize=12)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# legend outside, fontsize 12, title bold "Cell Type"
handles, labels = ax.get_legend_handles_labels()
leg = ax.legend(
    handles, labels,
    loc="center left", bbox_to_anchor=(1, 0.5),
    fontsize=12, title="Cell Type", frameon=False
)
leg.get_title().set_fontsize(12)
leg.get_title().set_fontweight("bold")  # ← make "Cell Type" bold

plt.subplots_adjust(right=0.75, bottom=0.15)
plt.savefig(umap_fig_path, dpi=600, bbox_inches="tight")
plt.close()

print(f"✅ UMAP figure saved to: {umap_fig_path}")

#------------Marker Dotplot Section-------------
# 📌 **Calculate Dynamic Figure Height**
num_markers = sum(len(genes) for genes in matching_marker_genes.values())
fig_height = min(max(num_markers * 0.15, 4), 10)  # Ensure proper height without too much blank space

# 1. Draw into the current figure, but don’t show it yet
sc.pl.dotplot(
    adata,
    matching_marker_genes,
    groupby="predicted_cell_type",
    figsize=(8, fig_height),
    color_map="Reds",
    dot_max=0.8, dot_min=0.05,
    standard_scale="var",
    dendrogram=False,
    var_group_rotation=0,
    show=False
)

# 2. Grab that figure
fig = plt.gcf()

# 3. Rotate the bottom x-tick labels (your genes)
for ax in fig.axes:
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

# 4. Save and close
dotplot_fig_path = os.path.join(data_folder, f"Leiden_{leiden_res}_Dotplot.png")
fig.savefig(dotplot_fig_path, dpi=600, bbox_inches="tight")
plt.close(fig)

print(f"✅ Dotplot figure saved to: {dotplot_fig_path}")

# ===== Stacked violin: Top 10 DE genes across predicted cell types =====
# Grouping to show on the stacked violin
groupby_key_sv = 'predicted_cell_type'

# Run DE for these groups without overwriting earlier DE results
de_key_sv = 'rank_genes_predicted_celltype'
if (de_key_sv not in adata.uns) or (
    adata.uns.get(de_key_sv, {}).get('params', {}).get('groupby') != groupby_key_sv
):
    sc.tl.rank_genes_groups(adata, groupby=groupby_key_sv, method='wilcoxon', key_added=de_key_sv)

# Gather all DE results; pick overall top 10 unique genes (highest score/logFC)
de_df = sc.get.rank_genes_groups_df(adata, group=None, key=de_key_sv)
sort_col = 'scores' if 'scores' in de_df.columns else (
    'score' if 'score' in de_df.columns else (
    'logfoldchanges' if 'logfoldchanges' in de_df.columns else None))
if sort_col:
    de_df = de_df.sort_values(by=sort_col, ascending=False)

_use_raw = getattr(adata, 'raw', None) is not None
valid_names = adata.raw.var_names if _use_raw else adata.var_names

top_genes = []
for g in de_df['names'].tolist():
    if g in valid_names and g not in top_genes:
        top_genes.append(g)
    if len(top_genes) == 10:
        break

if not top_genes:
    print("⚠️ No valid DE genes found for stacked violin.")
else:
    # Size scales with number of groups/genes
    n_groups = adata.obs[groupby_key_sv].nunique()
    fig_w = max(8, 0.8 * len(top_genes))
    fig_h = max(4.5, 0.45 * n_groups)

    sc.pl.stacked_violin(
        adata,
        var_names=top_genes,
        groupby=groupby_key_sv,
        use_raw=_use_raw,
        standard_scale='var',
        swap_axes=False,           # genes on X, cell types on Y (as in your figure)
        figsize=(fig_w, fig_h),
        show=False,
    )

    # Rotate the X-axis gene labels for readability (no swap_axes)
    fig = plt.gcf()
    for ax in fig.axes:
        labs = ax.get_xticklabels()
        if labs and any(lbl.get_text() for lbl in labs):
            plt.setp(labs, rotation=45, ha="right")

    stacked_out = os.path.join(
        data_folder, f"StackedViolin_top10_{groupby_key_sv}_Leiden_{leiden_res}.png"
    )
    plt.tight_layout()
    plt.savefig(stacked_out, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"✅ Stacked violin (top 10 across {groupby_key_sv}) saved to: {stacked_out}")
# ======================================================================