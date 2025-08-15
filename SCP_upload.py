# ----------------------------------------------------------------------------------------
# Written by Qi Wu, Department of Biomedicine, Aarhus University, with the aid of ChatGPT
# Date: 15/August/2025
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
fig, ax = plt.subplots(figsize=(5,4), dpi=600)

# scatter the cells, colored by pseudotime
scat = ax.scatter(
    coords[:,0], coords[:,1],
    c=pseudotime,
    cmap='viridis',
    s=30,          # dot size; bump this up if you want larger dots
    alpha=0.8,
    linewidth=0,
    rasterized=True,
)

# axes labels
ax.set_xlabel('UMAP1', fontsize=10)
ax.set_ylabel('UMAP2', fontsize=10)

# no title
ax.set_title('')

# optional: remove ticks
# ax.set_xticks([]); ax.set_yticks([])

# add your own colorbar on the right
cbar = fig.colorbar(scat, ax=ax, pad=0.01)
cbar.set_label('Diffusion Pseudotime', rotation=90, labelpad=15, fontsize=10)

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
    txt.set_fontsize(12)
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
# Pseudotime distribution by cell type (boxplot)
# ----------------------------------------------------------------------------
cell_types   = adata.obs['predicted_cell_type']
pseudotimes  = adata.obs['dpt_pseudotime']

dpt_group_fig = os.path.join(data_folder, f"predicted_cell_type_pseudotime_boxplot.png")
plt.figure(figsize=(8, 6))

# draw box + individual points
sns.boxplot(x=cell_types, y=pseudotimes, palette='Set2')
sns.stripplot(x=cell_types, y=pseudotimes,
              color='black', size=3, alpha=0.4, jitter=True)

# remove the default x-axis label entirely
plt.gca().set_xlabel('')

# keep only the y-axis label
plt.ylabel('Diffusion Pseudotime', fontsize=12)

# rotate tick labels, disable grid
plt.xticks(rotation=45, ha='right')
plt.gca().grid(False)

plt.tight_layout()
plt.savefig(dpt_group_fig, dpi=300, bbox_inches='tight')
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

        plt.figure(figsize=(5,4))
        pts = plt.scatter(
            dpt, expr,
            c=dpt, cmap="viridis",
            s=20, edgecolors="none"
        )
        plt.colorbar(pts, label="Diffusion Pseudotime")
        plt.xlabel("Diffusion pseudotime")
        plt.ylabel(f"{disp} expression")
#       plt.title(f"{disp} vs. pseudotime")

        out = os.path.join(data_folder, f"{gene}_expr_vs_pseudotime.png")
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

# 📌 **Save UMAP Plot**
umap_fig_path = os.path.join(data_folder, f"Leiden_{leiden_res}_UMAP_plot.png")

fig, ax = plt.subplots(figsize=(10, 8))

sns.scatterplot(
    x=umap_df["UMAP1"], y=umap_df["UMAP2"], hue=umap_df["Cell Type"],   
#    palette="tab20", 
    edgecolor="black", alpha=0.8, s=50, ax=ax
)

ax.set_xlabel("UMAP1", fontsize=10, fontweight="bold", labelpad=10)
ax.set_ylabel("UMAP2", fontsize=10, fontweight="bold", labelpad=10)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Move legend outside
handles, labels = ax.get_legend_handles_labels()
legend = ax.legend(handles, labels, loc="center left", bbox_to_anchor=(1, 0.5), fontsize=10, title="Cell Type", frameon=False)

# Adjust layout and save figure
plt.subplots_adjust(right=0.75, bottom=0.15)
plt.savefig(umap_fig_path, dpi=600, bbox_inches="tight")  # Save at high resolution
plt.close()  # Close figure to free memory

print(f"✅ UMAP figure saved to: {umap_fig_path}")
        
# 📌 **Ensure Differential Expression Analysis Runs Before Extracting Genes**
sc.pp.filter_genes(adata, min_cells=1)  # Avoids log2 errors from zero-expressed genes

if "rank_genes_groups" not in adata.uns:
    print("⚠️ Running differential expression analysis (rank_genes_groups)...")
    sc.tl.rank_genes_groups(adata, groupby=leiden_key, method="wilcoxon")

# 📌 **Choose a Cell Type and Find Its DE Genes**
cell_type_of_interest = "ProLIF"  # Change this as needed
cell_to_cluster_map = adata.obs.groupby(leiden_key)[f"{leiden_key}_named"].apply(lambda x: x.mode()[0]).to_dict()

if cell_type_of_interest not in cell_to_cluster_map.values():
    print(f"⚠️ No cluster found for cell type '{cell_type_of_interest}'. Skipping DE analysis.")
else:
    cluster_id = str([k for k, v in cell_to_cluster_map.items() if v == cell_type_of_interest][0])  # Get cluster ID

    # 📌 **Get Top X DE Genes**
    dc_cluster_genes = sc.get.rank_genes_groups_df(adata, group=cluster_id).head(11)["names"].tolist()
    
    # Ensure valid genes
    valid_genes = [gene for gene in dc_cluster_genes if gene in adata.var_names]

    if valid_genes:
        # 📌 **Save DE UMAP Plot**
        umap_de_fig_path = os.path.join(data_folder, f"UMAP_DE_{cell_type_of_interest.replace(' ', '_')}_Leiden_{leiden_res}.png")
        sc.pl.umap(adata, color=valid_genes + [leiden_key], legend_loc="on data",ncols=3, show=False)
        plt.savefig(umap_de_fig_path, dpi=600, bbox_inches="tight")
        plt.close()

        print(f"✅ DE UMAP figure saved to: {umap_de_fig_path}")

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
    show=False
)

# 2. Grab that figure
fig = plt.gcf()

# 3a. Rotate the bottom x-tick labels (your genes)
for ax in fig.axes:
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

# 3b. Rotate the top group-name labels by finding them among the figure’s text objects
group_labels = adata.obs["predicted_cell_type"].unique().tolist()
for txt in fig.texts:
    if txt.get_text() in group_labels:
        txt.set_rotation(45)
        txt.set_ha("right")

# 4. Save and close
dotplot_fig_path = os.path.join(data_folder, f"Leiden_{leiden_res}_Dotplot.png")
fig.savefig(dotplot_fig_path, dpi=600, bbox_inches="tight")
plt.close(fig)

print(f"✅ Dotplot figure saved to: {dotplot_fig_path}")

# ----------------------------------------------------------------------------
# In-silico knockout of selected genes
# ----------------------------------------------------------------------------
# 1) Define your knock-out genes
knockout_genes = ['Mki67', 'Slc12a3', 'Ppp1r1a', 'Ptms', 'Tpd52', 'Vcp', 'Arpp19', 'Itgb1', 'Capzb', 'Gdi2', 'Lima1', 'Cdh16', 'Anxa6']  # raw var_name(s)

for g in knockout_genes:
    # 2) Copy and zero out
    adata_ko = adata.copy()
    if g in adata_ko.var_names:
        X = adata_ko[:, g].X
        X[...] = 0
        adata_ko[:, g].X = X
    else:
        print(f"⚠️  Gene {g} not found in adata.var_names; skipping")
        continue

    # 3) Recompute neighbors, diffusion map, and DPT (rooted in DCT1)
    sc.pp.neighbors(adata_ko, n_pcs=50, use_rep='X')
    sc.tl.diffmap(adata_ko)
    root_idx = np.flatnonzero(adata_ko.obs['predicted_cell_type']=='DCT1')[0]
    adata_ko.uns['iroot'] = root_idx
    sc.tl.dpt(adata_ko, n_dcs=10)

    # 4) Gather pseudotimes
    pt_orig = adata.obs['dpt_pseudotime']
    pt_ko   = adata_ko.obs['dpt_pseudotime']

    # 5) Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False, dpi=300)
    sns.boxplot(x=adata.obs['predicted_cell_type'], y=pt_orig,
                ax=axes[0], palette='Set2')
    axes[0].set_title('Original Pseudotime')
    axes[0].set_xlabel(''); axes[0].set_ylabel('Diffusion Pseudotime')

    sns.boxplot(x=adata_ko.obs['predicted_cell_type'], y=pt_ko,
                ax=axes[1], palette='Set2')
    axes[1].set_title(f'{g} Knock-Out Pseudotime')  # gene name in axis title
    axes[1].set_xlabel(''); axes[1].set_ylabel('Diffusion Pseudotime')

    # 6) Save with gene in filename
    out = os.path.join(data_folder, f'pseudotime_KO_comparison_{g}.png')
    plt.tight_layout(rect=[0, 0, 1, 0.92])  # leave room for suptitle
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"✅ In silico KO comparison for {g} saved to: {out}")

