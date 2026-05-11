from DenoiseST import DenoiseST
import os
import torch
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt

from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    adjusted_mutual_info_score,
    fowlkes_mallows_score,
    homogeneity_score,
    completeness_score
)

from utils import clustering

# ========================
# 路径
# ========================
data_dir = "/work/data1/GUOMENGKE/CuiYaxuan/SpatialAD/SCtoSTRNA/cyclevaegeneraterna/"
output_dir = os.path.join(data_dir, "results")
os.makedirs(output_dir, exist_ok=True)

# ========================
# 参数
# ========================
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
n_clusters = 7
radius = 50
tool = 'mclust'

# ========================
# 结果容器
# ========================
results = []

# ========================
# 遍历所有 h5ad 文件
# ========================
for file in os.listdir(data_dir):

    if not file.endswith("_sp_to_sc.h5ad"):
        continue

    file_path = os.path.join(data_dir, file)
    sample_name = file.replace(".h5ad", "")

    print(f"\nProcessing: {sample_name}")

    try:
        # ========================
        # 读取数据
        # ========================
        adata = sc.read_h5ad(file_path)

        # ========================
        # 跳过大样本
        # ========================
        n_spots = adata.n_obs
        if n_spots > 50000:
            print(f"Skip {sample_name} (spots={n_spots} > 50000)")
            continue

        print(f"Spots: {n_spots}")

        # ========================
        # 模型训练
        # ========================
        model = DenoiseST(adata, device=device, n_top_genes=500)
        adata = model.train()

        # ========================
        # 聚类
        # ========================
        if tool == 'mclust':
            clustering(adata, n_clusters, radius=radius, method=tool, refinement=True)
        elif tool in ['leiden', 'louvain']:
            clustering(
                adata,
                n_clusters,
                radius=radius,
                method=tool,
                start=0.1,
                end=2.0,
                increment=0.01,
                refinement=False
            )

        # ========================
        # 保存空间图
        # ========================
        pdf_path = os.path.join(output_dir, f"{sample_name}_RNAdomain.pdf")

        sc.pl.spatial(
            adata,
            color="domain",
            spot_size=150,
            title="domain",
            save= f"{sample_name}_RNAdomain.pdf"
        )

        # ========================
        # 计算指标
        # ========================
        if 'Layer annotation' not in adata.obs:
            print(f"Warning: {sample_name} missing 'Layer annotation'")
            continue

        gt = adata.obs['Layer annotation']
        pred = adata.obs['domain']

        mask = gt.notna()
        gt_valid = gt[mask].astype(str)
        pred_valid = pred[mask].astype(str)

        ari = adjusted_rand_score(gt_valid, pred_valid)
        nmi = normalized_mutual_info_score(gt_valid, pred_valid)
        ami = adjusted_mutual_info_score(gt_valid, pred_valid)
        fmi = fowlkes_mallows_score(gt_valid, pred_valid)
        homo = homogeneity_score(gt_valid, pred_valid)
        como = completeness_score(gt_valid, pred_valid)

        print(f"ARI={ari:.4f}, NMI={nmi:.4f}")

        # ========================
        # 保存结果
        # ========================
        results.append({
            "sample": sample_name,
            "spots": n_spots,
            "ARI": ari,
            "NMI": nmi,
            "AMI": ami,
            "FMI": fmi,
            "HOMO": homo,
            "COMO": como
        })

    except Exception as e:
        print(f"Error processing {sample_name}: {e}")

# ========================
# 保存 CSV
# ========================
if len(results) > 0:
    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, "RNA_metrics_summary.csv")
    df.to_csv(csv_path, index=False)

    print("\nAll done!")
    print(f"Results saved to: {csv_path}")
else:
    print("\nNo valid results.")