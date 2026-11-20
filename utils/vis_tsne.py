import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def tsne_visualization(features, labels, save_path=None):
    print("正在执行t-SNE降维...")
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features)
    fig = plt.figure(figsize=(10, 8), dpi=300)
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap="viridis", alpha=0.8)
    cbar = plt.colorbar(scatter)
    cbar.set_label("Label Value", fontsize=24)
    cbar.ax.tick_params(labelsize=22)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"t-SNE 可视化已保存到: {save_path}")
    return fig
