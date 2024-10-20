import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import os

def plot_yy(csv_path,save_dir):
    # CSVデータの読み込み
    data = pd.read_csv(csv_path)

    # 物性値のリストを取得
    properties = list(set([col.replace('_target', '').replace('_pred', '') for col in data.columns if '_target' in col or '_pred' in col]))

    # yyプロットの作成
    for prop in properties:
        target_col = f'{prop}_target'
        pred_col = f'{prop}_pred'

        target = data[target_col]
        pred = data[pred_col]

        rmse = np.sqrt(mean_squared_error(target, pred))
        r2 = r2_score(target, pred)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(target, pred, alpha=0.5)
        ax.plot([target.min(), target.max()], [target.min(), target.max()], 'k--')

        ax.set_xlabel(f'True {prop}')
        ax.set_ylabel(f'Predicted {prop}')

        ax.text(0.05, 0.95, f'RMSE: {rmse:.4f}\nR2: {r2:.4f}', transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.5))

        plt.title(f'yy plot for {prop}')

        save_fig_path = os.path.join(save_dir,f'yy_plot_{prop}.png')
        plt.savefig(save_fig_path)
        # plt.show()


if __name__ == '__main__':
    import sys
    csv_path = sys.argv[1]
    save_dir = sys.argv[2]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plot_yy(csv_path, save_dir)