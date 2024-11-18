import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def df_to_png(csv_path, filename):
    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(10, 2))
    ax.axis("tight")
    ax.axis("off")
    table_data = []
    columns = list(df.columns)
    for row in df.itertuples():
        table_data.append(row[1:])
    table = ax.table(cellText=table_data, colLabels=columns, loc="center")
    plt.savefig(filename, dpi=300)


if __name__ == "__main__":
    import sys
    csv_path = sys.argv[1]
    save_img_path = sys.argv[2]
    df_to_png(csv_path, save_img_path)
