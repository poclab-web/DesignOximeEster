import polars as pl
import numpy as np

def log_transform(df: pl.DataFrame, column_name: str) -> pl.DataFrame:
    return df.with_columns([
        (pl.col(column_name) + 1e-10).log().alias(f"log_{column_name}")
    ])

def reset_oxim_id(df: pl.DataFrame) -> pl.DataFrame:
    # 'oxim_id'カラムをユニークにし、小さい順にソート
    unique_oxim_ids = df['oxim_id'].unique().sort()
    
    # ユニークな値にインデックスをマッピング
    oxim_id_mapping = {oxim_id: index for index, oxim_id in enumerate(unique_oxim_ids)}
    
    # 'oxim_id'カラムの値を新しいインデックスで置き換え
    new_oxim_id = df['oxim_id'].map_dict(oxim_id_mapping).alias('oxim_id')
    
    # 不要なカラムを削除し、新しいカラムを追加
    return df.drop('oxim_id').with_columns([new_oxim_id])

def remove_duplicate_rows(df: pl.DataFrame, column_names: list) -> pl.DataFrame:
    return df.unique(subset=column_names)

def remove_rows_with_null(df: pl.DataFrame, column_names: list) -> pl.DataFrame:
    condition = None
    for column_name in column_names:
        if condition is None:
            condition = df[column_name].is_not_null()
        else:
            condition = condition & df[column_name].is_not_null()
    return df.filter(condition)

if __name__ == '__main__':
    import sys
    csv_path = sys.argv[1]
    df = pl.read_csv(csv_path)

    df = reset_oxim_id(df)

    columns = ['smiles','BDE','T1','S1','S2','S3','S1_osc','S2_osc','S3_osc']

    columns_target = ['S1_osc','S2_osc','S3_osc']

    for column in columns_target:
        df = log_transform(df, column)

    # # Remove duplicate rows
    # df = remove_duplicate_rows(df, columns)
    # df = remove_rows_with_null(df, columns)

    df.write_csv(csv_path)
