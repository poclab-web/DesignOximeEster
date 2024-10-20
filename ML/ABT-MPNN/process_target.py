import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import ast
import sys


def apply_transforms(csv_file, transforms, save_dir):
    df = pd.read_csv(csv_file)

    for col, func in transforms.items():
        if col in df.columns:
            # transformed_col = f"{col}_transformed_{func.__name__}"
            transformed_col = f"{col}_transformed"
            df[transformed_col] = df[col].apply(func)
    # print(df.head())
    for col in df.columns:
        plt.figure()
        try:
            df[col].plot(kind='hist', bins=30, title=f'Distribution of {col}')
            save_path = os.path.join(save_dir, f'{col}_hist.png')
            plt.savefig(save_path)
            plt.show()
        except TypeError:
            pass
    return df

def parse_transforms(transform_strings):
    transforms = {}
    for transform_string in transform_strings:
        col, func_str = transform_string.split(':')
        func = eval('lambda x: ' + func_str)
        func.__name__ = func_str.replace(' ', '')  # Remove spaces for column naming
        transforms[col] = func
    return transforms

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Apply linear transforms to columns of a CSV file and save as new columns.')
    parser.add_argument('csv_file', type=str, help='Path to the CSV file')
    parser.add_argument('--save_dir', type=str, help='Path to the CSV file')
    parser.add_argument('--transform', action='append', help='Transformations in the format column:transform, e.g., column1:x*2+3')

    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    transforms = parse_transforms(args.transform)
    transformed_df = apply_transforms(args.csv_file, transforms, args.save_dir)

    save_command_path = os.path.join(args.save_dir, 'command.txt')
    with open(save_command_path, 'w') as f:
        f.write(' '.join(sys.argv))

    # Save the transformed dataframe if needed
    save_path = os.path.join(args.save_dir, os.path.basename(args.csv_file))
    transformed_df.to_csv(save_path, index=False)
