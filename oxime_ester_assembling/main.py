from oxime_ester_fragment import OximeEsterFragment

### TASK オキシムエステルを側鎖、アルキル鎖に分解してデータベースに保存する ###

# TODO 何かしらの方法でオキシムエステルの全データを取得
import pandas as pd

df = pd.read_csv("dataset.csv")
print(df)
# オキシムエステルのリスト
all_oxime_ester = df["smiles"].to_list

for smi in all_oxime_ester:
    # オブジェクト作成
    oxime_ester_frag = OximeEsterFragment(smi)

    # アルキル鎖取得
    alkyl_chaines = oxime_ester_frag.alkylChain()

    # 骨格構造取得
    scaffold = oxime_ester_frag.getScaffold()

    # ベース構造取得
    base_struct = oxime_ester_frag.getOximeEsterBaseStruct()

    # TODO 側鎖は無条件でデータベースに保存

    # TODO scaffold smilesに対して対称性を調べる

    # TODO 対称性ある場合はデータベースに追加する
    
