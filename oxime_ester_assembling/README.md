# oxime_ester_assembling

This code is about assebly method

# OverView

The molecules in the dataset were split into side chains and the main skeleton by the MurckoScaffold function in RDKit. The split side chains and main skeleton were shuffled. The Gasteiger charges38 of the aromatic atoms of the main frame containing aromatic rings were calculated, and the molecule was designed by introducing side chains containing oxime esters at the positions of aromatic atoms with low Gasteiger charges. This resulted in a molecular structure containing one to three oxime ester active sites.

# dataset

Oxime Ester datasets are prepared in `data/oxime_ester/oxime.smi`

# running code

```

python python main.py --trg_path ./data/oxime_ester/oxime.smi --filename oxime --base_smarts "COC(=O)N=C(\*)\*" --method brics

```
