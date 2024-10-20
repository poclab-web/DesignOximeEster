import json

def saveJson(json_data, file_path):
    with open(file_path, 'w') as f:
        json.dump(json_data, f, indent=4, ensure_ascii=False)

values = ["\n", "#", "&", "(", ")", "-", "1", "2", "3", "4", "=", "C", "N", "O", "c", "n", "o", "S", "s"]
dictionary = {i: v for i, v in enumerate(values)}

save_path = "/mnt/ssd2/Chem/photopolymerization_initiator/ML/MolGen/data/tokenizers/myChromoCharTokenizer.json"
saveJson(dictionary, save_path)