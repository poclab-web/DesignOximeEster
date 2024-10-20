import json
import sys, os
sys.path.append(os.path.dirname(__file__))

# load setting file
json_file = open('./setting.json', 'r')
global setting
setting = json.load(json_file)

if __name__ == '__main__':
    print(setting['KFOLD_NUM'])