# import json

# with open('videodatainfo_2017.json') as f:
#     captions = json.load(f)
#     info = captions['info']
#     print(info)

import os

flg = os.path.exists('data/save/model_score.txt')
print(flg)