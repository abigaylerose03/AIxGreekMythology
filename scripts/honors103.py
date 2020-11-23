"""
@author Abigayle Peterson
@date 11/21/20

"""

import pandas as pd
import json

# greek = pd.read_json ('')

with open('/Users/abigaylepeterson/Desktop/all.json') as f:
  data = json.load(f)

df = pd.DataFrame.from_dict(pd.json_normalize(data), orient='columns')
print(df.columns)
