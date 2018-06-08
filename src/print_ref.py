import json
import sys
import os

data_file = os.path.join(sys.argv[1], 'desc.json')

with open(data_file, 'r') as fh:
    data = json.load(fh)

for entry in data['test']:
    print("%s (%s)" % (entry['trans'], entry['id']))
