# Experiments involving STAC api.

import json 
import numpy as np
from pathlib import Path 

# Must be installed via pip; conda version outdated.
from satsearch import Search 

# Specify geometry
geom = {
    "type": "Polygon",
    "coordinates": [
      [
        [
          -66.3958740234375,
          43.305193797650546
        ],
        [
          -64.390869140625,
          43.305193797650546
        ],
        [
          -64.390869140625,
          44.22945656830167
        ],
        [
          -66.3958740234375,
          44.22945656830167
        ],
        [
          -66.3958740234375,
          43.305193797650546
        ]
      ]
    ]
}

# Search for all landsat-8-l1 data intersecting with that geometry
search = Search.search(intersects = geom, collection='landsat-8-l1')

# Collect urls/items having to do with that data
items = search.items()

# print summary
print(items.summary())

# Get just one item and download all associated assets (TIFs for bands B0-B11):
# Each item with all bands is roughly 1/2 a gb?
items = search.items(limit = 1)
print(items.summary())
items.download_assets(path=str(Path.cwd()) + '/${date}')