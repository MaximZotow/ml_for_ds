#!/bin/bash

mkdir -p csvdata

# Wholesale customers (plain CSV)
curl -fsSL -o "csvdata/Wholesale customers data.csv" \
  "https://archive.ics.uci.edu/ml/machine-learning-databases/00292/Wholesale%20customers%20data.csv"

# Mice Protein Expression (zip containing Data_Cortex_Nuclear.xls)
curl -fsSL -o mice.zip "https://archive.ics.uci.edu/static/public/342/mice+protein+expression.zip"
unzip -o mice.zip -d mice && rm mice.zip

# Convert xls -> csv (needs: pip install pandas xlrd)
python3 -c "import pandas as pd; pd.read_excel('mice/Data_Cortex_Nuclear.xls').to_csv('csvdata/CortexNuclear.csv', index=False)"
rm -r mice

# Chinook database (zip containing chinook.db)
curl -fsSL -o chinook.zip "https://www.sqlitetutorial.net/wp-content/uploads/2018/03/chinook.zip"
unzip -o chinook.zip && rm chinook.zip

# Images 0-9 (public Wikimedia Commons substitutes, 960 px wide; no login needed)
mkdir -p imagedata
python3 - <<'PY'
import urllib.parse, urllib.request
files = [
 "Pelargonium-Sitia-Crete-1.jpg",
 "Red_Garden_Geranium_(Pelargonium_Zonale)_on_May_10,_2023.jpg",
 "Red_Garden_Geranium_(Pelargonium_Zonale)_in_Athens_on_April_30,_2023.jpg",
 "Red_and_white_horseshoe_geranium_in_Tuntorp_1.jpg",
 "Garden_Geranium_Pelargonium.jpg",
 "Ferrari-Monaco-4071021.jpg",
 "Ferrari_LaFerrari_at_the_Ferrari_museum.jpg",
 "Silver_Ferrari_Luxury_Sports_Car.jpg",
 "Red_Ferrari_sports_car_in_HK_20190608_191049.jpg",
 "Ferrari_California_Sports_Car_WORKOUT_19-12-2022.jpg",
]
for i, f in enumerate(files):
    url = "https://commons.wikimedia.org/wiki/Special:FilePath/" + urllib.parse.quote(f) + "?width=960"
    req = urllib.request.Request(url, headers={"User-Agent": "ml-exercise-sheet01/1.0"})
    open(f"imagedata/{i}.jpg", "wb").write(urllib.request.urlopen(req).read())
PY