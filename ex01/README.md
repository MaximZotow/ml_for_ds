# Exercise Sheet 1

Notebook: [sheet01.ipynb](sheet01.ipynb)

## Data sources (UCI Machine Learning Repository)

Files are stored in `csvdata/`.

| File | Dataset | Link |
|------|---------|------|
| `csvdata/Wholesale customers data.csv` | Wholesale customers | https://archive.ics.uci.edu/dataset/292/wholesale+customers (direct CSV: https://archive.ics.uci.edu/ml/machine-learning-databases/00292/Wholesale%20customers%20data.csv) |
| `csvdata/CortexNuclear.csv` | Mice Protein Expression | https://archive.ics.uci.edu/dataset/342/mice+protein+expression (zip: https://archive.ics.uci.edu/static/public/342/mice+protein+expression.zip) |

`CortexNuclear.csv` was converted from the original `Data_Cortex_Nuclear.xls` in the zip (1080 rows, 82 columns).

## Download and unzip

Run from this folder (`01/`):

```bash
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
```

## Other sources

- `chinook.db`: https://www.sqlitetutorial.net/sqlite-sample-database/ (download command above)
- `imagedata/0.jpg` … `9.jpg`: the sheet's original images are not publicly available, so the download command above fetches the substitutes listed below.

## Substitute images

`imagedata/` currently holds substitutes from Wikimedia Commons (resized to 960 px wide): 0–4 are geraniums (Pelargonium) and 5–9 are Ferrari cars. The distance matrices will therefore differ from the sheet's figures.

| # | Commons file |
|---|--------------|
| 0 | [Pelargonium-Sitia-Crete-1.jpg](https://commons.wikimedia.org/wiki/File:Pelargonium-Sitia-Crete-1.jpg) |
| 1 | [Red Garden Geranium (Pelargonium Zonale) on May 10, 2023.jpg](https://commons.wikimedia.org/wiki/File:Red_Garden_Geranium_(Pelargonium_Zonale)_on_May_10,_2023.jpg) |
| 2 | [Red Garden Geranium (Pelargonium Zonale) in Athens on April 30, 2023.jpg](https://commons.wikimedia.org/wiki/File:Red_Garden_Geranium_(Pelargonium_Zonale)_in_Athens_on_April_30,_2023.jpg) |
| 3 | [Red and white horseshoe geranium in Tuntorp 1.jpg](https://commons.wikimedia.org/wiki/File:Red_and_white_horseshoe_geranium_in_Tuntorp_1.jpg) |
| 4 | [Garden Geranium Pelargonium.jpg](https://commons.wikimedia.org/wiki/File:Garden_Geranium_Pelargonium.jpg) |
| 5 | [Ferrari-Monaco-4071021.jpg](https://commons.wikimedia.org/wiki/File:Ferrari-Monaco-4071021.jpg) |
| 6 | [Ferrari LaFerrari at the Ferrari museum.jpg](https://commons.wikimedia.org/wiki/File:Ferrari_LaFerrari_at_the_Ferrari_museum.jpg) |
| 7 | [Silver Ferrari Luxury Sports Car.jpg](https://commons.wikimedia.org/wiki/File:Silver_Ferrari_Luxury_Sports_Car.jpg) |
| 8 | [Red Ferrari sports car in HK 20190608 191049.jpg](https://commons.wikimedia.org/wiki/File:Red_Ferrari_sports_car_in_HK_20190608_191049.jpg) |
| 9 | [Ferrari California Sports Car WORKOUT 19-12-2022.jpg](https://commons.wikimedia.org/wiki/File:Ferrari_California_Sports_Car_WORKOUT_19-12-2022.jpg) |

Check each file's page for its license and attribution before reusing the images outside this exercise.
