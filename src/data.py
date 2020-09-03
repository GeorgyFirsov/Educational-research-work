import pandas as pd

# Строчки-имена файлов с данными
# Пути заданы относительно каталога с .ipynb файлом
beer_file_name = './data/beer.csv'
temperatures_file_name = './data/daily-min-temperatures.csv'
oxygen_file_name = './data/oxygen.csv'
sunspotarea_file_name = './data/sunspotarea.csv'
passengers_file_name = './data/TS Total passenger movements.csv'
wineind_file_name = './data/wineind.csv'

# Объекты датафреймов для вышеуказанных рядов
beer = pd.read_csv(beer_file_name)
temperatures = pd.read_csv(temperatures_file_name)
oxygen = pd.read_csv(oxygen_file_name)
sunspotarea = pd.read_csv(sunspotarea_file_name)
passengers = pd.read_csv(passengers_file_name)
wineind = pd.read_csv(wineind_file_name)