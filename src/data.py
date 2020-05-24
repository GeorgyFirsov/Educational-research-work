import pandas as pd

# Строчки-имена файлов с данными
# Пути заданы относительно каталога с .ipynb файлом
beer_file_name = './data/beer.csv'
temperatures_file_name = './data/daily-min-temperatures.csv'
oxygen_file_name = './data/oxygen.csv'
sunspotarea_file_name = './data/sunspotarea.csv'
passengers_file_name = './data/TS Total passenger movements.csv'
wineind_file_name = './data/wineind.csv'

# Объекты рядов из библиотечки pandas для вышеуказанных рядов
beer = pd.Series.from_csv(beer_file_name, header=0)
temperatures = pd.Series.from_csv(temperatures_file_name, header=0)
oxygen = pd.Series.from_csv(oxygen_file_name, header=0)
sunspotarea = pd.Series.from_csv(sunspotarea_file_name, header=0)
passengers = pd.Series.from_csv(passengers_file_name, header=0)
wineind = pd.Series.from_csv(wineind_file_name, header=0)

# Объекты датафреймов для вышеуказанных рядов
beer_df = pd.read_csv(beer_file_name)
temperatures_df = pd.DataFrame(temperatures)
oxygen_df = pd.read_csv(oxygen_file_name)
sunspotarea_df = pd.read_csv(sunspotarea_file_name)
passengers_df = pd.read_csv(passengers_file_name)
wineind_df = pd.read_csv(wineind_file_name)