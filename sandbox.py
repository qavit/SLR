import pandas as pd
df = pd.read_csv('手語影片URLs.csv')
labels = df['動作名稱'].to_list()
urls = df['網址'].to_list()
url_dict = dict(zip(labels, urls))