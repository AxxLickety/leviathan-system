import pandas_datareader.data as web
import pandas as pd

start = "1999-01-01"
end = None

series = ["FEDFUNDS", "CPIAUCSL", "CSUSHPINSA"]
df = web.DataReader(series, "fred", start, end)
print(df.tail())
print("OK, rows:", len(df))
