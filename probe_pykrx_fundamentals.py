from pykrx import stock
import pandas as pd, json
for dt in ["20191230","20201230","20251230","20260910"]:
    try:
        x=stock.get_market_fundamental_by_ticker(dt, market="KOSDAQ")
        print(dt, len(x), x.head(3).to_dict("index"))
    except Exception as e:
        print(dt, "ERR", repr(e))
