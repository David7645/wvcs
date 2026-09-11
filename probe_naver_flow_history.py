import pandas as pd, requests, re
from io import StringIO
url="https://finance.naver.com/item/frgn.naver?code=005930&page={}"
for p in [1,10,50,100,150,200]:
    try:
        r=requests.get(url.format(p),timeout=10,headers={"User-Agent":"Mozilla/5.0"})
        r.encoding="euc-kr"
        tabs=pd.read_html(StringIO(r.text))
        hit=None
        for t in tabs:
            cols=[" ".join(str(y).strip() for y in x if str(y).strip()) if isinstance(x,tuple) else str(x) for x in t.columns]
            if any("날짜" in c for c in cols) and any("기관" in c for c in cols) and any("외국인" in c for c in cols):
                t.columns=cols; hit=t;break
        if hit is None: print("P",p,"NONE"); continue
        datecol=next(c for c in hit.columns if "날짜" in c)
        vals=hit[datecol].dropna().astype(str)
        vals=vals[vals.str.contains(r"\d{2,4}\.\d{2}\.\d{2}",regex=True)]
        print("P",p,"N",len(vals),"FIRST",vals.iloc[0] if len(vals) else None,"LAST",vals.iloc[-1] if len(vals) else None)
    except Exception as e: print("P",p,"ERR",repr(e))
