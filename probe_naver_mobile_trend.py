import requests
H={"User-Agent":"Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15","Referer":"https://m.stock.naver.com/","Accept":"application/json"}
for n in [30,60,120,250,500,1000,3000]:
    url=f"https://m.stock.naver.com/api/stock/005930/trend?pageSize={n}"
    try:
        r=requests.get(url,headers=H,timeout=10)
        print("SIZE",n,"STATUS",r.status_code,"LEN",len(r.text))
        if r.ok:
            j=r.json(); arr=j if isinstance(j,list) else j.get("list") or j.get("result") or []
            ds=[str(x.get("bizdate") or x.get("stcTrdDd") or x.get("date") or "") for x in arr]
            print("N",len(arr),"FIRST",ds[0] if ds else None,"LAST",ds[-1] if ds else None)
    except Exception as e: print("ERR",n,repr(e))
