import requests, json
H={"User-Agent":"Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15","Referer":"https://m.stock.naver.com/","Accept":"application/json"}
for p in [1,2,5,10,20,50,100]:
    url=f"https://m.stock.naver.com/api/stock/005930/trend?pageSize=30&page={p}"
    try:
        r=requests.get(url,headers=H,timeout=10)
        print("PAGE",p,"STATUS",r.status_code,"LEN",len(r.text))
        if r.ok:
            j=r.json()
            arr=j if isinstance(j,list) else j.get("list") or j.get("result") or []
            ds=[str(x.get("bizdate") or x.get("stcTrdDd") or x.get("date") or "") for x in arr]
            print("N",len(arr),"FIRST",ds[0] if ds else None,"LAST",ds[-1] if ds else None)
        else: print(r.text[:200])
    except Exception as e: print("ERR",p,repr(e))
