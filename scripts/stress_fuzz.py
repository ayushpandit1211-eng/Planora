import csv, io, json, random, string, uuid, datetime as dt, httpx, sys
B="http://localhost:8000"; S="sess-fuzz"+uuid.uuid4().hex[:8]; H={"X-Session-Id":S}
def call(m,p,**kw):
    r=httpx.request(m,B+p,headers={**H,**kw.pop("headers",{})},timeout=300,**kw)
    try: return r.status_code,r.json()
    except Exception: return r.status_code,{"raw":r.text[:200]}
def up(name,data,ds):
    c,pv=call("POST",f"/api/v1/data/preview?filename={name}&dataset_name={ds}",content=data,headers={"content-type":"application/octet-stream"})
    if c!=200: return c,pv,None
    c,cm=call("POST","/api/v1/data/commit",json={"preview_id":pv["preview_id"],"allow_duplicate":True})
    return c,cm,pv
GENS={
 "int":lambda r,i:str(r.randint(-10**6,10**6)),
 "float":lambda r,i:f"{r.uniform(-1e4,1e4):.3f}",
 "word":lambda r,i:r.choice(["red","green","blue","Black","white"]),
 "text":lambda r,i:"".join(r.choice(string.ascii_letters+" ,'\"") for _ in range(r.randint(0,25))),
 "date":lambda r,i:(dt.date(2020,1,1)+dt.timedelta(days=r.randint(0,2000))).isoformat(),
 "ts":lambda r,i:(dt.datetime(2021,1,1)+dt.timedelta(seconds=r.randint(0,10**8))).isoformat(sep=" "),
 "bool":lambda r,i:r.choice(["true","false","TRUE","False"]),
 "uuid":lambda r,i:str(uuid.UUID(int=r.getrandbits(128))),
 "money":lambda r,i:f"${r.randint(1,9)},{r.randint(100,999)}.{r.randint(10,99)}",
 "code":lambda r,i:"0"+str(r.randint(100,999)),
}
fails=[]; ok=0
for k in range(int(sys.argv[1]) if len(sys.argv)>1 else 25):
    r=random.Random(1000+k)
    ncol=r.randint(1,12); kinds=[r.choice(list(GENS)) for _ in range(ncol)]
    names=[r.choice(["ID","Name","Value","Total Sales","date","Region","x","Amount (USD)","user-id","Created At","status","Group","order","%Rate"])+(str(j) if r.random()<.5 else "") for j in range(ncol)]
    nrows=r.randint(1,800); null=r.random()*0.3; delim=r.choice([",",";","\t","|"])
    def rows(n,off=0):
        return [[("" if r.random()<null else GENS[t](r,i)) for t in kinds] for i in range(off,off+n)]
    b=io.StringIO(); w=csv.writer(b,delimiter=delim); w.writerow(names); w.writerows(rows(nrows))
    ds=f"fz{k}"
    c,cm,pv=up(f"f{k}.csv",b.getvalue().encode(),ds)
    if c!=200: fails.append((k,"load",kinds,c,cm.get("error"),cm.get("message"),str(cm.get("details"))[:200])); continue
    sid=cm["data_source_id"]; tbl=cm["table"].split(".")[1]
    # append a second batch with bigger ints + one extra column
    b2=io.StringIO(); w2=csv.writer(b2,delimiter=delim); w2.writerow(names+["extra_col"])
    for row in rows(50,nrows): w2.writerow([(str(10**9) if kinds[j]=="int" and row[j] else row[j]) for j in range(ncol)]+["e"])
    c2,cm2,_=up(f"f{k}.csv",b2.getvalue().encode(),ds)
    if c2!=200: fails.append((k,"append",kinds,c2,cm2.get("error"),cm2.get("message"),str(cm2.get("details"))[:300]))
    c,sch=call("GET",f"/api/v1/sources/{sid}/schema"); cols=sch["tables"][0]["columns"]
    for sq in [f'SELECT * FROM "{tbl}" LIMIT 3', f'SELECT count(*) FROM "{tbl}"']+[f'SELECT "{x["name"]}", count(*) FROM "{tbl}" GROUP BY 1 LIMIT 5' for x in cols[:3]]:
        c,o=call("POST","/api/v1/nlq/run",json={"sql":sq,"data_source_id":sid,"optimize":False})
        if c!=200: fails.append((k,"sql",sq,c,o.get("error"),o.get("message")))
    for q in ["how many rows", f"count by {cols[0]['name'].replace('_',' ')}", f"average {cols[-1]['name'].replace('_',' ')}"]:
        c,o=call("POST","/api/v1/nlq/ask",json={"question":q,"data_source_id":sid,"optimize":False})
        if c!=200: fails.append((k,"ask",q,c,o.get("error"),o.get("message")))
    c,o=call("POST","/api/v1/nlq/run",json={"sql":f'SELECT * FROM "{tbl}" WHERE "{cols[0]["name"]}" IS NOT NULL LIMIT 20',"data_source_id":sid,"optimize":True})
    if c!=200 or (o.get("optimization") or {}).get("status")!="completed": fails.append((k,"optimize",c,o.get("error"),o.get("message"),(o.get("optimization") or {}).get("status")))
    ok+=1
print("datasets ok:",ok,"failures:",len(fails))
for f in fails: print("FAIL",f)
