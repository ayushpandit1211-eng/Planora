import json, sys, httpx, uuid
from pathlib import Path
B="http://localhost:8000"; S="sess-stress"+uuid.uuid4().hex[:8]; H={"X-Session-Id":S}
D=Path("/tmp/claude-0/stress/data")
def call(m,p,**kw):
    r=httpx.request(m,B+p,headers={**H,**kw.pop("headers",{})},timeout=300,**kw)
    try: return r.status_code, r.json()
    except Exception: return r.status_code, {"raw": r.text[:200]}
fails=[]; summary=[]
only=sys.argv[1:] 
for f in sorted(D.iterdir()):
    if only and not any(o in f.name for o in only): continue
    ds="st"+f.name[:2]
    c,pv=call("POST",f"/api/v1/data/preview?filename={f.name}&dataset_name={ds}",content=f.read_bytes(),headers={"content-type":"application/octet-stream"})
    if c!=200: fails.append((f.name,"preview",c,pv.get("error"),pv.get("message"))); continue
    c,cm=call("POST","/api/v1/data/commit",json={"preview_id":pv["preview_id"],"allow_duplicate":True})
    if c!=200: fails.append((f.name,"commit",c,cm.get("error"),cm.get("message"),cm.get("details"))); continue
    sid=cm["data_source_id"]; tbl=cm["table"].split(".")[1]
    c,sch=call("GET",f"/api/v1/sources/{sid}/schema")
    if c!=200: fails.append((f.name,"schema",c,sch)); continue
    cols=sch["tables"][0]["columns"]
    txt=[x["name"] for x in cols if x["type"].startswith(("text","character"))]
    num=[x["name"] for x in cols if x["type"].split("(")[0] in ("smallint","integer","bigint","numeric","double precision")]
    qi=lambda n:'"'+n.replace('"','""')+'"'
    sqls=[f"SELECT * FROM {qi(tbl)} LIMIT 5", f"SELECT count(*) FROM {qi(tbl)}"]
    if txt: sqls.append(f"SELECT {qi(txt[0])}, count(*) FROM {qi(tbl)} GROUP BY 1 ORDER BY 2 DESC")
    if txt and num: sqls.append(f"SELECT {qi(txt[0])}, avg({qi(num[-1])}), max({qi(num[-1])}) FROM {qi(tbl)} GROUP BY 1")
    if num: sqls.append(f"SELECT * FROM {qi(tbl)} WHERE {qi(num[0])} > 5 ORDER BY {qi(num[0])} DESC LIMIT 10")
    for i,sq in enumerate(sqls):
        c,o=call("POST","/api/v1/nlq/run",json={"sql":sq,"data_source_id":sid,"optimize": i==len(sqls)-1})
        if c!=200 or o.get("status")!="completed": fails.append((f.name,"sql",sq[:90],c,o.get("error"),o.get("message"),(o.get("details") or {}).get("stage")))
        elif i==len(sqls)-1 and (o.get("optimization") or {}).get("status")!="completed": fails.append((f.name,"optimize",o.get("optimization")))
    qs=["how many rows"]
    if txt: qs.append(f"count by {txt[0].replace('_',' ')}")
    if txt and num: qs.append(f"average {num[-1].replace('_',' ')} by {txt[0].replace('_',' ')}")
    if num: qs.append(f"total {num[-1].replace('_',' ')}")
    for q in qs:
        c,o=call("POST","/api/v1/nlq/ask",json={"question":q,"data_source_id":sid,"optimize":False})
        if c!=200: fails.append((f.name,"ask",q,c,o.get("error"),o.get("message"),(o.get("details") or {}).get("stage")))
        elif o.get("status") not in ("completed","clarification"): fails.append((f.name,"ask",q,o))
    summary.append((f.name, cm["rows_loaded"], cm["rows_rejected"], [(x["name"],x["type"]) for x in cols][:6]))
for s in summary: print("OK", s)
print("FAILURES", len(fails))
for x in fails: print("FAIL", x)
