"""Generate many awkward datasets."""
import csv, io, json, random, datetime as dt, uuid
from pathlib import Path
random.seed(42)
OUT = Path("/tmp/claude-0/stress/data"); OUT.mkdir(parents=True, exist_ok=True)
def w(name, text, enc="utf-8"): (OUT/name).write_bytes(text.encode(enc) if isinstance(text,str) else text)
def rows_csv(header, rows, delim=",", crlf=False):
    b=io.StringIO(); wr=csv.writer(b, delimiter=delim, lineterminator="\r\n" if crlf else "\n"); wr.writerow(header); wr.writerows(rows); return b.getvalue()
N=300
d0=dt.date(2023,1,1)
# 1 classic sales
w("01_sales.csv", rows_csv(["Order ID","Customer Name","Region","Amount ($)","Order Date"],
  [[i,f"Cust {i%50}",random.choice(["N","S","E","W"]),f"{random.uniform(1,999):.2f}",(d0+dt.timedelta(days=i%700)).isoformat()] for i in range(N)]))
# 2 semicolon + european-ish + BOM + CRLF
w("02_semicolon_bom.csv", "﻿"+rows_csv(["id","name","city","score"],[[i,f"Nom {i}",random.choice(["Paris","Lyon","Köln"]),random.randint(0,100)] for i in range(N)],";",True))
# 3 tab separated
w("03_tabs.tsv", rows_csv(["a","b","c"],[[i,i*2.5,random.choice(["x","y"])] for i in range(N)],"\t"))
# 4 pipe + quotes with commas and newlines inside
w("04_quoted.csv", rows_csv(["id","comment","category"],[[i,f'He said "hi", then\nleft {i}',random.choice(["a","b"])] for i in range(N)]))
# 5 reserved word / weird headers / duplicates / unicode / empty header
w("05_weird_headers.csv", rows_csv(["select","from","Group","group","","Ünïcødé Col","1st","a b c","x"*90],
  [[i,i,"g"+str(i%3),i%7,"v",f"é{i}",i,f"t{i%4}",i] for i in range(N)]))
# 6 mixed types in a column -> text
w("06_mixed.csv", rows_csv(["id","val","maybe_num"],[[i,random.choice(["1","2.5","abc","",None]),random.choice(["10","20","n/a"])] for i in range(N)]))
# 7 all empty column + all null tokens
w("07_empty_cols.csv", rows_csv(["id","empty","nulls","v"],[[i,"","NULL",i%5] for i in range(N)]))
# 8 booleans various
w("08_bools.csv", rows_csv(["id","is_active","flag","yes_no"],[[i,random.choice(["true","false"]),random.choice(["0","1"]),random.choice(["yes","no"])] for i in range(N)]))
# 9 dates: dd/mm/yyyy, mm/dd ambiguous, timestamps, tz
w("09_dates.csv", rows_csv(["id","d_dmy","d_us","ts","tstz"],[[i,f"{random.randint(13,28)}/{random.randint(1,12)}/2024",f"{random.randint(1,12)}/{random.randint(1,12)}/2024",f"2024-03-{i%28+1:02d} 10:{i%60:02d}:00",f"2024-03-{i%28+1:02d}T10:00:00+05:30"] for i in range(N)]))
# 10 big ints, negative, scientific, leading zero ids
w("10_numbers.csv", rows_csv(["id","big","neg","sci","zip","pct"],[[i,str(10**17+i),-i*1.5,f"{i}e3","0"+str(1000+i),f"{i%100}%"] for i in range(N)]))
# 11 single column
w("11_one_col.csv", rows_csv(["value"],[[i] for i in range(N)]))
# 12 wide table 60 cols
w("12_wide.csv", rows_csv([f"col{j}" for j in range(60)],[[i*j for j in range(60)] for i in range(50)]))
# 13 latin1 encoding
w("13_latin1.csv", rows_csv(["id","name"],[[i,f"José Ñandú {i}"] for i in range(N)]), enc="latin-1")
# 14 no header numeric
w("14_noheader.csv", "\n".join(f"{i},{i*3},{i%7}" for i in range(N))+"\n")
# 15 ragged rows (few)
lines=["id,a,b"]+[f"{i},{i},{i}" for i in range(N)]+["999,1,2,3,4","1000,1"]
w("15_ragged.csv","\n".join(lines)+"\n")
# 16 currency + thousands consistent
w("16_currency.csv", rows_csv(["id","price","revenue"],[[i,f"${random.randint(1,999)}.{random.randint(10,99)}",f"{random.randint(1,999)},{random.randint(100,999)}.00"] for i in range(N)]))
# 17 uuid + json column
w("17_uuid_json.csv", rows_csv(["id","ref","payload"],[[i,str(uuid.uuid4()),json.dumps({"k":i,"t":"x"})] for i in range(N)]))
# 18 json array nested
w("18_nested.json", json.dumps([{"id":i,"user":{"name":f"U{i}","age":20+i%40},"tags":["a","b"],"meta":{"deep":{"x":i}},"active":i%2==0} for i in range(N)]))
# 19 jsonl with missing keys
w("19_lines.jsonl", "\n".join(json.dumps({"id":i,**({"price":i*1.1} if i%2 else {"category":"c"+str(i%3)})}) for i in range(N)))
# 20 json object with one array
w("20_obj.json", json.dumps({"data":[{"sku":f"S{i}","qty":i%9,"store":random.choice(["A","B"])} for i in range(N)]}))
# 21 kaggle-like titanic-ish
w("21_titanic.csv", rows_csv(["PassengerId","Survived","Pclass","Name","Sex","Age","Fare","Embarked"],
  [[i,random.randint(0,1),random.randint(1,3),f"Name, Mr. {i}",random.choice(["male","female"]),random.choice(["",str(random.randint(1,80))]),f"{random.uniform(5,500):.4f}",random.choice(["S","C","Q",""])] for i in range(N)]))
# 22 iris-like
w("22_iris.csv", rows_csv(["sepal.length","sepal.width","petal.length","petal.width","variety"],[[round(random.uniform(4,8),1),round(random.uniform(2,4.5),1),round(random.uniform(1,7),1),round(random.uniform(0.1,2.5),1),random.choice(["Setosa","Versicolor","Virginica"])] for _ in range(150)]))
# 23 column names with sql injection text
w("23_injection.csv", rows_csv(["id","name\"; DROP TABLE x;--","o'brien"],[[i,f"Robert'); DROP TABLE students;-- {i}","x"] for i in range(N)]))
# 24 numeric-looking text with spaces / padded
w("24_spaces.csv", rows_csv([" id "," Amount ","  Cat  "],[[f" {i} ",f" {i*2.5} ",f" c{i%3} "] for i in range(N)]))
# 25 large-ish 100k rows
w("25_large.csv", rows_csv(["id","store","sku","qty","price","sold_at"],[[i,f"st{i%40}",f"sku{i%900}",i%11,f"{(i%500)+0.99}",(dt.datetime(2023,1,1)+dt.timedelta(minutes=i*7)).isoformat(sep=" ")] for i in range(100000)]))
# 26 duplicated identical rows
w("26_dupes.csv", rows_csv(["a","b"],[[1,"x"]]*N))
# 27 html-ish / emoji text
w("27_emoji.csv", rows_csv(["id","text","mood"],[[i,f"<b>hi</b> 😀 {i}",random.choice(["😀","😢"])] for i in range(N)]))
# 28 percent/null mix and 'None'
w("28_nullish.csv", rows_csv(["id","a","b"],[[i,random.choice(["None","nan","NaN","-",str(i)]),random.choice(["N/A",str(i*1.0)])] for i in range(N)]))
# 29 time only + interval-ish
w("29_times.csv", rows_csv(["id","t","dur"],[[i,f"{i%24:02d}:{i%60:02d}",f"{i} days"] for i in range(N)]))
# 30 header only (no rows)
w("30_header_only.csv","id,name\n")
print(len(list(OUT.iterdir())))
