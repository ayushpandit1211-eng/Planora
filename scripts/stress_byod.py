import httpx, uuid
B="http://localhost:8000"; S="sess-byod"+uuid.uuid4().hex[:8]; H={"X-Session-Id":S}
SCHEMAS={
"serial_fk": """CREATE TABLE authors (id serial PRIMARY KEY, name varchar(100) NOT NULL, country char(2));
CREATE TABLE books (id bigserial PRIMARY KEY, author_id int NOT NULL REFERENCES authors(id), title text NOT NULL, price numeric(8,2) CHECK (price > 0), published date);
CREATE INDEX books_author_idx ON books (author_id);""",
"identity_enum": """CREATE TYPE mood AS ENUM ('happy','sad','ok');
CREATE TABLE people (id int GENERATED ALWAYS AS IDENTITY PRIMARY KEY, email text UNIQUE NOT NULL, mood mood NOT NULL, born timestamptz);""",
"composite_pk": """CREATE TABLE students (id int PRIMARY KEY, name text);
CREATE TABLE courses (code text PRIMARY KEY, title text);
CREATE TABLE enrolments (student_id int REFERENCES students(id), course_code text REFERENCES courses(code), grade smallint, PRIMARY KEY (student_id, course_code));""",
"generated_col": """CREATE TABLE items (id int PRIMARY KEY, qty int NOT NULL, unit_price numeric(10,2) NOT NULL, total numeric GENERATED ALWAYS AS (qty * unit_price) STORED, tags jsonb, ref uuid DEFAULT gen_random_uuid());""",
"own_schema_view": """CREATE SCHEMA shop;
CREATE TABLE shop.customers (id int PRIMARY KEY, city text);
CREATE TABLE shop.orders (id int PRIMARY KEY, customer_id int REFERENCES shop.customers(id), amount numeric);
CREATE VIEW shop.big_orders AS SELECT * FROM shop.orders WHERE amount > 100;""",
"quoted_mixed_case": """CREATE TABLE "Customer Data" ("Id" int PRIMARY KEY, "Full Name" text, "Created At" timestamp);""",
"self_ref": """CREATE TABLE employees (id int PRIMARY KEY, manager_id int REFERENCES employees(id), salary int CHECK (salary >= 0), dept text);""",
"pgdump_alter_fks": open("/home/claude/work/db-opt-platform/examples/byod/shop_schema.sql").read(),
}
QS={"serial_fk":["count by country","SELECT a.name, count(b.id) FROM authors a JOIN books b ON b.author_id = a.id GROUP BY a.name ORDER BY 2 DESC LIMIT 5"],
    "identity_enum":["count by mood","SELECT mood, count(*) FROM people GROUP BY mood"],
    "composite_pk":["average grade","SELECT course_code, avg(grade) FROM enrolments GROUP BY 1"],
    "generated_col":["total qty","SELECT sum(total) FROM items"],
    "own_schema_view":["how many rows","SELECT count(*) FROM orders"],
    "quoted_mixed_case":["how many rows",'SELECT count(*) FROM "Customer Data"'],
    "self_ref":["average salary by dept","SELECT dept, max(salary) FROM employees GROUP BY dept"],
    "pgdump_alter_fks":["total amount of orders by status","SELECT c.country, sum(o.total) FROM orders o JOIN customers c ON c.id = o.customer_id GROUP BY 1"]}
fails=[]
for name, ddl in SCHEMAS.items():
    r=httpx.post(B+"/api/v1/sandboxes?scale=small&name="+name,content=ddl.encode(),headers=H,timeout=300); o=r.json()
    if r.status_code!=200: fails.append((name,"create",r.status_code,o.get("error"),o.get("message"),str(o.get("details"))[:300])); continue
    sid=o["data_source_id"]
    print(name, "tables:", o["synthetic_data"]["tables"], "unsatisfied:", [u["reason"][:60] for u in o["synthetic_data"]["unsatisfied"]])
    q,sql=QS[name]
    r=httpx.post(B+"/api/v1/nlq/ask",json={"question":q,"data_source_id":sid,"optimize":False},headers=H,timeout=300)
    if r.status_code!=200: fails.append((name,"ask",q,r.json().get("message")))
    r=httpx.post(B+"/api/v1/nlq/run",json={"sql":sql,"data_source_id":sid,"optimize":True},headers=H,timeout=300)
    o2=r.json()
    if r.status_code!=200: fails.append((name,"sql",sql,o2.get("error"),o2.get("message")))
    elif (o2.get("optimization") or {}).get("status")!="completed": fails.append((name,"optimize",o2.get("optimization")))
    httpx.delete(B+f"/api/v1/sandboxes/{sid}",headers=H,timeout=60)
print("failures:",len(fails))
for f in fails: print("FAIL",f)
