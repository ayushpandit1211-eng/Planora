CREATE TABLE customers (id int PRIMARY KEY, name text);
INSERT INTO customers VALUES (1, 'Asha');
COPY customers FROM '/etc/passwd';
CREATE FUNCTION hack() RETURNS int LANGUAGE sql AS $$ SELECT 1 $$;
GRANT ALL ON customers TO public;
DROP TABLE customers;
