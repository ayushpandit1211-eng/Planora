--
-- PostgreSQL database dump (schema only) - example BYOD upload for Planora
--
\restrict abc123
SET statement_timeout = 0;
SET client_encoding = 'UTF8';
SELECT pg_catalog.set_config('search_path', '', false);

CREATE TYPE public.order_status AS ENUM ('new', 'paid', 'shipped', 'cancelled');

CREATE TABLE public.customers (
    id bigint NOT NULL,
    email text NOT NULL,
    country text,
    created_at timestamp with time zone DEFAULT now() NOT NULL
);
ALTER TABLE public.customers OWNER TO shop_owner;

CREATE SEQUENCE public.customers_id_seq START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;
ALTER SEQUENCE public.customers_id_seq OWNED BY public.customers.id;

CREATE TABLE public.products (
    id integer NOT NULL,
    name text NOT NULL,
    category text,
    price numeric(10,2) NOT NULL CHECK (price >= 0)
);

CREATE TABLE public.orders (
    id bigint NOT NULL,
    customer_id bigint NOT NULL,
    status public.order_status DEFAULT 'new' NOT NULL,
    ordered_at timestamp with time zone NOT NULL,
    total numeric(12,2)
);

CREATE TABLE public.order_lines (
    order_id bigint NOT NULL,
    product_id integer NOT NULL,
    quantity integer NOT NULL CHECK (quantity > 0),
    unit_price numeric(10,2) NOT NULL
);

ALTER TABLE ONLY public.customers ALTER COLUMN id SET DEFAULT nextval('public.customers_id_seq'::regclass);
ALTER TABLE ONLY public.customers ADD CONSTRAINT customers_pkey PRIMARY KEY (id);
ALTER TABLE ONLY public.customers ADD CONSTRAINT customers_email_key UNIQUE (email);
ALTER TABLE ONLY public.products ADD CONSTRAINT products_pkey PRIMARY KEY (id);
ALTER TABLE ONLY public.orders ADD CONSTRAINT orders_pkey PRIMARY KEY (id);
ALTER TABLE ONLY public.order_lines ADD CONSTRAINT order_lines_pkey PRIMARY KEY (order_id, product_id);
ALTER TABLE ONLY public.orders ADD CONSTRAINT orders_customer_fk FOREIGN KEY (customer_id) REFERENCES public.customers(id);
ALTER TABLE ONLY public.order_lines ADD CONSTRAINT order_lines_order_fk FOREIGN KEY (order_id) REFERENCES public.orders(id);
ALTER TABLE ONLY public.order_lines ADD CONSTRAINT order_lines_product_fk FOREIGN KEY (product_id) REFERENCES public.products(id);
CREATE INDEX orders_status_idx ON public.orders USING btree (status);
COMMENT ON TABLE public.orders IS 'customer orders';
\unrestrict abc123
