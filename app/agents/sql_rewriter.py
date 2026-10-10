"""SQL Rewriter / SQL Expert agent (Section 23).

Operates on ANONYMISED SQL only (pseudonymised identifiers, $n parameters) and never invents
tables or columns. Deterministic, AST-based rewrite operators (each one also exposed to the RL
policy as a REWRITE_SQL low-level action):

* correlated_count_subquery_to_left_join - un-nests a correlated COUNT scalar subquery into a
  LEFT JOIN on a grouped derived table with COALESCE(cnt, 0)
* cast_date_equality_to_range - `CAST(col AS date) = $n` -> `col >= CAST($n AS date) AND
  col < CAST($n AS date) + 1` (timestamp/date columns only - makes the predicate sargable)
* remove_duplicate_predicates - drops repeated AND conjuncts
* or_equalities_to_in_list - `c = $1 OR c = $2` -> `c IN ($1, $2)`

Every proposal is checked with the AST semantic diff (R6); unexpected structural changes cause
rejection. An optional LLM proposal goes through the privacy guard, Pydantic validation,
identifier verification and the same semantic diff.
"""
from __future__ import annotations

import copy

from pglast import ast, parse_sql
from pglast.enums import A_Expr_Kind, BoolExprType, SubLinkType
from pglast.stream import RawStream

from app.contracts import RewriteProposal, SchemaMetadata
from app.masking_vault import PSEUDONYM_RE
from app.sql_semantic_diff import diff, risk_screen, verify

REWRITE_OPERATORS = ["correlated_count_subquery_to_left_join", "cast_date_equality_to_range",
                     "remove_duplicate_predicates", "or_equalities_to_in_list"]
DATE_TYPES = ("timestamp", "date")


def _deparse(node) -> str:
    return RawStream()(node)


def _colref_parts(node) -> list[str] | None:
    if isinstance(node, ast.ColumnRef) and all(isinstance(f, ast.String) for f in node.fields):
        return [f.sval for f in node.fields]
    return None


def _and_args(where) -> list:
    if isinstance(where, ast.BoolExpr) and where.boolop == BoolExprType.AND_EXPR:
        return list(where.args)
    return [where] if where is not None else []


def _make_and(args: list):
    if not args:
        return None
    if len(args) == 1:
        return args[0]
    return ast.BoolExpr(boolop=BoolExprType.AND_EXPR, args=tuple(args))


class SQLRewriter:
    def __init__(self, schema: SchemaMetadata | None = None):
        """`schema`: PSEUDONYMISED metadata (used for column types and identifier checks)."""
        self.schema = schema

    # ------------------------------------------------------------------ helpers
    def _col_type(self, alias_map: dict[str, str], parts: list[str]) -> str | None:
        if not self.schema:
            return None
        col = parts[-1]
        tables = [alias_map.get(parts[-2], parts[-2])] if len(parts) >= 2 else list(set(alias_map.values()))
        for t in tables:
            c = self.schema.column(t, col)
            if c is not None:
                return c.data_type
        return None

    @staticmethod
    def _alias_map(stmt: ast.SelectStmt) -> dict[str, str]:
        amap: dict[str, str] = {}

        def walk(item):
            if isinstance(item, ast.RangeVar):
                amap[item.alias.aliasname if item.alias else item.relname] = item.relname
                amap.setdefault(item.relname, item.relname)
            elif isinstance(item, ast.JoinExpr):
                walk(item.larg)
                walk(item.rarg)
        for it in stmt.fromClause or ():
            walk(it)
        return amap

    # ------------------------------------------------------------------ rules
    def _rule_correlated_count(self, stmt: ast.SelectStmt) -> tuple[bool, str]:
        if stmt.groupClause or stmt.distinctClause or stmt.havingClause or len(stmt.fromClause or ()) != 1:
            return False, ""
        outer_aliases = set(self._alias_map(stmt))
        for i, rt in enumerate(stmt.targetList or ()):
            sl = rt.val
            if not (isinstance(sl, ast.SubLink) and sl.subLinkType == SubLinkType.EXPR_SUBLINK):
                continue
            sub = sl.subselect
            if not isinstance(sub, ast.SelectStmt) or len(sub.targetList or ()) != 1 or len(sub.fromClause or ()) != 1:
                continue
            if sub.groupClause or sub.havingClause or sub.limitCount is not None or sub.sortClause:
                continue
            agg = sub.targetList[0].val
            if not (isinstance(agg, ast.FuncCall) and [n.sval for n in agg.funcname][-1] == "count"
                    and agg.agg_filter is None and not agg.agg_distinct):
                continue
            inner = sub.fromClause[0]
            if not isinstance(inner, ast.RangeVar):
                continue
            inner_alias = inner.alias.aliasname if inner.alias else inner.relname
            corr, rest = None, []
            for conj in _and_args(sub.whereClause):
                if isinstance(conj, ast.A_Expr) and conj.kind == A_Expr_Kind.AEXPR_OP and \
                        [n.sval for n in conj.name] == ["="]:
                    l, r = _colref_parts(conj.lexpr), _colref_parts(conj.rexpr)
                    if l and r and len(l) == 2 and len(r) == 2:
                        if l[0] == inner_alias and r[0] in outer_aliases and corr is None:
                            corr = (l, r)
                            continue
                        if r[0] == inner_alias and l[0] in outer_aliases and corr is None:
                            corr = (r, l)
                            continue
                rest.append(conj)
            if corr is None:
                continue
            # remaining inner predicates must not reference the outer query
            if any(any(p[0] in outer_aliases and p[0] != inner_alias for p in self._refs(c)) for c in rest):
                continue
            (in_col, out_col) = corr
            count_arg = "*" if agg.agg_star else _deparse(agg.args[0])
            where_sql = f" WHERE {' AND '.join(_deparse(c) for c in rest)}" if rest else ""
            derived = (f"SELECT {inner_alias}.{in_col[1]} AS qs_k, count({count_arg}) AS qs_cnt "
                       f"FROM {_deparse(inner)}{where_sql} GROUP BY {inner_alias}.{in_col[1]}")
            from_sql = (f"SELECT 1 FROM {_deparse(stmt.fromClause[0])} LEFT JOIN ({derived}) AS qs_sq{i} "
                        f"ON qs_sq{i}.qs_k = {out_col[0]}.{out_col[1]}")
            new_from = parse_sql(from_sql)[0].stmt.fromClause
            coalesce = parse_sql(f"SELECT COALESCE(qs_sq{i}.qs_cnt, 0)")[0].stmt.targetList[0].val
            stmt.fromClause = new_from
            targets = list(stmt.targetList)
            targets[i] = ast.ResTarget(name=rt.name or "count", val=coalesce)
            stmt.targetList = tuple(targets)
            return True, (f"correlated COUNT subquery on {inner.relname} executed per outer row was un-nested into "
                          f"a LEFT JOIN on a grouped derived table; COALESCE(...,0) preserves zero counts")
        return False, ""

    @staticmethod
    def _refs(node) -> list[list[str]]:
        out: list[list[str]] = []

        def walk(n):
            if isinstance(n, ast.ColumnRef):
                p = _colref_parts(n)
                if p and len(p) >= 2:
                    out.append(p)
            elif isinstance(n, ast.Node):
                for name in n.__slots__:
                    walk(getattr(n, name, None))
            elif isinstance(n, tuple):
                for x in n:
                    walk(x)
        walk(node)
        return out

    def _rule_cast_date(self, stmt: ast.SelectStmt) -> tuple[bool, str]:
        amap = self._alias_map(stmt)
        args = _and_args(stmt.whereClause)
        changed, notes = False, []
        new_args = []
        for conj in args:
            if isinstance(conj, ast.A_Expr) and conj.kind == A_Expr_Kind.AEXPR_OP and [n.sval for n in conj.name] == ["="] \
                    and isinstance(conj.lexpr, ast.TypeCast) \
                    and [n.sval for n in conj.lexpr.typeName.names][-1] == "date" \
                    and _colref_parts(conj.lexpr.arg) and isinstance(conj.rexpr, (ast.ParamRef, ast.TypeCast, ast.A_Const)):
                parts = _colref_parts(conj.lexpr.arg)
                ctype = self._col_type(amap, parts) or ""
                if ctype.startswith(DATE_TYPES):
                    col = _deparse(conj.lexpr.arg)
                    val = _deparse(conj.rexpr)
                    repl = parse_sql(f"SELECT 1 WHERE {col} >= CAST({val} AS date) AND {col} < CAST({val} AS date) + 1")[0].stmt.whereClause
                    new_args.extend(repl.args)
                    changed = True
                    notes.append(f"CAST({col} AS date) = ... rewritten to a half-open range so an index on {parts[-1]} can be used")
                    continue
            new_args.append(conj)
        if changed:
            stmt.whereClause = _make_and(new_args)
        return changed, "; ".join(notes)

    @staticmethod
    def _rule_dedupe(stmt: ast.SelectStmt) -> tuple[bool, str]:
        args = _and_args(stmt.whereClause)
        seen, keep = set(), []
        for a in args:
            k = _deparse(a)
            if k in seen:
                continue
            seen.add(k)
            keep.append(a)
        if len(keep) < len(args):
            stmt.whereClause = _make_and(keep)
            return True, f"removed {len(args) - len(keep)} duplicate predicate(s)"
        return False, ""

    @staticmethod
    def _rule_or_to_in(stmt: ast.SelectStmt) -> tuple[bool, str]:
        args = _and_args(stmt.whereClause)
        changed = False
        out = []
        for a in args:
            if isinstance(a, ast.BoolExpr) and a.boolop == BoolExprType.OR_EXPR:
                cols, vals = set(), []
                ok = True
                for o in a.args:
                    if isinstance(o, ast.A_Expr) and o.kind == A_Expr_Kind.AEXPR_OP and [n.sval for n in o.name] == ["="] \
                            and _colref_parts(o.lexpr) and isinstance(o.rexpr, (ast.ParamRef, ast.A_Const, ast.TypeCast)):
                        cols.add(_deparse(o.lexpr))
                        vals.append(_deparse(o.rexpr))
                    else:
                        ok = False
                        break
                if ok and len(cols) == 1 and len(vals) >= 2:
                    out.append(parse_sql(f"SELECT 1 WHERE {cols.pop()} IN ({', '.join(vals)})")[0].stmt.whereClause)
                    changed = True
                    continue
            out.append(a)
        if changed:
            stmt.whereClause = _make_and(out)
            return True, "OR of equalities on one column rewritten as IN list"
        return False, ""

    # ------------------------------------------------------------------ public
    def candidate_operators(self, masked_sql: str) -> list[RewriteProposal]:
        """Each applicable operator applied ALONE (RL low-level REWRITE_SQL actions)."""
        out = []
        for rule in REWRITE_OPERATORS:
            p = self.apply(masked_sql, [rule])
            if p is not None:
                out.append(p)
        return out

    def apply(self, masked_sql: str, rules: list[str] | None = None) -> RewriteProposal | None:
        rules = rules or REWRITE_OPERATORS
        try:
            stmt = parse_sql(masked_sql)[0].stmt
        except Exception:  # noqa: BLE001
            return None
        if not isinstance(stmt, ast.SelectStmt) or stmt.larg is not None:
            return None
        work = copy.deepcopy(stmt)
        applied, notes = [], []
        fns = {"correlated_count_subquery_to_left_join": self._rule_correlated_count,
               "cast_date_equality_to_range": self._rule_cast_date,
               "remove_duplicate_predicates": self._rule_dedupe,
               "or_equalities_to_in_list": self._rule_or_to_in}
        for r in rules:
            ok, note = fns[r](work)
            if ok:
                applied.append(r)
                notes.append(note)
        if not applied:
            return None
        rewritten = _deparse(work)
        # identifiers: the rewrite may only use pseudonyms that already appear in the input
        new_idents = {m.group(0) for m in PSEUDONYM_RE.finditer(rewritten)} - {m.group(0) for m in PSEUDONYM_RE.finditer(masked_sql)}
        report = diff(masked_sql, rewritten)
        verified_all, problems = True, []
        for r in applied:
            ok, probs = verify(r, report)
            verified_all &= ok
            problems += probs
        if new_idents:
            verified_all = False
            problems.append("rewrite introduced identifiers not present in the original query")
        report["verification_problems"] = sorted(set(problems))
        screens = [risk_screen(masked_sql, rewritten, r) for r in applied]
        order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
        report["risk_screening"] = max(screens, key=lambda x: order[x["risk"]])
        return RewriteProposal(rewritten_masked_sql=rewritten, rules_applied=applied,
                               reasoning="; ".join(notes), semantic_diff=report,
                               semantics_verified=verified_all, requires_human_review=True)

    def needs_rewrite(self, masked_sql: str) -> bool:
        return self.apply(masked_sql) is not None

    def validate_llm_rewrite(self, masked_sql: str, llm_sql: str) -> RewriteProposal | None:
        """LLM-proposed SQL: must parse, be a single SELECT, use only known pseudonyms and keep
        tables/outputs/order/group/limit. Always requires human review."""
        try:
            stmts = parse_sql(llm_sql)
        except Exception:  # noqa: BLE001
            return None
        if len(stmts) != 1 or not isinstance(stmts[0].stmt, ast.SelectStmt):
            return None
        known = {m.group(0) for m in PSEUDONYM_RE.finditer(masked_sql)}
        used = {m.group(0) for m in PSEUDONYM_RE.finditer(llm_sql)}
        report = diff(masked_sql, llm_sql)
        problems = []
        if used - known:
            problems.append("unknown identifiers")
        for k in ("tables_preserved", "output_column_count_preserved"):
            if not report[k]:
                problems.append(k.replace("_preserved", "_changed"))
        for k in ("group_by_changed", "order_by_changed", "limit_changed", "distinct_changed"):
            if report[k]:
                problems.append(k)
        report["verification_problems"] = problems
        report["risk_screening"] = risk_screen(masked_sql, llm_sql)
        if "unknown identifiers" in problems:
            return None
        return RewriteProposal(rewritten_masked_sql=_deparse(stmts[0].stmt), rules_applied=["llm_proposal"],
                               reasoning="LLM-proposed rewrite (structurally checked; semantics NOT proven)",
                               semantic_diff=report, semantics_verified=False, requires_human_review=True,
                               source="llm")
