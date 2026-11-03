"""Specification-driven self-checks of the deliverables (no model call, no checker access).

The instruction text usually fixes the output contract precisely: JSON templates with typed placeholders
(`"sigma": <float>`, `"smile_present": <bool>`, `"n_prices": 21`), CSV column lists ("Columns: `T, K, ...`"),
row counts ("21 rows"), sort orders ("sorted by T then K ascending").  Hidden graders assert exactly these
things, so after every clean run we compare the files against the parsed contract:

    hard issues  -> fed back to the model as validation problems (missing keys/columns, wrong JSON types,
                    wrong fixed counts, wrong row counts)
    soft warnings-> shown to the model in the review step only (NaNs, implausible magnitudes, extra columns)

Everything here is best effort and conservative: when a template cannot be parsed, nothing is checked.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_JSON_BLOCK_RE = re.compile(r"```(?:json|javascript|js)?[ \t]*\r?\n(.*?)```", re.S)
_PLACEHOLDER_RE = re.compile(r"\"?<([^<>\"\n]{1,80})>\"?")
_FILE_RE = re.compile(r"[\w\-./]+\.(?:json|csv|tsv|parquet|pqt|txt|html|png|jpg|jpeg|svg|pdf|xlsx|md|npy|npz|pkl)\b", re.I)
_COUNT_KEY_RE = re.compile(r"(^|_)(n|num|number|count|total|size|len|length|seed|n_\w+)(_|$)|_count$|^n_|^num_", re.I)


# ------------------------------------------------------------------------------------------
# Template parsing
# ------------------------------------------------------------------------------------------

_TYPE_WORDS = {
    "bool": "bool", "boolean": "bool", "int": "int", "integer": "int", "float": "float", "double": "float",
    "number": "float", "numeric": "float", "decimal": "float", "str": "string", "string": "string",
    "list": "list", "array": "list", "object": "object", "dict": "object",
}


def _placeholder_type(text: str) -> str:
    """Type implied by a placeholder such as <int>, <float, 6 dp>, <array[string]>, <bull|bear>, <YYYY-MM-DD>.

    Only the leading word decides (descriptions like <local vol at K among interior strikes> are untyped), so
    free-text placeholders never produce false type errors."""
    t = text.strip().lower()
    words = re.findall(r"[a-z]+", t)
    if not words:
        return "any"
    if words[0] in _TYPE_WORDS:
        return _TYPE_WORDS[words[0]]
    if re.fullmatch(r"[a-z_]+(\s*\|\s*[a-z_]+)+", t):          # enumerations: bull|bear, true|false
        return "bool" if set(words) <= {"true", "false"} else "string"
    if "yyyy" in t or words[0] in ("iso", "date", "datetime", "timestamp"):
        return "string"
    return "any"


def _template_to_json(block: str) -> Any | None:
    s = block
    # prose stand-ins: `{ ... same structure ... }`, `true/false`, `"a" or "b"`, `"a" | "b" | "c"`
    s = re.sub(r"\{\s*(\.\.\.|…)[^{}]*\}", "\"__T:object__\"", s)
    s = re.sub(r"\btrue\s*/\s*false\b|\bfalse\s*/\s*true\b|<\s*true\s*\|\s*false\s*>", "\"__T:bool__\"", s, flags=re.I)
    s = re.sub(r"\"[^\"\n]*\"(\s*(\bor\b|\|)\s*\"[^\"\n]*\")+", "\"__T:string__\"", s)
    # strip whole-line // comments first; trailing comments are removed after the placeholders are replaced
    s = re.sub(r"(?m)^\s*//.*$", "", s)
    # ellipsis entries ("...", "…", "// ..."), inside lists or as whole lines
    s = re.sub(r"(?m)^\s*(\"?\.\.\.\"?|…)\s*,?\s*$", "", s)
    s = re.sub(r",\s*(\"?\.\.\.\"?|…)\s*,", ",", s)
    s = re.sub(r",\s*(\"?\.\.\.\"?|…)\s*(?=[\]}])", "", s)
    s = re.sub(r"(?<=[\[{])\s*(\"?\.\.\.\"?|…)\s*,", "", s)
    s = re.sub(r"(?m)^\s*\"[^\"]*\"\s*:\s*(\.\.\.|…)\s*,?\s*$", "", s)
    # typed placeholders -> sentinel strings
    nullable = set()

    def repl(m: re.Match) -> str:
        inner = m.group(1)
        typ = _placeholder_type(inner)
        if "null" in inner.lower() or "none" in inner.lower():
            nullable.add(typ)
            typ = typ + "|null"
        return f"\"__T:{typ}__\""

    s = _PLACEHOLDER_RE.sub(repl, s)
    s = re.sub(r"(?m)(?<=[,{\[\]}0-9\"a-z_])\s+//.*$", "", s)
    s = re.sub(r"(?m)(?<=[,{\[\]}0-9\"a-z_])\s+#\s.*$", "", s)
    # bare words used as values (true/false/null are fine); things like `float` or `int` without brackets
    s = re.sub(r":\s*(float|int|integer|bool|boolean|str|string|number)\s*(?=[,}\]\n])", lambda m: f": \"__T:{_placeholder_type(m.group(1))}__\"", s)
    # trailing commas
    s = re.sub(r",(\s*[}\]])", r"\1", s)
    # python literals
    s = re.sub(r"\bTrue\b", "true", s)
    s = re.sub(r"\bFalse\b", "false", s)
    s = re.sub(r"\bNone\b", "null", s)
    s = re.sub(r"\bNaN\b", "\"__T:float__\"", s)
    try:
        return json.loads(s)
    except Exception:
        pass
    # last resort: the block starts with an object/array but has trailing prose -> cut at the balanced end
    stripped = s.lstrip()
    for opener, closer in (("{", "}"), ("[", "]")):
        if not stripped.startswith(opener):
            continue
        i = s.find(opener)
        depth = 0
        for j in range(i, len(s)):
            if s[j] == opener:
                depth += 1
            elif s[j] == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(s[i:j + 1])
                    except Exception:
                        break
    return None


def _is_sentinel(v: Any) -> str | None:
    if isinstance(v, str) and v.startswith("__T:") and v.endswith("__"):
        return v[4:-2]
    return None


@dataclass
class JsonSpec:
    file: str
    template: Any


@dataclass
class CsvSpec:
    file: str
    columns: list[str] = field(default_factory=list)
    exact_order: bool = False
    n_rows: int | None = None
    sort_by: list[str] = field(default_factory=list)
    allows_nan: bool = False


@dataclass
class SpecContract:
    json_specs: dict[str, JsonSpec] = field(default_factory=dict)
    csv_specs: dict[str, CsvSpec] = field(default_factory=dict)

    def summary(self) -> str:
        bits = []
        for f, js in self.json_specs.items():
            keys = list(js.template.keys())[:12] if isinstance(js.template, dict) else ["<array>"]
            bits.append(f"{f}: keys {keys}")
        for f, cs in self.csv_specs.items():
            bits.append(f"{f}: cols {cs.columns[:12]}" + (f" rows={cs.n_rows}" if cs.n_rows else "") + (f" sort={cs.sort_by}" if cs.sort_by else ""))
        return "; ".join(bits)


def _basename(name: str) -> str:
    return name.strip("`'\" ").rsplit("/", 1)[-1]


def parse_contract(instruction: str, deliverable_names: list[str] | None = None) -> SpecContract:
    contract = SpecContract()
    try:
        _parse_json_templates(instruction, contract)
    except Exception:
        pass
    try:
        _parse_csv_specs(instruction, contract)
    except Exception:
        pass
    if deliverable_names:
        wanted = {_basename(n) for n in deliverable_names}
        contract.json_specs = {k: v for k, v in contract.json_specs.items() if k in wanted}
        contract.csv_specs = {k: v for k, v in contract.csv_specs.items() if k in wanted}
    return contract


def _parse_json_templates(instruction: str, contract: SpecContract) -> None:
    for m in _JSON_BLOCK_RE.finditer(instruction):
        block = m.group(1)
        if "<" not in block and not re.search(r"\b(float|int|bool|string)\b", block):
            continue  # an illustrative example rather than a template
        pre = instruction[max(0, m.start() - 700): m.start()]
        names = [_basename(n) for n in _FILE_RE.findall(pre) if n.lower().endswith(".json")]
        if not names:
            continue
        fname = names[-1]
        tpl = _template_to_json(block)
        if tpl is None or not isinstance(tpl, (dict, list)):
            continue
        if fname in contract.json_specs:
            # several blocks for the same file (e.g. per-section) -> merge top-level keys when both are objects
            old = contract.json_specs[fname].template
            if isinstance(old, dict) and isinstance(tpl, dict):
                merged = dict(old)
                merged.update(tpl)
                tpl = merged
        contract.json_specs[fname] = JsonSpec(fname, tpl)


_COL_KW_RE = re.compile(
    r"(?im)(?:^\s*(?:[-*•]\s*)?(?:\*\*)?(?:columns?|header|fields?)(?:\*\*)?\s*(?:\([^)]*\))?\s*(?::|=|must be|are|-|—)"
    r"|\bwith\s+(?:the\s+)?(?:following\s+|exact\s+|these\s+)?columns?\s*(?:\([^)]*\))?\s*:?"
    r"|\bcolumns?\s*(?:\([^)]*\))?\s*:)"
)


def _split_header(line: str) -> list[str]:
    parts = [x.strip().strip("`") for x in line.split(",")]
    if len(parts) >= 2 and all(re.fullmatch(r"[A-Za-z_][\w\-.%]*", x) for x in parts):
        return parts
    return []


def _extract_columns(text: str) -> tuple[list[str], bool]:
    """Column names from a 'Columns:' statement: inline backticks, a fenced header line, a backticked
    comma list spanning lines, a bullet list or a markdown table."""
    cols: list[str] = []
    exact = bool(re.search(r"exact(ly)?\s+(this|that|the)?\s*order|in\s+(this|that)\s+order|in\s+order\b", text, re.I))
    m = _COL_KW_RE.search(text)
    if not m:
        return cols, exact
    rest = text[m.end():]
    line_end = rest.find("\n")
    first_line = (rest if line_end < 0 else rest[:line_end]).strip()
    first_line = re.sub(r"\([^()]*\)", "", first_line)          # drop parenthetical annotations
    first_line = re.split(r"\.\s+[A-Z]", first_line)[0]         # stop at the end of the sentence
    ticks = re.findall(r"`([^`\n]+)`", first_line)
    for t in ticks:
        t = t.strip()
        if "," in t:
            cols.extend(_split_header(t))
        elif re.fullmatch(r"[A-Za-z_][\w\-.% ]{0,60}", t):
            cols.append(t)
    if not ticks and first_line and not first_line.startswith("```"):
        cols.extend(_split_header(first_line.rstrip(".")))
    if not cols and line_end >= 0:
        lines = rest[line_end + 1:].splitlines()
        i = 0
        while i < len(lines) and i < 3 and not lines[i].strip():
            i += 1
        if i < len(lines):
            s0 = lines[i].strip()
            if s0.startswith("```"):
                j = i + 1
                while j < len(lines) and not lines[j].strip():
                    j += 1
                if j < len(lines) and not lines[j].strip().startswith("```"):
                    cols = _split_header(lines[j].strip())
            elif "`" in s0 and re.fullmatch(r"(`[^`]+`\s*[,;]?\s*(and\s*)?)+\.?", s0):
                for ln in lines[i:]:
                    st = ln.strip()
                    if "`" in st and re.fullmatch(r"(`[^`]+`\s*[,;]?\s*(and\s*)?)+\.?", st):
                        cols.extend(x.strip() for x in re.findall(r"`([^`]+)`", st))
                    else:
                        break
            elif re.match(r"^(?:[-*•]|\d+[.)])\s*`", s0):
                for ln in lines[i:]:
                    st = ln.strip()
                    bm = re.match(r"^(?:[-*•]|\d+[.)])\s*`([^`]+)`", st)
                    if bm:
                        cols.append(bm.group(1).strip())
                    elif st:
                        break
            elif s0.startswith("|"):
                for ln in lines[i:]:
                    st = ln.strip()
                    if not st.startswith("|"):
                        break
                    if set(st.replace("|", "").strip()) <= set("-: "):
                        continue
                    tm = re.match(r"^\|\s*`?([A-Za-z_][\w\-.% ]*?)`?\s*\|", st)
                    if not tm:
                        continue
                    name = tm.group(1).strip()
                    if name.lower() in ("column", "columns", "field", "fields", "name", "key", "column name", "col"):
                        continue
                    cols.append(name)
    cols = [c for c in cols if re.fullmatch(r"[A-Za-z_][\w\-.%]*(?: [\w\-.%]+){0,2}", c)]
    seen = set()
    cols = [c for c in cols if not (c in seen or seen.add(c))]
    return cols, exact


def _parse_csv_specs(instruction: str, contract: SpecContract) -> None:
    mentions = [(m.start(), _basename(m.group(0))) for m in _FILE_RE.finditer(instruction) if m.group(0).lower().endswith((".csv", ".tsv"))]
    if not mentions:
        return
    seen_first: dict[str, int] = {}
    for pos, name in mentions:
        # region: from this mention to the next mention of a *different* file (or 1200 chars)
        end = min(len(instruction), pos + 1200)
        for p2, n2 in mentions:
            if p2 > pos and n2 != name:
                end = min(end, p2)
                break
        region = instruction[pos:end]
        spec = contract.csv_specs.get(name) or CsvSpec(name)
        cols, exact = _extract_columns(region)
        if cols and (not spec.columns or len(cols) > len(spec.columns)):
            spec.columns = cols
            spec.exact_order = exact or spec.exact_order
        rm = re.search(r"(?<![\w.~])(\d{1,6})\s+rows?\b", region)
        if rm and not re.search(r"(at least|up to|about|approximately|roughly|~|max(imum)?|min(imum)?|more than|fewer than|less than|per|each)\s+\d{1,6}\s+rows?", region, re.I):
            # only when the count is stated for the output (not "input has 504 rows")
            before = region[max(0, rm.start() - 80): rm.start()].lower()
            if "input" not in before and "given" not in before and "provided" not in before:
                spec.n_rows = int(rm.group(1))
        sm = re.search(r"sorted?\s+by\s+`?([A-Za-z_][\w]*)`?(?:\s*(?:,|then|and)\s*`?([A-Za-z_][\w]*)`?)?(?:\s*(?:,|then|and)\s*`?([A-Za-z_][\w]*)`?)?\s*(ascending|asc|increasing)?", region, re.I)
        if sm:
            spec.sort_by = [g for g in sm.groups()[:3] if g]
            if re.search(r"descending|desc\b|decreasing", region[sm.start(): sm.end() + 20], re.I):
                spec.sort_by = []  # only ascending orders are verified
        if re.search(r"\bnan\b|\bnull\b|\bmissing\b|empty string", region, re.I):
            spec.allows_nan = True
        if spec.columns or spec.n_rows or spec.sort_by:
            contract.csv_specs[name] = spec
            seen_first.setdefault(name, pos)


# ------------------------------------------------------------------------------------------
# Checking
# ------------------------------------------------------------------------------------------

def _type_ok(expected: str, actual: Any) -> tuple[bool, str]:
    """Return (ok, severity) where severity in {'hard','soft'} when not ok."""
    base, _, null = expected.partition("|")
    if actual is None:
        return (True, "") if null == "null" or base == "any" else (False, "hard")
    if base == "bool":
        return (isinstance(actual, bool), "hard")
    if base == "int":
        if isinstance(actual, bool):
            return False, "hard"
        if isinstance(actual, int):
            return True, ""
        if isinstance(actual, float) and math.isfinite(actual) and float(actual).is_integer():
            return False, "soft"
        return False, "hard"
    if base == "float":
        if isinstance(actual, bool):
            return False, "hard"
        return (isinstance(actual, (int, float)), "hard")
    if base == "string":
        return (isinstance(actual, str), "hard")
    if base == "list":
        return (isinstance(actual, list), "hard")
    if base == "object":
        return (isinstance(actual, dict), "hard")
    return True, ""


def _check_json(tpl: Any, actual: Any, path: str, hard: list[str], soft: list[str], depth: int = 0, in_list: bool = False) -> None:
    if depth > 6 or len(hard) > 20:
        return
    sent = _is_sentinel(tpl)
    if sent is not None:
        ok, sev = _type_ok(sent, actual)
        if not ok:
            msg = f"{path}: expected {sent.replace('|null', ' or null')}, got {_short(actual)}"
            (hard if sev == "hard" else soft).append(msg)
        return
    if isinstance(tpl, dict):
        if not isinstance(actual, dict):
            hard.append(f"{path}: expected an object, got {type(actual).__name__}")
            return
        dyn_keys = [k for k in tpl if _is_sentinel(k) is not None or re.fullmatch(r"<[^>]+>", k) or k.startswith("<")]
        fixed = [k for k in tpl if k not in dyn_keys]
        missing = [k for k in fixed if k not in actual]
        # keys whose template value is a plain literal string ("note": "posts remaining after ...") are
        # documentation in most task templates, not required output
        descriptive = [k for k in missing if isinstance(tpl[k], str) and _is_sentinel(tpl[k]) is None]
        if descriptive:
            soft.append(f"{path}: template also shows {descriptive[:5]} (descriptive fields?)")
            missing = [k for k in missing if k not in descriptive]
        if missing:
            if fixed and len(missing) >= max(2, len(fixed) * 0.6) and depth > 0:
                soft.append(f"{path}: keys {missing[:8]} not found (dynamic keys?)")
            else:
                hard.append(f"{path}: missing keys {missing[:10]}")
        extra = [k for k in actual if k not in tpl]
        if extra and not dyn_keys and depth == 0:
            soft.append(f"{path}: extra keys {extra[:8]} (not in the task's template)")
        for k in fixed:
            if k in actual:
                _check_json(tpl[k], actual[k], f"{path}.{k}", hard, soft, depth + 1, in_list)
        if dyn_keys:
            sub_tpl = tpl[dyn_keys[0]]
            others = [k for k in actual if k not in fixed]
            if not others:
                hard.append(f"{path}: expected entries keyed by {dyn_keys[0]}, found none")
            for k in others[:25]:
                _check_json(sub_tpl, actual[k], f"{path}[{k}]", hard, soft, depth + 1, True)
        return
    if isinstance(tpl, list):
        if not isinstance(actual, list):
            hard.append(f"{path}: expected an array, got {type(actual).__name__}")
            return
        if tpl and actual:
            # every element follows the first template element
            for i, item in enumerate(actual[:50]):
                _check_json(tpl[0], item, f"{path}[{i}]", hard, soft, depth + 1, True)
                if len(hard) > 20:
                    break
        elif tpl and not actual:
            soft.append(f"{path}: array is empty")
        return
    # literal in the template
    if isinstance(tpl, bool):
        if not isinstance(actual, bool):
            hard.append(f"{path}: expected a boolean, got {_short(actual)}")
        return
    if isinstance(tpl, (int, float)):
        if isinstance(actual, bool) or not isinstance(actual, (int, float)):
            hard.append(f"{path}: expected a number, got {_short(actual)}")
            return
        key = path.rsplit(".", 1)[-1]
        if not in_list and _COUNT_KEY_RE.search(key) and float(actual) != float(tpl):
            hard.append(f"{path}: the task fixes this value to {tpl}, got {_short(actual)}")
        return
    if isinstance(tpl, str):
        if not isinstance(actual, str) and not (in_list and actual is None):
            soft.append(f"{path}: template shows a string ({tpl!r}), got {_short(actual)}")
        return


def _short(v: Any, n: int = 40) -> str:
    s = json.dumps(v, default=str) if not isinstance(v, str) else repr(v)
    return s if len(s) <= n else s[: n - 3] + "..."


def check_outputs(out_dir: Path, contract: SpecContract) -> tuple[list[str], list[str]]:
    hard: list[str] = []
    soft: list[str] = []
    for fname, spec in contract.json_specs.items():
        p = out_dir / fname
        if not p.is_file():
            continue
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue  # validate_outputs already reports unparsable files
        h: list[str] = []
        s: list[str] = []
        try:
            _check_json(spec.template, obj, fname, h, s)
        except Exception:
            continue
        hard.extend(h[:10])
        soft.extend(s[:6])
    for fname, spec in contract.csv_specs.items():
        p = out_dir / fname
        if not p.is_file():
            continue
        try:
            _check_csv(p, spec, hard, soft)
        except Exception:
            continue
    return hard[:14], soft[:12]


def _check_csv(p: Path, spec: CsvSpec, hard: list[str], soft: list[str]) -> None:
    import pandas as pd

    sep = "\t" if p.suffix.lower() == ".tsv" else ","
    df = pd.read_csv(p, sep=sep)
    cols = [str(c) for c in df.columns]
    if spec.columns:
        missing = [c for c in spec.columns if c not in cols]
        if missing:
            ci = {c.lower(): c for c in cols}
            case_fixable = [c for c in missing if c.lower() in ci]
            if len(missing) >= max(2, int(0.75 * len(spec.columns))) and not case_fixable:
                soft.append(f"{p.name}: none of the task's columns {spec.columns[:8]} found (has {cols[:10]})")
            else:
                hard.append(f"{p.name}: missing columns {missing[:10]} (has {cols[:12]})")
        extra = [c for c in cols if c not in spec.columns]
        if extra and not missing:
            soft.append(f"{p.name}: extra columns {extra[:8]} not in the task's column list")
        if not missing and not extra and cols != spec.columns and len(cols) == len(spec.columns):
            (hard if spec.exact_order else soft).append(f"{p.name}: column order differs from the task's list {spec.columns[:10]}")
    if spec.n_rows is not None and len(df) != spec.n_rows:
        hard.append(f"{p.name}: has {len(df)} rows, the task says {spec.n_rows}")
    if spec.sort_by:
        keys = [c for c in spec.sort_by if c in cols]
        if keys and len(df) > 1:
            try:
                sub = df[keys]
                sorted_ok = sub.equals(sub.sort_values(keys, kind="mergesort").reset_index(drop=True)) if sub.index.equals(pd.RangeIndex(len(sub))) else True
                if not sorted_ok:
                    soft.append(f"{p.name}: rows are not sorted by {keys} ascending as the task says")
            except Exception:
                pass
    # NaNs and plausibility (soft)
    if not spec.allows_nan:
        nan_cols = [c for c in cols if df[c].isna().any()]
        if nan_cols:
            soft.append(f"{p.name}: NaN/empty values in columns {nan_cols[:8]}")
    num = df.select_dtypes("number")
    for c in num.columns:
        cl = str(c).lower()
        col = num[c].dropna()
        if col.empty:
            continue
        if ("prob" in cl or cl.endswith("_p") or cl in ("p_value", "pvalue")) and ((col < -1e-9).any() or (col > 1 + 1e-9).any()):
            soft.append(f"{p.name}.{c}: probability outside [0, 1]")
        elif ("corr" in cl or cl in ("rho", "kendall_tau", "spearman", "spearman_rho", "pearson")) and ((col < -1 - 1e-9).any() or (col > 1 + 1e-9).any()):
            soft.append(f"{p.name}.{c}: correlation outside [-1, 1]")
        elif (cl.endswith("price") or cl.endswith("_price") or cl in ("price", "premium")) and (col < -1e-9).any():
            soft.append(f"{p.name}.{c}: negative prices")
        elif ("implied_vol" in cl or cl in ("iv", "sigma", "vol", "volatility", "implied_volatility")) and ((col < 0).any() or (col > 5).any()):
            soft.append(f"{p.name}.{c}: volatility outside (0, 5)")
        if len(soft) > 12:
            break


if __name__ == "__main__":  # debug: python selfcheck.py <instruction.md> [out_dir]
    import sys

    text = Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
    c = parse_contract(text)
    print(c.summary())
    if len(sys.argv) > 2:
        h, s = check_outputs(Path(sys.argv[2]), c)
        print("HARD:", *h, sep="\n  ")
        print("SOFT:", *s, sep="\n  ")
