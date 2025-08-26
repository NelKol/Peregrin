import re
import math
import pandas as pd

def _pick_encoding(path, encodings=("utf-8", "cp1252", "latin1", "iso8859_15")):
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False if enc != "utf-8" else True)
        except UnicodeDecodeError:
            continue


def _is_numeric_like(x):
    if pd.isna(x):
        return False
    s = str(x).strip()
    print(f"Checking if '{s}' is numeric-like")
    # numeric, scientific, or percentage looks like data
    return bool(re.fullmatch(r"[+-]?(\d+(\.\d+)?|\.\d+)([eE][+-]?\d+)?%?", s))

def _detect_header_depth(sample_df, threshold=0.5):
    """Return number of header rows before data begins."""
    for i in range(len(sample_df)):
        row = sample_df.iloc[i].tolist()
        n = len(row)
        numericish = sum(_is_numeric_like(v) for v in row)
        print(f"Row {i}: {n} total, {numericish} numeric-ish")
        # consider it a data row if at least half look numeric-ish
        if n > 0 and numericish >= math.ceil(threshold * n):
            return i  # header rows are 0..i-1
    # If we never hit a data-looking row, assume first row is header
    return 1

def _clean_levels(levels):
    # Replace μm -> microns and tidy whitespace
    return [ ("" if pd.isna(x) else str(x)).replace("(μm)", "(microns)").strip()
             for x in levels ]

def _flatten_columns(cols):
    out = []
    for tup in cols:
        parts = []
        for p in tup:
            if not p or str(p).startswith("Unnamed:"):
                continue
            clean = str(p).replace("(µm)", "(microns)").replace("(μm)", "(microns)").strip()
            parts.append(clean)
        # remove duplicates while preserving order
        seen = set()
        uniq = [x for x in parts if not (x in seen or seen.add(x))]
        out.append(" ".join(uniq).strip())
    return out


def load_csv_multiheader(path, keep_multiindex=False):
    enc = _pick_encoding(path)
    print(f"Using encoding: {enc}")
    # peek a few rows to detect header depth
    sample = pd.read_csv(path, encoding=enc, header=None, nrows=50, low_memory=False)
    header_rows = _detect_header_depth(sample)
    print(f"Detected {header_rows} header rows")
    # read with all header rows
    if header_rows <= 0:
        header_rows = 1
    header_idx = list(range(header_rows))
    print(f"Reading CSV with header rows: {header_idx}")
    df = pd.read_csv(path, encoding=enc, header=header_idx, low_memory=False)

    # Clean header levels
    if isinstance(df.columns, pd.MultiIndex):
        new_levels = list(zip(*[ _clean_levels(level) for level in df.columns.levels ]))
        # Remap each column via level values
        mapper = {}
        for col in df.columns:
            levels = tuple(_clean_levels(col))
            mapper[col] = levels
        df.columns = pd.MultiIndex.from_tuples([mapper[c] for c in df.columns])

        if not keep_multiindex:
            df.columns = _flatten_columns(df.columns)
    else:
        df.columns = [c.replace("(μm)", "(microns)").strip() for c in df.columns]

    return df

# ---- usage ----
# df = load_csv_multiheader(r"C:\Users\modri\Desktop\python\Peregrin\Peregrin\test data\2025_03_31 NEU_Hoechst_FaDu_spots_6F-01.csv", True)              # flattened headers
# df = load_csv_multiheader(r"C:\path\to\your.csv", True)        # keep MultiIndex

# print(df)

print(pd.read_csv(r"C:\Users\modri\Desktop\python\Peregrin\Peregrin\test data\2025_03_31 NEU_Hoechst_FaDu_spots_6F-01.csv", encoding="cp1252", low_memory=False))

print(_pick_encoding(r"C:\Users\modri\Desktop\python\Peregrin\Peregrin\test data\2025_03_31 NEU_Hoechst_FaDu_spots_6F-01.csv"))


value = None
print(type(value))