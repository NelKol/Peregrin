import pandas as pd

df = {'X': [1, 2, 3, 4, 5], 'Y': [5, 4, 3, 2, 1], 'Track ID': [1, 1, 2, 2, 3]}

df = pd.DataFrame(df)

x = df[['X', 'Track ID']]
y = df[['Y', 'Track ID']]
# print(x)

tbl = (
    x.dropna(subset=['X'])
      .merge(
          y.dropna(subset=['Y']),
          on='Track ID', how='inner', sort=False  # sort=False is default, just explicit
      )
      .rename(columns={'X': '_x', 'Y': '_y'})
)

print(tbl)


from pandas.api.types import is_object_dtype, is_string_dtype, is_categorical_dtype

def _has_strings(s: pd.Series) -> bool:
    # pandas "string" dtype (pyarrow/python)
    if isinstance(s.dtype, pd.StringDtype):
        return s.notna().any()
    # categorical of strings?
    if isinstance(s.dtype, pd.CategoricalDtype):
        return isinstance(s.dtype.categories.dtype, pd.StringDtype) and s.notna().any()
    # numeric, datetime, bool, etc.
    if not is_object_dtype(s.dtype):
        return False
    # Fallback for object-dtype (mixed types): minimal Python loop over NumPy array
    arr = s.to_numpy(dtype=object, copy=False)
    return any(isinstance(v, (str, np.str_)) for v in arr)


def Normalize_01(df, col):
    """
    Normalize a column to the [0, 1] range.
    """
    s = pd.to_numeric(df[col], errors='coerce')
    if _has_strings(s):
        normalized = pd.Series(0.0, index=s.index, name=col)
    lo, hi = s.min(), s.max()
    if lo == hi:
        normalized = pd.Series(0.0, index=s.index, name=col)
    else:
        normalized = pd.Series((s - lo) / (hi - lo), index=s.index, name=col)
    return normalized  # <-- keeps index


data = pd.DataFrame({
    'A': [1, 2, 3, 4, 5],
    'B': [5, 4, 3, 2, 1],
    'C': ['a', 'b', 'c', 'd', 'e']
})

data.set_index('C', inplace=True, drop=True)

# print(data)
normalized = Normalize_01(data, 'A')
print(normalized)