import pandas as pd, re, sys
path = r".\downloads\1765351239-demo-audio-data.csv"
print('Reading:', path)
df = pd.read_csv(path)
print('\nCOLUMNS:', list(df.columns))
print('\nFIRST 10 ROWS:')
print(df.head(10).to_string(index=False))

# pick first data column (skip index-like first col)
cols = list(df.columns)
first = cols[0]
sample = df[first].dropna().astype(str).head(20).tolist()
is_index = all(s.strip().isdigit() for s in sample if s.strip()!='')
data_col = cols[1] if is_index and len(cols) > 1 else cols[0]
print('\nUsing data column:', data_col)

def clean_num(x):
    s = str(x).strip()
    s = s.replace(',', '')
    s = re.sub(r'[^\d\.\-+]', '', s)
    if s=='':
        return None
    return float(s)

nums = [clean_num(x) for x in df[data_col].tolist()]
nums = [n for n in nums if n is not None]
print('\nParsed numeric count:', len(nums))
print('First 20 parsed numeric values:', nums[:20])
total = sum(nums)
print('\nSUM_all:', total)
if abs(total - round(total)) < 1e-9:
    print('SUM_all (int):', int(round(total)))
