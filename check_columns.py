import pandas as pd
from pathlib import Path
hp={'BIA-BIA_BMC','BIA-BIA_BMR','BIA-BIA_DEE','BIA-BIA_ECW','BIA-BIA_FFM','BIA-BIA_FFMI','BIA-BIA_FMI','BIA-BIA_Fat','BIA-BIA_ICW','BIA-BIA_LDM','BIA-BIA_LST','BIA-BIA_SMM','BIA-BIA_TBW','PCIAT-PCIAT_Total','cluster_id','is_outlier'}
root=Path('.')
train=pd.read_csv(root/'data/processed/train_clean.csv')
train_cols=[c for c in train.columns if c not in ('sii','id')]
train_cols_after=[c for c in train_cols if c not in hp]
try:
    test=pd.read_csv(root/'data/raw/test.csv')
except Exception as e:
    print('Cannot read test.csv:', e)
    raise SystemExit

test_cols=[c for c in test.columns if c not in ('id','sii')]
missing=sorted(set(train_cols_after)-set(test_cols))
extra=sorted(set(test_cols)-set(train_cols_after))
print('Train feature count after drop purity:', len(train_cols_after))
print('Test feature count:', len(test_cols))
print('Missing in test (first 50):', missing[:50])
print('Extra in test (first 50):', extra[:50])
