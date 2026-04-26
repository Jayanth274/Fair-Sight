import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric

df = pd.read_csv('adult.csv').dropna()

# Encode categorical variables
encoders = {}
encoded_df = df.copy()
for col in encoded_df.columns:
    if encoded_df[col].dtype == 'object':
        le = LabelEncoder()
        encoded_df[col] = le.fit_transform(encoded_df[col].astype(str))
        encoders[col] = le

sensitive_col = 'sex'
target_col = 'income'

# Get privileged and unprivileged groups
privileged_val = float(encoded_df[sensitive_col].mode()[0])
unprivileged_val = float([v for v in encoded_df[sensitive_col].unique() if v != privileged_val][0])
privileged_groups = [{sensitive_col: privileged_val}]
unprivileged_groups = [{sensitive_col: unprivileged_val}]

# Get favorable label
favorable_label = 1.0
unfavorable_label = 0.0

dataset_true = BinaryLabelDataset(
    df=encoded_df,
    label_names=[target_col],
    protected_attribute_names=[sensitive_col],
    favorable_label=favorable_label,
    unfavorable_label=unfavorable_label
)

# Train model
X = encoded_df.drop(columns=[target_col])
y = encoded_df[target_col]
clf = RandomForestClassifier(n_estimators=10, random_state=42)
clf.fit(X, y)
preds = clf.predict(X)

dataset_pred = dataset_true.copy()
dataset_pred.labels = preds.reshape(-1, 1)

metric = ClassificationMetric(
    dataset_true, dataset_pred, 
    unprivileged_groups=unprivileged_groups,
    privileged_groups=privileged_groups
)

print('DI:', metric.disparate_impact())
print('SPD:', metric.statistical_parity_difference())
print('EOD:', metric.equal_opportunity_difference())
