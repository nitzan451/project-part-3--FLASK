import pandas as pd
import pickle
from sklearn.linear_model import ElasticNetCV
from assets_data_prep import prepare_data

# שלב 1: טען את הנתונים
df = pd.read_csv("train.csv")  # שנה ל-train.xlsx אם צריך

# שלב 2: עיבוד מוקדם
df_prepared = prepare_data(df, dataset_type='train')

# שלב 3: פיצול ל-X ו-y
X = df_prepared.drop(columns='price')
y = df_prepared['price']

# שלב 4: הגדרת המודל עם cross-validation
model = ElasticNetCV(
    l1_ratio=0.1,                     # כמה L1 לעומת L2
    alphas=[0.01, 0.1, 1, 10],        # אלפא candidates
    cv=10,                            # 10 קיפולים
    max_iter=10000,
    random_state=42
)

# שלב 5: אימון המודל
model.fit(X, y)

# שלב 6: שמירה
with open("trained_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ המודל אומן ונשמר בהצלחה.")
