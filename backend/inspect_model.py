import joblib
import pandas as pd

# ✅ Step 1: Define the function BEFORE loading the model
def map_behavioral_cols(df):
    behavioral_cols = [
        'Social Smile', 'Attention', 'Eye contact', 'Sitting behavio',
        'Hyperactivity', 'Echolalia', 'Recognition of parents',
        'Excessive crying', 'Restlessness', 'Temper tantrums',
        'Self-injurious behaviour (when young)', 'Head banging',
        'Vacant staring', 'Self-muttering', 'Stubborn', 'Laziness'
    ]
    df_mapped = df.copy()
    mapping = {'P': 1, 'A': 0, 'NaN': 0}
    for col in behavioral_cols:
        df_mapped[col] = df_mapped[col].astype(str).map(mapping)
    return df_mapped

# ✅ Step 2: Load the model
model_path = "therapy_recommendation_pipeline.pkl"
model = joblib.load(model_path)

print("Model loaded successfully!")
print("Model type:", type(model))

# ✅ Step 3: Test prediction
behavioral_cols = [
    'Social Smile', 'Attention', 'Eye contact', 'Sitting behavio',
    'Hyperactivity', 'Echolalia', 'Recognition of parents',
    'Excessive crying', 'Restlessness', 'Temper tantrums',
    'Self-injurious behaviour (when young)', 'Head banging',
    'Vacant staring', 'Self-muttering', 'Stubborn', 'Laziness'
]

test_df = pd.DataFrame([{
    'Chief_Complaints': 'test input',
    **{col: 1 for col in behavioral_cols}
}])

prediction = model.predict(test_df)
print("\nRaw model prediction:", prediction)
