from flask import Flask, request, jsonify
import joblib
from flask_cors import CORS
import traceback
import pandas as pd

# behavioral columns
behavioral_cols = [
    'Social Smile', 'Attention', 'Eye contact', 'Sitting behavio',
    'Hyperactivity', 'Echolalia', 'Recognition of parents',
    'Excessive crying', 'Restlessness', 'Temper tantrums',
    'Self-injurious behaviour (when young)', 'Head banging',
    'Vacant staring', 'Self-muttering', 'Stubborn', 'Laziness'
]

# Needed because the model pickle references it
def map_behavioral_cols(df):
    df_mapped = df.copy()
    mapping = {'P': 1, 'A': 0, 'NaN': 0}
    for col in behavioral_cols:
        df_mapped[col] = df_mapped[col].astype(str).map(mapping)
    return df_mapped

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # allow cross-origin requests

# Load the pipeline
model = joblib.load("therapy_recommendation_pipeline.pkl")


# 👇 Add this line to see exactly which feature names your model expects
if hasattr(model, "feature_names_in_"):
    print("Model expects features:", model.feature_names_in_)
else:
    print("⚠️ Model does not have feature_names_in_. Let's inspect:")
    try:
        print(model)
    except Exception as e:
        print("Error inspecting model:", e)

THERAPIES = [
    'Medical Consultation',
    'Psychology Consultation',
    'Speech and Audiology',
    'Physical Therapy',
    'Occupational Therapy',
    'Special Education'
]

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        symptoms = data.get("symptoms", {})
        chief_complaints = data.get("chief_complaints", "")

        # Create DataFrame with all expected columns
        df = pd.DataFrame([{
            col: 1 if symptoms.get(col, "No") == "Yes" else 0
            for col in behavioral_cols
        }])

        # Add the text column
        df["Chief_Complaints"] = chief_complaints
        # Now predict
        
        print("Incoming symptoms:", symptoms)
        print("DataFrame sent to model:")
        print(df)
        prediction = model.predict(df)[0]
        print("Raw model prediction:", prediction)

        # Convert to therapy names
        recommended_therapy = []
        if hasattr(prediction, "__iter__") and not isinstance(prediction, (str, bytes)):
            recommended_therapy = [THERAPIES[i] for i, val in enumerate(prediction) if int(val) == 1]
        elif isinstance(prediction, (int, float)):
            recommended_therapy = [THERAPIES[int(prediction)]]
        else:
            recommended_therapy = [str(prediction)]

        return jsonify({"recommended_therapy": recommended_therapy})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
