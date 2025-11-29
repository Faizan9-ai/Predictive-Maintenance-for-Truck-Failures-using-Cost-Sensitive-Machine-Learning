import gradio as gr
import pandas as pd
import numpy as np
import joblib

# Load model
model = joblib.load("aps_best_model.pkl")

def predict_from_csv(file):
    try:
        # Read uploaded file
        df = pd.read_csv(file.name)

        # If target column exists, drop it
        if "class" in df.columns:
            df = df.drop("class", axis=1)

        # Replace "na" and convert to float
        df = df.replace("na", np.nan)
        df = df.astype(float)

        # Make predictions
        pred = model.predict(df)
        prob = model.predict_proba(df)[:, 1]

        # Prepare result
        output = pd.DataFrame({
            "predicted_class": pred,
            "failure_probability": prob
        })

        return output
    
    except Exception as e:
        return f"Error: {str(e)}"

# Gradio App
app = gr.Interface(
    fn=predict_from_csv,
    inputs=gr.File(label="Upload APS sensor CSV file"),
    outputs=gr.DataFrame(label="Predictions"),
    title="APS Failure Detection System",
    description="Upload a CSV file containing APS sensor readings. The model predicts if there is a failure.",
)

app.launch()
