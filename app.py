
import gradio as gr
import joblib
import numpy as np
from huggingface_hub import hf_hub_download
 
MODEL_REPO = "KubraParmak/water-potability-model"
ARTIFACT_FILE = "water_quality_artifact.joblib"
 
artifact_path = hf_hub_download(repo_id=MODEL_REPO, filename=ARTIFACT_FILE)
artifact = joblib.load(artifact_path)
 
models = artifact["models"]         
feature_names = artifact["feature_names"]
x_min = artifact["x_min"]
x_max = artifact["x_max"]
 
 
def predict(model_name, ph, hardness, solids, chloramines, sulfate,
            conductivity, organic_carbon, trihalomethanes, turbidity):
    raw = np.array([[ph, hardness, solids, chloramines, sulfate,
                      conductivity, organic_carbon, trihalomethanes, turbidity]])
 
    # Eğitimde kullanılan aynı min-max ölçekleme
    scaled = (raw - x_min) / (x_max - x_min)
 
    model = models[model_name]
    pred = model.predict(scaled)[0]
    proba = model.predict_proba(scaled)[0][1] if hasattr(model, "predict_proba") else None
 
    label = "✅ İçilebilir" if pred == 1 else "❌ İçilemez"
    if proba is not None:
        label += f"  (İçilebilirlik olasılığı: %{proba * 100:.1f})"
    return label
 
 
demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Radio(list(models.keys()), value=list(models.keys())[-1], label="Model seç"),
        gr.Number(label="pH", value=7.0),
        gr.Number(label="Hardness", value=200.0),
        gr.Number(label="Solids", value=20000.0),
        gr.Number(label="Chloramines", value=7.0),
        gr.Number(label="Sulfate", value=330.0),
        gr.Number(label="Conductivity", value=420.0),
        gr.Number(label="Organic Carbon", value=14.0),
        gr.Number(label="Trihalomethanes", value=66.0),
        gr.Number(label="Turbidity", value=4.0),
    ],
    outputs=gr.Textbox(label="Tahmin"),
    title="💧 Su İçilebilirlik Tahmini",
    description=(
        "Bir su örneğinin fiziksel/kimyasal özelliklerini girerek "
        "içilebilir olup olmadığını tahmin edin. Model, Kaggle'daki "
        "water-potability veri seti üzerinde eğitilmiştir."
    ),
)
 
if __name__ == "__main__":
    demo.launch()