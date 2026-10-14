from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import joblib
import numpy as np
import os

app = FastAPI(title="SensorMind AI: Industrial Predictive Maintenance Platform")
templates = Jinja2Templates(directory="templates")

# Load model engine safely
MODEL_PATH = 'maintenance_model.pkl'
model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

@app.get("/", response_class=HTMLResponse)
async def dashboard_home(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request, "prediction_text": None})

@app.post("/predict", response_class=HTMLResponse)
async def run_diagnostics(
    request: Request,
    air_temp: float = Form(...),
    proc_temp: float = Form(...),
    speed: float = Form(...),
    torque: float = Form(...),
    wear: float = Form(...)
):
    if model is None:
        return templates.TemplateResponse("dashboard.html", {"request": request, "prediction_text": "System Error: Model missing."})

    try:
        # Apply your exact custom physics equations to live streaming inputs
        thermal_dissipation_delta = proc_temp - air_temp
        mechanical_power_index = speed * torque
        wear_strain_factor = wear * torque
        
        # Build vector array
        input_data = np.array([[
            air_temp, proc_temp, speed, torque, wear,
            thermal_dissipation_delta, mechanical_power_index, wear_strain_factor
        ]])
        
        # Run classification check
        prediction = model.predict(input_data)
        
        if prediction == 1:
            result = "⚠️ CRITICAL ANOMALY: High failure probability signature detected. Schedule maintenance!"
            css_state = "danger-alert"
        else:
            result = "✅ SYSTEM HEALTHY: Machinery tracking variables are stable within normal limits."
            css_state = "success-alert"
            
        return templates.TemplateResponse("dashboard.html", {
            "request": request, 
            "prediction_text": result, 
            "alert_style": css_state
        })
    except Exception as e:
        return templates.TemplateResponse("dashboard.html", {"request": request, "prediction_text": f"Error: {str(e)}", "alert_style": "danger-alert"})
