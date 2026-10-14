# ⚙️ SensorMind AI: Asynchronous Predictive Maintenance Anomaly Classifier

SensorMind AI is a production-grade Industrial Artificial Intelligence and Machine Learning platform designed to predict manufacturing machinery downtime before it happens. By processing live IoT sensor telemetry streams, this engine runs multidimensional anomaly classification to calculate physical component breakdown risk factors in real time.

---

## 🧠 Core AI/ML Fields & Methodologies

This individual capstone project goes beyond simple software engineering loops to showcase core data science pillars:
- **Predictive Analytics (PdM):** Modeling structural degradation tracks using real historical hardware streams.
- **Supervised Anomaly Detection:** Evaluating multi-feature sensor data points against binary targets (`1` for failed state, `0` for functional operations).
- **Class Skew Management:** Implementing automated tree-balancing strategies (`class_weight='balanced'`) to successfully combat heavily imbalanced industrial datasets (97% healthy runtime vs. 3% rare failure occurrences).

---

## 📊 Advanced Physics Feature Engineering

To give the underlying classifier deep diagnostic capabilities, the pipeline transforms raw sensor fields into custom physics-based mathematical arrays:

1. **Thermal Dissipation Delta:**  
   \[\text{Delta} = \text{Process Temperature} - \text{Air Temperature}\]  
   *Justification:* Traces internal frictional core heat buildup that fails to dissipate into ambient surroundings, accelerating motor melt risk.
   
2. **Mechanical Power Index:**  
   \[\text{Power} = \text{Rotational Speed} \times \text{Torque}\]  
   *Justification:* Captures the structural mechanical workload strain forced onto the motor assembly over time.
   
3. **Wear Acceleration Strain Factor:**  
   \[\text{Strain} = \text{Tool Wear Time} \times \text{Torque}\]  
   *Justification:* Maps how structural load resistance heavily amplifies decay speeds when a degradation tool is near its expiration boundary.

---

## 📂 Project Architecture Map

```text
SensorMind_AI/
├── data/
│   └── predictive_maintenance.csv     # AI4I 2020 Industrial Kaggle Telemetry Dataset
├── templates/
│   └── dashboard.html                 # Sleek real-time interactive IoT control dashboard
├── train_engine.py                    # Preprocessing, dynamic column mapping & ML training script
├── main.py                            # Asynchronous FastAPI production API gateway web server
├── requirements.txt                   # Production package dependency versions manifest
└── maintenance_model.pkl              # Compiled Random Forest Ensemble Classifier (98% Accuracy)
```

---

## 🛠️ Step-by-Step System Workspace Execution

### 1. Initialize Pipeline Libraries
Install the structural data packages securely:
```bash
pip install -r requirements.txt
```

### 2. Run the Machine Learning Training Script
Process the telemetry dataset and compile the artificial intelligence asset:
```bash
python train_engine.py
```

### 3. Bring the Asynchronous FastAPI Portal Live
Boot up your high-speed production microservice environment:
```bash
uvicorn main:app --reload
```
*Open your web browser window to `http://127.0.0.1:8000` to interact with the active diagnostic control panel panel.*
