from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

app = Flask(__name__)

rf_model = pickle.load(open('random_forest_model.pkl', 'rb'))
scaler   = pickle.load(open('scalerr.pkl', 'rb'))
ohe      = pickle.load(open('ohe.pkl', 'rb'))
le_sex      = pickle.load(open('le_sex.pkl', 'rb'))
le_exercise = pickle.load(open('le_exercise.pkl', 'rb'))


class HeartModelV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.heart_mode = nn.Sequential(
            nn.Linear(in_features=18, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=1),
        )
    def forward(self, x):
        return self.heart_mode(x)
    
pytorch_model = HeartModelV1()
pytorch_model.load_state_dict(torch.load('pytorch_model.pth', map_location='cpu'))
pytorch_model.eval()

@app.route('/', methods=['GET', 'POST'])
def home():
    rf_prediction = None
    rf_confidence = None
    pt_prediction = None
    pt_confidence = None

    if request.method == 'POST':
        age            = int(request.form['age'])
        sex            = request.form['sex']
        chest_pain     = request.form['chest_pain']
        resting_bp     = int(request.form['resting_bp'])
        cholesterol    = int(request.form['cholesterol'])
        fasting_bs     = int(request.form['fasting_bs'])
        resting_ecg    = request.form['resting_ecg']
        max_hr         = int(request.form['max_hr'])
        exercise_angina = request.form['exercise_angina']
        oldpeak        = float(request.form['oldpeak'])
        st_slope       = request.form['st_slope']

        sex_encoded             = 1 if sex == 'M' else 0
        exercise_angina_encoded = 1 if exercise_angina == 'Y' else 0
        cat_input = pd.DataFrame([[chest_pain, resting_ecg, st_slope]],
                                  columns=['ChestPainType', 'RestingECG', 'ST_Slope'])
        ohe_encoded = ohe.transform(cat_input)

        numeric_input = np.array([[age, resting_bp, cholesterol, max_hr, oldpeak]])
        scaled_input  = scaler.transform(numeric_input)
        
        final_input = np.concatenate([
            [sex_encoded],
            scaled_input[0],
            [fasting_bs],
            [exercise_angina_encoded],
            ohe_encoded[0]
        ]).reshape(1, -1)

        rf_prediction = int(rf_model.predict(final_input)[0])
        rf_confidence = round(rf_model.predict_proba(final_input)[0][1] * 100, 1)

        input_tensor = torch.tensor(final_input, dtype=torch.float32)
        with torch.inference_mode():
            pt_logit      = pytorch_model(input_tensor).squeeze()
            pt_confidence = round(torch.sigmoid(pt_logit).item() * 100, 1)
            pt_prediction = 1 if pt_confidence > 50 else 0

    return render_template('index.html', rf_prediction=rf_prediction,
                           rf_confidence=rf_confidence,
                           pt_prediction=pt_prediction,
                           pt_confidence=pt_confidence)
    

if __name__ == '__main__':
    app.run(debug=True,use_reloader=False)