import os
import pickle
import numpy as np
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Load the SVM Model
MODEL_PATH = "SVM_model.pkl"
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
else:
    model = None

# Integrated HTML + Interactive CSS Animations
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SVM Prediction Dashboard</title>
    <style>
        :root {
            --bg-color: #0d1117;
            --card-bg: rgba(22, 27, 34, 0.85);
            --accent-glow: #6366f1;
            --accent-hover: #4f46e5;
            --text-main: #f0f6fc;
            --text-muted: #8b949e;
            --input-bg: #21262d;
            --border-color: #30363d;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }

        body {
            background: radial-gradient(circle at top left, #1e1b4b, var(--bg-color));
            color: var(--text-main);
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            padding: 40px 20px;
        }

        .container {
            background-color: var(--card-bg);
            backdrop-filter: blur(12px);
            border: 1px solid var(--border-color);
            padding: 40px;
            border-radius: 16px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.6);
            width: 100%;
            max-width: 600px;
            animation: fadeIn 0.8s ease-out;
        }

        @keyframes fadeIn {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        h2 {
            text-align: center;
            font-weight: 700;
            margin-bottom: 8px;
            letter-spacing: -0.5px;
            background: linear-gradient(135deg, #a5b4fc, #6366f1);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        p.subtitle {
            text-align: center;
            color: var(--text-muted);
            font-size: 14px;
            margin-bottom: 30px;
        }

        .grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
        }

        .form-group {
            display: flex;
            flex-direction: column;
        }

        .full-width {
            grid-column: span 2;
        }

        label {
            margin-bottom: 6px;
            font-size: 13px;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        input, select {
            width: 100%;
            padding: 12px;
            background-color: var(--input-bg);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            color: var(--text-main);
            font-size: 15px;
            outline: none;
            transition: all 0.3s ease;
        }

        input:focus, select:focus {
            border-color: var(--accent-glow);
            box-shadow: 0 0 10px rgba(99, 102, 241, 0.3);
        }

        button {
            width: 100%;
            padding: 14px;
            background: linear-gradient(135deg, #6366f1, #4f46e5);
            border: none;
            border-radius: 8px;
            color: #ffffff;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            margin-top: 20px;
            box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4);
        }

        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(99, 102, 241, 0.6);
        }

        button:active {
            transform: translateY(0);
        }

        .result-box {
            margin-top: 25px;
            padding: 18px;
            border-radius: 8px;
            text-align: center;
            font-weight: 700;
            font-size: 20px;
            background: rgba(99, 102, 241, 0.15);
            border: 1px solid var(--accent-glow);
            animation: pulseGlow 2s infinite alternate;
        }

        @keyframes pulseGlow {
            0% {
                box-shadow: 0 0 5px rgba(99, 102, 241, 0.2);
            }
            100% {
                box-shadow: 0 0 20px rgba(99, 102, 241, 0.6);
            }
        }
    </style>
</head>
<body>

<div class="container">
    <h2>SVM Prediction Model</h2>
    <p class="subtitle">Fill in the metrics below to generate a prediction</p>
    
    <form method="POST" action="/predict">
        <div class="grid">
            <div class="form-group">
                <label for="gender">Gender</label>
                <select id="gender" name="gender" required>
                    <option value="1">Male</option>
                    <option value="0">Female</option>
                </select>
            </div>

            <div class="form-group">
                <label for="age">Age</label>
                <input type="number" id="age" name="age" step="any" required placeholder="e.g. 20">
            </div>

            <div class="form-group">
                <label for="study_hours">Study Hours / Wk</label>
                <input type="number" id="study_hours" name="study_hours_per_week" step="any" required placeholder="e.g. 15">
            </div>

            <div class="form-group">
                <label for="attendance">Attendance Rate (%)</label>
                <input type="number" id="attendance" name="attendance_rate" step="any" required placeholder="e.g. 85">
            </div>

            <div class="form-group">
                <label for="parent_edu">Parent Education</label>
                <input type="number" id="parent_edu" name="parent_education" step="any" required placeholder="Level scale (e.g. 1-5)">
            </div>

            <div class="form-group">
                <label for="internet">Internet Access</label>
                <select id="internet" name="internet_access" required>
                    <option value="1">Yes</option>
                    <option value="0">No</option>
                </select>
            </div>

            <div class="form-group">
                <label for="extracurricular">Extracurricular</label>
                <select id="extracurricular" name="extracurricular" required>
                    <option value="1">Yes</option>
                    <option value="0">No</option>
                </select>
            </div>

            <div class="form-group">
                <label for="prev_score">Previous Score</label>
                <input type="number" id="prev_score" name="previous_score" step="any" required placeholder="e.g. 75">
            </div>

            <div class="form-group full-width">
                <label for="final_score">Final Score</label>
                <input type="number" id="final_score" name="final_score" step="any" required placeholder="e.g. 80">
            </div>
        </div>
        
        <button type="submit">Predict Outcome</button>
    </form>

    {% if prediction is not none %}
    <div class="result-box">
        Prediction Result: {{ prediction }}
    </div>
    {% endif %}
</div>

</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE, prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template_string(HTML_TEMPLATE, prediction="Error: SVM_model.pkl not found.")
    
    try:
        # Extract features from form in expected exact sequence
        features = [
            float(request.form.get('gender')),
            float(request.form.get('age')),
            float(request.form.get('study_hours_per_week')),
            float(request.form.get('attendance_rate')),
            float(request.form.get('parent_education')),
            float(request.form.get('internet_access')),
            float(request.form.get('extracurricular')),
            float(request.form.get('previous_score')),
            float(request.form.get('final_score'))
        ]
        
        input_data = np.array([features])
        prediction = model.predict(input_data)[0]
        
        return render_template_string(HTML_TEMPLATE, prediction=str(prediction))
    except Exception as e:
        return render_template_string(HTML_TEMPLATE, prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
