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

# HTML Template with Cute & Animated CSS Layout
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>✨ SVM Student Predictor ✨</title>
    <link href="https://fonts.googleapis.com/css2?family=Fredoka:wght@400;600;700&family=Nunito:wght@600;700;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-gradient: linear-gradient(135deg, #fbc2eb 0%, #a6c1ee 100%);
            --card-bg: rgba(255, 255, 255, 0.88);
            --accent-pink: #ff758c;
            --accent-purple: #a855f7;
            --accent-hover: #ff7eb3;
            --text-dark: #4a4a68;
            --input-bg: #f8f9fe;
            --border-soft: #e2e8f0;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Nunito', sans-serif;
        }

        body {
            background: var(--bg-gradient);
            background-size: 200% 200%;
            animation: gradientShift 10s ease infinite;
            color: var(--text-dark);
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            padding: 30px 15px;
        }

        @keyframes gradientShift {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        .container {
            background-color: var(--card-bg);
            backdrop-filter: blur(16px);
            border: 3px solid #ffffff;
            padding: 35px;
            border-radius: 28px;
            box-shadow: 0 15px 35px rgba(166, 193, 238, 0.5), 0 5px 15px rgba(255, 117, 140, 0.2);
            width: 100%;
            max-width: 580px;
            animation: floatIn 0.8s cubic-bezier(0.68, -0.55, 0.265, 1.55);
            position: relative;
        }

        @keyframes floatIn {
            0% {
                opacity: 0;
                transform: translateY(40px) scale(0.9);
            }
            100% {
                opacity: 1;
                transform: translateY(0) scale(1);
            }
        }

        /* Floating decoration badges */
        .decoration {
            position: absolute;
            font-size: 24px;
            animation: bounceSoft 3s ease-in-out infinite alternate;
        }
        .deco-1 { top: -15px; left: -15px; }
        .deco-2 { top: -15px; right: -15px; animation-delay: 1.5s; }

        @keyframes bounceSoft {
            0% { transform: translateY(0) rotate(0deg); }
            100% { transform: translateY(-10px) rotate(12deg); }
        }

        h2 {
            font-family: 'Fredoka', sans-serif;
            text-align: center;
            font-size: 28px;
            font-weight: 700;
            color: #5b468a;
            margin-bottom: 6px;
        }

        p.subtitle {
            text-align: center;
            color: #8c7ba6;
            font-size: 15px;
            font-weight: 700;
            margin-bottom: 25px;
        }

        .grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 14px;
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
            font-weight: 800;
            color: #6a539b;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        input, select {
            width: 100%;
            padding: 12px 16px;
            background-color: var(--input-bg);
            border: 2px solid var(--border-soft);
            border-radius: 16px;
            color: var(--text-dark);
            font-size: 15px;
            font-weight: 700;
            outline: none;
            transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        }

        input:focus, select:focus {
            border-color: var(--accent-pink);
            background-color: #ffffff;
            box-shadow: 0 0 0 4px rgba(255, 117, 140, 0.2);
            transform: scale(1.02);
        }

        button {
            width: 100%;
            padding: 15px;
            background: linear-gradient(135deg, var(--accent-pink), var(--accent-purple));
            border: none;
            border-radius: 20px;
            color: #ffffff;
            font-family: 'Fredoka', sans-serif;
            font-size: 18px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            margin-top: 22px;
            box-shadow: 0 8px 20px rgba(255, 117, 140, 0.4);
            position: relative;
            overflow: hidden;
        }

        button:hover {
            transform: translateY(-3px) scale(1.01);
            box-shadow: 0 12px 25px rgba(255, 117, 140, 0.6);
        }

        button:active {
            transform: translateY(1px) scale(0.98);
        }

        /* Result Popup Card */
        .result-box {
            margin-top: 25px;
            padding: 18px;
            border-radius: 18px;
            text-align: center;
            font-family: 'Fredoka', sans-serif;
            font-weight: 700;
            font-size: 22px;
            color: #fff;
            background: linear-gradient(135deg, #a855f7, #7c3aed);
            border: 2px solid #ffffff;
            box-shadow: 0 10px 20px rgba(124, 58, 237, 0.3);
            animation: popUp 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        }

        @keyframes popUp {
            0% {
                opacity: 0;
                transform: scale(0.5);
            }
            100% {
                opacity: 1;
                transform: scale(1);
            }
        }
    </style>
</head>
<body>

<div class="container">
    <div class="decoration deco-1">🌸</div>
    <div class="decoration deco-2">⭐</div>

    <h2>✨ SVM Predictor ✨</h2>
    <p class="subtitle">Fill in the details below to check prediction!</p>
    
    <form method="POST" action="/predict">
        <div class="grid">
            <div class="form-group">
                <label for="gender">Gender 👤</label>
                <select id="gender" name="gender" required>
                    <option value="1">Male</option>
                    <option value="0">Female</option>
                </select>
            </div>

            <div class="form-group">
                <label for="age">Age 🎂</label>
                <input type="number" id="age" name="age" step="any" required placeholder="e.g. 20">
            </div>

            <div class="form-group">
                <label for="study_hours">Study Hours 📚</label>
                <input type="number" id="study_hours" name="study_hours_per_week" step="any" required placeholder="e.g. 15">
            </div>

            <div class="form-group">
                <label for="attendance">Attendance 🎒</label>
                <input type="number" id="attendance" name="attendance_rate" step="any" required placeholder="e.g. 85%">
            </div>

            <div class="form-group">
                <label for="parent_edu">Parent Edu 🎓</label>
                <input type="number" id="parent_edu" name="parent_education" step="any" required placeholder="Scale 1-5">
            </div>

            <div class="form-group">
                <label for="internet">Internet 🌐</label>
                <select id="internet" name="internet_access" required>
                    <option value="1">Yes</option>
                    <option value="0">No</option>
                </select>
            </div>

            <div class="form-group">
                <label for="extracurricular">Activities 🎨</label>
                <select id="extracurricular" name="extracurricular" required>
                    <option value="1">Yes</option>
                    <option value="0">No</option>
                </select>
            </div>

            <div class="form-group">
                <label for="prev_score">Prev Score 📝</label>
                <input type="number" id="prev_score" name="previous_score" step="any" required placeholder="e.g. 75">
            </div>

            <div class="form-group full-width">
                <label for="final_score">Final Score 🎯</label>
                <input type="number" id="final_score" name="final_score" step="any" required placeholder="e.g. 80">
            </div>
        </div>
        
        <button type="submit">Predict Result 🔮</button>
    </form>

    {% if prediction is not none %}
    <div class="result-box">
        🎉 Prediction: {{ prediction }} 🎉
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
