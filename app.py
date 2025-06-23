# app.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import shap
import joblib
from flask import Flask, request, render_template, redirect, url_for
import base64
from io import BytesIO

app = Flask(__name__)

# Create necessary folders
if not os.path.exists('static'):
    os.makedirs('static')

# Global model and scaler
model = None
scaler = None
feature_names = ['gdp_growth', 'fdi', 'inflation', 'education_spending', 'ease_business', 'business_inflation']

# Generate sample data
def generate_sample_data(num_countries=150):
    np.random.seed(42)
    data = {
        'country': [f'Country_{i}' for i in range(num_countries)],
        'region': np.random.choice(['Africa', 'Asia', 'Europe', 'Americas'], num_countries),
        'income_group': np.random.choice(['Low', 'Middle', 'High'], num_countries, p=[0.3, 0.5, 0.2]),
        'gdp_growth': np.random.uniform(-5, 10, num_countries),
        'fdi': np.random.uniform(0, 10, num_countries),
        'inflation': np.random.uniform(0, 30, num_countries),
        'education_spending': np.random.uniform(1, 8, num_countries),
        'ease_business': np.random.uniform(0, 100, num_countries),
    }
    
    # Simulate youth unemployment (target)
    factors = (
        0.3 * data['education_spending'] + 
        0.25 * (100 - data['ease_business']) +
        0.2 * data['inflation'] - 
        0.15 * data['gdp_growth'] - 
        0.1 * data['fdi']
    )
    data['youth_unemployment'] = np.clip(5 + factors + np.random.normal(0, 3, num_countries), 2, 40)
    
    return pd.DataFrame(data)

# Initialize model
def initialize_model():
    global model, scaler
    
    model_path = 'xgboost_unemployment_model.pkl'
    scaler_path = 'feature_scaler.pkl'
    
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
    else:
        print("Training model...")
        data = generate_sample_data()
        
        # Preprocessing
        data['business_inflation'] = data['ease_business'] / (data['inflation'] + 1)
        X = data[feature_names]
        y = data['youth_unemployment']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=data['region'], random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        
        # Train model
        model = XGBRegressor(random_state=42)
        model.fit(X_train, y_train)
        
        # Save model
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)

# Generate visual explanations as base64 images
def generate_explanations(input_features):
    # Create explainer
    explainer = shap.TreeExplainer(model)
    
    # Generate SHAP values
    shap_values = explainer.shap_values(input_features)
    
    # Create plots
    plt.ioff()  # Turn off interactive mode
    
    # Feature importance plot
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.barh(feature_names, model.feature_importances_)
    ax1.set_title('Feature Importance')
    ax1.set_xlabel('Importance Score')
    plt.tight_layout()
    
    # Save to buffer
    buf1 = BytesIO()
    fig1.savefig(buf1, format='png', bbox_inches='tight')
    plt.close(fig1)
    buf1.seek(0)
    feature_importance_img = base64.b64encode(buf1.read()).decode('utf-8')
    
    # SHAP force plot for this prediction
    fig2 = plt.figure(figsize=(10, 4))
    shap.force_plot(
        explainer.expected_value, 
        shap_values[0], 
        input_features[0], 
        feature_names=feature_names,
        matplotlib=True,
        show=False
    )
    plt.tight_layout()
    
    # Save to buffer
    buf2 = BytesIO()
    fig2.savefig(buf2, format='png', bbox_inches='tight')
    plt.close(fig2)
    buf2.seek(0)
    shap_force_img = base64.b64encode(buf2.read()).decode('utf-8')
    
    return feature_importance_img, shap_force_img

# Generate policy recommendations
def generate_recommendations(data, prediction):
    recommendations = []
    
    if data['education_spending'] < 4 and prediction > 15:
        recommendations.append("Increase education spending to at least 4% of GDP")
    
    if data['ease_business'] < 60 and prediction > 12:
        recommendations.append("Improve business regulations to boost entrepreneurship")
    
    if data['inflation'] > 15:
        recommendations.append("Implement inflation control measures")
    
    if data['fdi'] < 2 and prediction > 10:
        recommendations.append("Develop targeted FDI incentives in high-employment sectors")
    
    if not recommendations:
        return ["Current economic policies appear balanced"]
    
    return recommendations

# Initialize the model
initialize_model()

@app.route('/')
def home():
    # Simple HTML form without template file
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>SDG #8: Youth Unemployment Predictor</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            .card { border-radius: 15px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); margin-bottom: 20px; }
            .header { background: linear-gradient(135deg, #1a6fc4 0%, #0d47a1 100%); color: white; padding: 30px 0; }
            .form-label { font-weight: 500; }
            .footer { background-color: #f8f9fa; padding: 20px 0; margin-top: 30px; }
        </style>
    </head>
    <body>
        <div class="header">
            <div class="container text-center">
                <h1>Youth Unemployment Predictor</h1>
                <p class="lead">Supporting UN Sustainable Development Goal #8: Decent Work & Economic Growth</p>
            </div>
        </div>
        
        <div class="container my-4">
            <div class="card p-4">
                <h2 class="text-center mb-4">Economic Indicators Input</h2>
                <form action="/predict" method="POST">
                    <div class="row g-3">
                        <div class="col-md-6">
                            <label class="form-label">GDP Growth (%)</label>
                            <input type="number" step="0.1" class="form-control" name="gdp_growth" required
                                   min="-10" max="20" value="2.5">
                        </div>
                        <div class="col-md-6">
                            <label class="form-label">Foreign Direct Investment (% of GDP)</label>
                            <input type="number" step="0.1" class="form-control" name="fdi" required
                                   min="0" max="20" value="3.2">
                        </div>
                        <div class="col-md-6">
                            <label class="form-label">Inflation Rate (%)</label>
                            <input type="number" step="0.1" class="form-control" name="inflation" required
                                   min="0" max="50" value="7.8">
                        </div>
                        <div class="col-md-6">
                            <label class="form-label">Education Spending (% of GDP)</label>
                            <input type="number" step="0.1" class="form-control" name="education_spending" required
                                   min="1" max="10" value="4.1">
                        </div>
                        <div class="col-md-12">
                            <label class="form-label">Ease of Business Score (0-100)</label>
                            <input type="number" step="1" class="form-control" name="ease_business" required
                                   min="0" max="100" value="65">
                            <small class="form-text">Higher score = better business regulations</small>
                        </div>
                    </div>
                    <div class="text-center mt-4">
                        <button type="submit" class="btn btn-primary btn-lg px-5">Predict Unemployment</button>
                    </div>
                </form>
            </div>
        </div>
        
        <div class="footer">
            <div class="container text-center">
                <p>Developed for UN Sustainable Development Goals | Machine Learning for Social Good</p>
            </div>
        </div>
    </body>
    </html>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        data = {
            'gdp_growth': float(request.form['gdp_growth']),
            'fdi': float(request.form['fdi']),
            'inflation': float(request.form['inflation']),
            'education_spending': float(request.form['education_spending']),
            'ease_business': float(request.form['ease_business'])
        }
        
        # Calculate derived feature
        data['business_inflation'] = data['ease_business'] / (data['inflation'] + 1e-5)
        
        # Prepare features
        features = np.array([data[feature] for feature in feature_names]).reshape(1, -1)
        
        # Scale features
        scaled_features = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(scaled_features)[0]
        
        # Generate policy recommendations
        recommendations = generate_recommendations(data, prediction)
        
        # Generate visual explanations
        feature_importance_img, shap_force_img = generate_explanations(scaled_features)
        
        # Create results page
        return f'''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Prediction Results - SDG #8</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
            <style>
                .card {{ border-radius: 15px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); margin-bottom: 20px; }}
                .header {{ background: linear-gradient(135deg, #1a6fc4 0%, #0d47a1 100%); color: white; padding: 20px 0; }}
                .result-card {{ background: #f8f9fa; border-left: 5px solid #0d47a1; }}
                .recommendation {{ border-left: 3px solid #28a745; padding-left: 15px; margin-bottom: 10px; }}
                .explanation-img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 4px; padding: 5px; background: white; }}
            </style>
        </head>
        <body>
            <div class="header text-center">
                <h1>Youth Unemployment Prediction Results</h1>
            </div>
            
            <div class="container my-4">
                <div class="card result-card p-4">
                    <h2 class="text-center">Predicted Youth Unemployment Rate</h2>
                    <div class="display-3 text-center text-primary my-3">{round(prediction, 1)}%</div>
                    <p class="text-center">Based on your economic indicators input</p>
                </div>
                
                <div class="card p-4">
                    <h3>Policy Recommendations</h3>
                    <div class="mt-3">
                        {"".join(f'<div class="recommendation"><strong>Recommendation:</strong> {rec}</div>' for rec in recommendations)}
                    </div>
                </div>
                
                <div class="card p-4">
                    <h3>Your Input Summary</h3>
                    <div class="table-responsive mt-3">
                        <table class="table table-bordered">
                            <tr><th>GDP Growth:</th><td>{data['gdp_growth']}%</td></tr>
                            <tr><th>FDI (% GDP):</th><td>{data['fdi']}%</td></tr>
                            <tr><th>Inflation Rate:</th><td>{data['inflation']}%</td></tr>
                            <tr><th>Education Spending:</th><td>{data['education_spending']}% of GDP</td></tr>
                            <tr><th>Ease of Business:</th><td>{data['ease_business']}/100</td></tr>
                        </table>
                    </div>
                </div>
                
                <div class="card p-4">
                    <h3>Model Explanation</h3>
                    <div class="row mt-4">
                        <div class="col-md-6 mb-4">
                            <h4>Feature Importance</h4>
                            <p>Which factors most influence youth unemployment predictions:</p>
                            <img src="data:image/png;base64,{feature_importance_img}" class="explanation-img">
                        </div>
                        <div class="col-md-6 mb-4">
                            <h4>Prediction Explanation</h4>
                            <p>How each feature impacted this specific prediction:</p>
                            <img src="data:image/png;base64,{shap_force_img}" class="explanation-img">
                        </div>
                    </div>
                </div>
                
                <div class="text-center mt-4">
                    <a href="/" class="btn btn-primary btn-lg">Make Another Prediction</a>
                </div>
            </div>
            
            <div class="footer">
                <div class="container text-center">
                    <p>UN Sustainable Development Goal #8: Promote sustained, inclusive and sustainable economic growth</p>
                </div>
            </div>
        </body>
        </html>
        '''
    
    except Exception as e:
        # Simple error page
        return f'''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Error - SDG #8 Predictor</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
            <style>
                .error-container {{ margin-top: 100px; }}
            </style>
        </head>
        <body>
            <div class="container error-container">
                <div class="card border-danger">
                    <div class="card-header bg-danger text-white">
                        <h2>Application Error</h2>
                    </div>
                    <div class="card-body">
                        <h4 class="text-danger">Something went wrong</h4>
                        <div class="alert alert-danger">
                            <strong>Error Details:</strong>
                            <p>{str(e)}</p>
                        </div>
                        <a href="/" class="btn btn-primary mt-3">Back to Home</a>
                    </div>
                </div>
            </div>
        </body>
        </html>
        '''

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)