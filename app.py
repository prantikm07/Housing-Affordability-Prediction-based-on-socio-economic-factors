from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Feature descriptions mapping
feature_descriptions = {
    'fold': 'Community Group',
    'numbUrban': 'Urban Population Count',
    'PctUsePubTrans': 'Public Transport Usage (%)',
    'rapes': 'Reported Rape Cases',
    'PctBSorMore': 'Population with Bachelor\'s Degree or Higher (%)',
    'burglPerPop': 'Burglaries per Population',
    'perCapInc': 'Per Capita Income',
    'agePct65up': 'Population Aged 65+ (%)',
    'nonViolPerPop': 'Non-Violent Crimes per Population',
    'PctLargHouseOccup': 'Large Households (%)',
    'pctWRetire': 'Retirement Income Population (%)',
    'FemalePctDiv': 'Divorced Females (%)'
}

# List of features in correct order (must match model's expected input)
FEATURES = list(feature_descriptions.keys())

# Affordability score statistics
AFFORDABILITY_STATS = {
    'min': 10005.0,
    'max': 3485398.0,
    'mean': 49834.58,
    '25_percentile': 14364.5,
    '50_percentile': 22785.5,
    '75_percentile': 42961.75
}

def normalize_to_percentage(score):
    """Normalize the affordability score to a percentage (0-100)."""
    return min(max((score - AFFORDABILITY_STATS['min']) / 
                  (AFFORDABILITY_STATS['max'] - AFFORDABILITY_STATS['min']) * 100, 0), 100)

def get_recommendation(score):
    """Provide a recommendation based on the affordability score."""
    if score < AFFORDABILITY_STATS['25_percentile']:
        return "Highly Affordable - Great choice! This area is very affordable."
    elif score < AFFORDABILITY_STATS['50_percentile']:
        return "Affordable - This is a good option for most people."
    elif score < AFFORDABILITY_STATS['75_percentile']:
        return "Moderately Affordable - Consider your budget carefully."
    else:
        return "Less Affordable - This area might be expensive for many."

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get input values in correct order
            input_data = [float(request.form[feat]) for feat in FEATURES]
            
            # Load model and predict
            with open('model.pkl', 'rb') as f:
                model = pickle.load(f)
            
            # Get raw prediction
            raw_prediction = model.predict([input_data])[0]
            
            # Normalize to percentage
            prediction_percentage = normalize_to_percentage(raw_prediction)
            
            # Get recommendation
            recommendation = get_recommendation(raw_prediction)
            
            return render_template('predict.html', 
                                raw_prediction=round(raw_prediction, 2),
                                prediction_percentage=round(prediction_percentage, 2),
                                recommendation=recommendation,
                                features=FEATURES,
                                feature_descriptions=feature_descriptions)
        
        except Exception as e:
            # Handle errors gracefully
            error_message = f"An error occurred: {str(e)}"
            return render_template('predict.html', 
                                error=error_message,
                                features=FEATURES,
                                feature_descriptions=feature_descriptions)
    
    # GET request - show empty form
    return render_template('predict.html', 
                         features=FEATURES,
                         feature_descriptions=feature_descriptions)

if __name__ == '__main__':
    app.run(debug=True)