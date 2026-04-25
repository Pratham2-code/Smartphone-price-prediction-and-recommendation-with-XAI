from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

# =========================
# XAI IMPORTS
# =========================
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
import os

# Ensure static folder exists
os.makedirs("static", exist_ok=True)

# =========================
# CONFIG
# =========================
ALLOWED_PROCESSOR_BRANDS = ['snapdragon', 'mediatek', 'apple', 'unisoc', 'samsung']
PRICE_TOLERANCE_PERCENT = 0.10

# =========================
# LOAD MODEL & DATA
# =========================
try:
    gbr_model = joblib.load('smartphone_price_model.joblib')
    feature_names = joblib.load('model_features.joblib')

    df_raw = pd.read_csv('smartphones_data3.csv')

    unique_os = sorted(df_raw['OS'].unique().tolist())
    unique_processors = [p for p in sorted(df_raw['processor_brand'].unique().tolist()) if p in ALLOWED_PROCESSOR_BRANDS]
    unique_displays = sorted(df_raw['display_types'].unique().tolist())

except FileNotFoundError:
    print("ERROR: Required files not found.")
    exit()

# =========================
# INITIALIZE XAI
# =========================
shap_explainer = shap.TreeExplainer(gbr_model)

# Prepare training data for LIME (same encoding as model)
df_lime = df_raw.copy()
df_lime = df_lime.drop('Model Name', axis=1)

df_lime['RAM'] = df_lime['RAM'].astype(int)
df_lime['storage'] = df_lime['storage'].astype(int)
df_lime['primery_rear_camera'] = df_lime['primery_rear_camera'].astype(int)
df_lime['primary_front_camera'] = df_lime['primary_front_camera'].astype(int)

df_lime['has_5g'] = df_lime['has_5g'].fillna('No')
df_lime['has_5g'] = df_lime['has_5g'].apply(lambda x: 1 if x in ['Yes', 'yes'] else 0)
df_lime['has_fast_charging'] = df_lime['has_fast_charging'].apply(lambda x: 1 if x in ['Yes', 'yes'] else 0)

df_lime['refresh_rate(hz)'] = df_lime['refresh_rate(hz)'].fillna(60).astype(int)

df_lime = pd.get_dummies(df_lime.drop('Price', axis=1), drop_first=True)

# Align columns with model
df_lime = df_lime.reindex(columns=feature_names, fill_value=0)

lime_explainer = LimeTabularExplainer(
    training_data=df_lime.values,
    feature_names=feature_names,
    mode='regression',
    random_state=42
)

# =========================
# FLASK APP
# =========================
app = Flask(__name__)

# =========================
# PREPROCESS FUNCTION
# =========================
def prepare_user_input(form_data, feature_names):
    data = {feature: [0] for feature in feature_names}
    input_df = pd.DataFrame(data)

    input_df['RAM'] = int(form_data.get('ram') or 0)
    input_df['storage'] = int(form_data.get('storage') or 0)
    input_df['Battery_capacity'] = int(form_data.get('battery_capacity') or 0)
    input_df['primery_rear_camera'] = int(form_data.get('rear_camera') or 0)
    input_df['primary_front_camera'] = int(form_data.get('front_camera') or 0)
    input_df['refresh_rate(hz)'] = int(form_data.get('refresh_rate') or 0)

    input_df['has_fast_charging'] = 1 if form_data.get('fast_charging') == 'on' else 0
    input_df['has_5g'] = 1 if form_data.get('has_5g') == 'on' else 0

    os_col = f"OS_{form_data['os']}"
    if os_col in input_df.columns:
        input_df[os_col] = 1

    processor_col = f"processor_brand_{form_data['processor_brand']}"
    if processor_col in input_df.columns:
        input_df[processor_col] = 1

    display_col = f"display_types_{form_data['display_types']}"
    if display_col in input_df.columns:
        input_df[display_col] = 1

    return input_df[feature_names]

# =========================
# PREDICTION FUNCTION
# =========================
def predict_price(input_df):
    log_price = gbr_model.predict(input_df)[0]
    return np.expm1(log_price)

# =========================
# RECOMMENDATION FUNCTION
# =========================
def get_recommendations(predicted_price, df, tolerance):
    min_price = predicted_price * (1 - tolerance)
    max_price = predicted_price * (1 + tolerance)

    recommendations_df = df[
        (df['Price'] >= min_price) &
        (df['Price'] <= max_price)
    ].copy()

    recommendations_df['price_diff'] = np.abs(recommendations_df['Price'] - predicted_price)

    top_recommendations = recommendations_df.sort_values(by='price_diff').head(5)

    result = []
    for _, row in top_recommendations.iterrows():
        result.append({
            'name': row['Model Name'],
            'price': f"₹{row['Price']:,.0f}",
            'brand': row['brand_name']
        })

    return result

# =========================
# XAI FUNCTIONS
# =========================
def generate_shap_plot(input_df):
    shap_values = shap_explainer.shap_values(input_df)

    explanation = shap.Explanation(
        values=shap_values[0],
        base_values=shap_explainer.expected_value,
        data=input_df.iloc[0],
        feature_names=input_df.columns
    )

    plt.figure()
    shap.plots.waterfall(explanation, show=False, max_display=10)
    plt.tight_layout()

    filepath = "static/shap_plot.png"
    plt.savefig(filepath, dpi=120, bbox_inches='tight')
    plt.close()

    return "shap_plot.png"


def predict_original_price(X):
    return np.expm1(gbr_model.predict(X))


def get_lime_data(input_df, predicted_price):
    exp = lime_explainer.explain_instance(
        input_df.values[0],
        predict_original_price,
        num_features=6
    )
    
    name_map = {
        'RAM': 'RAM',
        'storage': 'Storage',
        'Battery_capacity': 'Battery',
        'primery_rear_camera': 'Rear Camera',
        'primary_front_camera': 'Front Camera',
        'refresh_rate(hz)': 'Refresh Rate',
        'has_fast_charging': 'Fast Charging',
        'has_5g': '5G Connectivity'
    }
    
    raw_explanations = exp.as_list()
    ranking = []
    pos_factors = []
    neg_factors = []
    
    for feature_cond, weight in raw_explanations:
        # 1. Identify the core feature name from the condition string
        # We look for our known feature names in the condition string
        core_name = "Unknown"
        for internal in feature_names:
            if internal in feature_cond:
                core_name = internal
                break
        
        # 2. Determine if it's a "Not" condition for categorical features
        is_categorical = any(x in core_name for x in ['OS_', 'processor_brand_', 'display_types_', 'brand_name_'])
        is_negated = is_categorical and ("<= 0" in feature_cond or " = 0" in feature_cond)
        
        # 3. Create friendly name
        friendly_name = core_name
        for internal, friendly in name_map.items():
            friendly_name = friendly_name.replace(internal, friendly)
        
        # Clean up prefixes
        for prefix in ['brand_name_', 'processor_brand_', 'display_types_', 'OS_']:
            friendly_name = friendly_name.replace(prefix, '')
            
        friendly_name = friendly_name.replace('display', '').strip().capitalize()
        
        # Add "Processor" or "OS" context
        if 'processor' in core_name.lower():
            friendly_name += " Processor"
        elif 'OS_' in core_name:
            friendly_name += " OS"

        # 4. Final Display Texts
        if is_negated:
            display_text_list = f"Not {friendly_name}"
            display_text_narrative = f"the absence of {friendly_name}"
        else:
            display_text_list = friendly_name
            display_text_narrative = f"your {friendly_name}"

        # 5. Populate lists
        ranking.append({
            'text': display_text_list,
            'impact': "increases" if weight > 0 else "decreases",
            'amount': f"₹{abs(weight):,.0f}",
            'is_positive': weight > 0
        })

        if weight > 0:
            pos_factors.append(display_text_narrative.lower())
        else:
            neg_factors.append(display_text_narrative.lower())

    # Build narrative
    narrative = f"Based on the specifications you provided, the predicted price of ₹{predicted_price:,.0f} is driven by a few key aspects. "
    if pos_factors:
        narrative += f"The value is primarily pushed higher by {', '.join(pos_factors[:-1]) + (' and ' + pos_factors[-1] if len(pos_factors) > 1 else pos_factors[0])}. "
    if neg_factors:
        narrative += f"On the other hand, the total remains balanced and more affordable due to {', '.join(neg_factors[:-1]) + (' and ' + neg_factors[-1] if len(neg_factors) > 1 else neg_factors[0])}."
    
    return narrative, ranking

# =========================
# ROUTES
# =========================
@app.route('/')
def index():
    return render_template(
        'index.html',
        oss=unique_os,
        processors=unique_processors,
        displays=unique_displays
    )


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Prepare input
        input_data = prepare_user_input(request.form, feature_names)

        # 2. Predict
        predicted_price = predict_price(input_data)

        # 3. Recommendations
        recommendations = get_recommendations(
            predicted_price, df_raw, PRICE_TOLERANCE_PERCENT
        )

        # 4. XAI
        shap_img = generate_shap_plot(input_data)
        lime_narrative, lime_ranking = get_lime_data(input_data, predicted_price)

        return render_template(
            'index.html',
            prediction_text=f"The estimated price is: ₹{predicted_price:,.0f}",
            recommendations=recommendations,
            shap_img=shap_img,
            lime_narrative=lime_narrative,
            lime_ranking=lime_ranking,
            oss=unique_os,
            processors=unique_processors,
            displays=unique_displays
        )

    except Exception as e:
        return render_template(
            'index.html',
            prediction_text=f"Error: {str(e)}",
            oss=unique_os,
            processors=unique_processors,
            displays=unique_displays
        )


if __name__ == '__main__':
    app.run(debug=True)