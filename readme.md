# SPP XAI - Smartphone Price Prediction

A machine learning project that predicts the price of a smartphone based on its specifications. Explainability artifacts (SHAP / LIME outputs) have been removed from this repository; the app focuses on prediction and similar-product recommendations.

This project is built as a Flask web application, where users can enter smartphone details such as RAM, storage, battery, camera, display type, OS, processor brand, and connectivity features. The application predicts an estimated market price and shows which features are influencing the result.

---

## Project Overview

The goal of this project is to combine:

- Predictive modeling for smartphone pricing
- User-friendly web interface
- Explainable AI for transparent decision-making
- Similar-product recommendations based on price tolerance

In simple terms, this project answers two questions:

1. How much should this smartphone cost?
2. Why did the model give that price?

---

## Why This Project Matters

Traditional price prediction models often act like black boxes. A user sees a price output, but not the reasons behind it.

This project focuses on reliable price prediction and user-facing recommendations. Earlier analysis used SHAP and LIME, but generated XAI artifacts are not included in this repository.

---

## Key Features

- Smartphone price prediction using a Gradient Boosting Regressor
- Input form for key smartphone parameters
- Price estimate in Indian Rupees (₹)
- Similar-price smartphone recommendations
-- Prediction output and similar-price recommendations
- Modern Flask UI for interactive use

---

## Tech Stack

- Python
- Flask
- Pandas
- NumPy
- Scikit-learn
- Joblib
-- (Note) SHAP and LIME were used during development; their generated files are not bundled here.
- Matplotlib
- HTML / CSS / JavaScript

---

## Project Structure

```text
SPP XAI/
├── static/                         # UI assets (styles, example outputs)
│   └── lime.html                   # optional example LIME HTML (if present)
├── templates/                      # Front-end templates
│   └── index.html                  # main web UI
├── Procfile                        # Gunicorn entrypoint for deployment
├── app.py                          # Flask application and prediction logic
├── model_features.joblib           # Expected model feature columns
├── model_setup_with_xai.py         # Training script (XAI generation removed from repo)
|── readme.md                       # Project README
├── requirements.txt                # Python dependencies
├── smartphone_price_model.joblib   # Trained model artifact
├── smartphones_data3.csv           # Smartphone dataset used for training and recommendations

```

---

## Dataset

The project uses a smartphone dataset containing columns such as:

- Model Name
- brand_name
- RAM
- storage
- Battery_capacity
- primery_rear_camera
- primary_front_camera
- refresh_rate(hz)
- has_5g
- has_fast_charging
- OS
- processor_brand
- display_types
- Price

This dataset is used to train the prediction model and provide realistic recommendation comparisons.

---

## Machine Learning Model

The project uses a Gradient Boosting Regressor, which is well suited for structured tabular data like smartphone specifications.

### Training workflow

1. Load the smartphone dataset
2. Clean and convert numeric values
3. Encode categorical variables such as brand, OS, processor, and display type
4. Create a log-transformed target for price prediction
5. Split data into training and test sets
6. Train the GradientBoostingRegressor
7. Evaluate performance using RMSE and R-squared
8. Save the trained model and feature names

The model predicts the price in log space and converts it back to actual rupees using `np.expm1()`.

---

## Explainable AI (XAI)

This project used SHAP and LIME during model analysis to help explain why the machine learning model produced each price estimate. The repository does not include the large generated image/HTML artifacts, but the application contains the runtime code to generate lightweight explanations and the training script to reproduce full artifacts locally.

### SHAP (SHapley Additive exPlanations)

- Implementation: We use `shap.TreeExplainer(gbr_model)` to compute SHAP values for the trained `GradientBoostingRegressor`.
- Runtime behavior: The Flask route calls `generate_shap_plot(input_df)` which:
	- computes SHAP values for the single input
	- attempts to render a SHAP waterfall plot (preferred)
	- if waterfall rendering fails (common on some hosts), falls back to a horizontal bar chart of absolute SHAP values
	- writes the plot to an in-memory PNG and returns a base64 data URI which is embedded in the page (`templates/index.html`).
- Deployment notes: headless servers require the `Agg` backend (the app sets `matplotlib.use('Agg')`). On Render or other hosted platforms, SHAP plotting may fail due to environment differences — in that case the code returns no image and logs the exception; see the Render runtime logs for the traceback.

How to reproduce SHAP artifacts locally:

1. Create and activate a Python environment and install dependencies:

```bash
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

2. Run the analysis script to generate static artifacts (if implemented in `model_setup_with_xai.py`):

```bash
python model_setup_with_xai.py
```

Or run the Flask app locally and submit the form to generate and view the SHAP plot in the browser:

```bash
python app.py
```

If you want static PNGs saved to the repo for demonstration, modify `generate_shap_plot()` to write to `static/shap_<id>.png` instead of returning base64, then commit the files.

### LIME (Local Interpretable Model-agnostic Explanations)

- Implementation: We initialize `LimeTabularExplainer` with the model-aligned training data produced by `model_setup_with_xai.py` preprocessing.
- Runtime behavior: `get_lime_data(input_df, predicted_price)` calls `explain_instance(...)`, extracts the top features and weights, maps internal feature names to friendly labels, and builds both a short narrative and a ranked list shown in the UI. The LIME outputs are rendered directly as HTML elements in `templates/index.html`.
- Reproducing LIME HTML: to save full LIME HTML explanations, add `exp.save_to_file('static/lime.html')` (or similar) where the LIME explanation is created.

### Practical notes and troubleshooting

- If SHAP plotting fails in deployment, check the runtime logs for a Python exception. The app is defensive: when plotting fails it returns `None` and the page hides the image.
- Ensure `requirements.txt` is installed during your build (it already includes `shap`, `lime`, and `matplotlib`).
- For reproducible, shareable explanation assets, generate them locally (or in a CI job) and add them to `static/` or `xai_outputs/` before deploying.

If you want, I can add a small `make_xai_outputs.py` utility that runs the preprocessing, computes SHAP/LIME for a sample set, and writes files to `static/xai_outputs/` so they are viewable on any deployment.

---

## Web Application Flow

The Flask app works in the following way:

1. User selects smartphone specifications in the web form
2. Input values are converted into model-ready feature vectors
3. The trained model predicts a price
4. The app compares the predicted value with similar phones in the dataset
5. The app computes a prediction and shows similar phones from the dataset
6. The final result is displayed on the webpage

---

## How the App Works

### Frontend

The user interface is built in `templates/index.html` and includes:

- OS dropdown
- Processor brand dropdown
- Display type dropdown
- RAM and storage inputs
- Battery capacity field
- Camera fields
- Refresh rate field
- Toggle options like 5G and fast charging

### Backend

The main logic is handled in `app.py`, which includes:

- loading the trained model and feature metadata
- preparing user input for prediction
- calling the model
- generating recommendations
	- preparing input and running the trained model for a price prediction
	- finding similar phones in the dataset for comparison
- rendering the final result page

---

## Important Files Explained

### `app.py`

This is the main Flask application. It:

- loads the trained model and dataset
- prepares user input
- predicts smartphone price
- finds similar phones in the same price range
- creates SHAP visuals
- builds the LIME explanation narrative
- renders the web page

### `model_setup_with_xai.py`

This file is the model-building script for the project. During development it also contained XAI-generation steps; those generated artifacts are not included in this repo. It covers:

- dataset preprocessing
- feature encoding
- train/test split
- model training
- evaluation metrics
	- saving the final model and features

### `smartphone_price_model.joblib`

This is the trained machine learning model saved using Joblib.

### `model_features.joblib`

This file stores the exact feature columns that the model expects. It ensures the same ordering when user input is passed to the model.

### `smartphones_data3.csv`

This is the dataset used to train and test the price prediction model.

### `templates/index.html`

This is the interactive frontend for the app. It provides the form and visual output layout.

### `static/`

This folder may store UI assets or generated example outputs such as LIME or SHAP HTML/PNG files. It is not required by the main Flask route logic.

### XAI artifacts

Generated XAI files are not bundled in this repository. If you need SHAP/LIME outputs, re-run the analysis steps in `model_setup_with_xai.py` and enable artifact generation.

### `DEPLOYMENT_GUIDE.md`

Contains notes about production deployment, platform choices, and roadmap for future scaling.

---

## Setup Instructions

### 1. Clone the project

```bash
git clone <your-repository-url>
cd "SPP XAI"
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

On Windows:

```bash
venv\Scripts\activate
```

On Linux/macOS:

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
python app.py
```

Then open the browser at:

```text
http://127.0.0.1:5000
```

---

## Example Usage

A user may enter:

- RAM: 8GB
- Storage: 128GB
- Battery: 5000mAh
- Rear Camera: 64MP
- Front Camera: 16MP
- OS: Android
- Display Type: AMOLED
- Processor Brand: Snapdragon
- 5G: Yes
- Fast Charging: Yes

The system will estimate a price such as ₹26,500 and show which features pushed the price up or down.

---

## Results and Outputs

The app provides:

- predicted smartphone price
- similar products in the same price range
- explanation of important features
-- similar-product recommendations
-- prediction estimate

This makes the solution more transparent and easier to trust than a plain regression model.

---

## Project Impact

This project can be useful for:

- smartphone buyers comparing market price ranges
- e-commerce price estimation
- market analysis and product benchmarking
- educational demonstration of explainable AI in machine learning

---

## Limitations

This is a solid demo project, but there are some practical limitations:

- the model depends on dataset quality and variety
- prices may vary by region, seller, and promotional discounts
- the app is best suited for local demo use rather than large-scale production deployment
- additional validation and deployment hardening would be needed for live production systems

---

## Future Improvements

Possible enhancements include:

- support for more smartphone brands and models
- improved preprocessing for missing values and inconsistent labels
- better model tuning and cross-validation
- deployment on Render / Heroku / cloud platform
- Docker support
- API endpoint for prediction integration
- user authentication and saved prediction history
- stronger production-ready deployment configuration

---

## Conclusion

The SPP XAI project provides a compact example of end-to-end price prediction and similar-product recommendations. XAI artifacts were used during analysis but are not included here to keep the repository focused on the deployed prediction service.

This makes it an excellent project for:

- learning ML model training
- working with Flask web apps
- building a lightweight prediction service

---

## License

This project is intended for educational and personal use unless otherwise specified by the repository owner.

---

## Author / Project Note

This project was developed as a Smartphone Price Prediction with XAI demonstration and is suitable for GitHub portfolio sharing, academic presentation, and further extension.
