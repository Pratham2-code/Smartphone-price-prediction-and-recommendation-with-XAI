# SPP XAI - Smartphone Price Prediction with Explainable AI

A machine learning and explainable AI project that predicts the price of a smartphone based on its specifications and explains the prediction using SHAP and LIME.

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

This project improves trust and usability by using Explainable AI (XAI) methods:

- SHAP: explains feature contributions at both global and local levels
- LIME: explains a single prediction by approximating a local model around that input

That makes the predictions more understandable for users, researchers, and business stakeholders.

---

## Key Features

- Smartphone price prediction using a Gradient Boosting Regressor
- Input form for key smartphone parameters
- Price estimate in Indian Rupees (₹)
- Similar-price smartphone recommendations
- SHAP-based explanation for model behavior
- LIME explanation for individual prediction interpretation
- Modern Flask UI for interactive use

---

## Tech Stack

- Python
- Flask
- Pandas
- NumPy
- Scikit-learn
- Joblib
- SHAP
- LIME
- Matplotlib
- HTML / CSS / JavaScript

---

## Project Structure

```text
SPP XAI/
├── app.py                          # Flask application and prediction logic
├── model_setup_with_xai.py         # Training and XAI generation script
├── requirements.txt                # Python dependencies
├── Procfile                        # Gunicorn entrypoint for deployment
├── smartphones_data3.csv          # Smartphone dataset used for training and recommendations
├── smartphone_price_model.joblib   # Trained model artifact
├── model_features.joblib            # Expected model feature columns
├── static/                         # Optional generated static assets
│   ├── lime.html                   # Example LIME HTML output
│   └── shap_plot.png               # Example SHAP image output
├── templates/
│   └── index.html                  # Front-end user interface
├── xai_outputs/                    # Generated XAI explanation artifacts
│   ├── 1_builtin_feature_importance.png
│   ├── 2_shap_global_bar.png
│   ├── 3_shap_beeswarm.png
│   ├── 4_shap_waterfall_single_phone.png
│   ├── 5_shap_dependence_RAM.png
│   ├── 6_shap_dependence_storage.png
│   ├── 7_lime_single_phone.html
│   └── 7_lime_single_phone.png
└── documents/                      # Project documentation and presentation assets
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

This project uses two major explanation methods.

### 1. SHAP

SHAP explains both the overall importance of each feature and the influence of each feature on a single prediction.

Examples of SHAP outputs in the project:

- global feature importance bar chart
- beeswarm summary plot
- waterfall plot for one phone prediction
- dependence plots for RAM and storage

This helps answer questions like:

- Which feature is most important overall?
- How much did RAM increase the predicted price?
- Did 5G or a larger battery push the price upward?

### 2. LIME

LIME explains an individual prediction by building a simpler local model around the selected input.

It is especially useful for showing the most influential factors for one specific smartphone configuration.

The project saves LIME visualizations in the `xai_outputs` folder and also outputs an HTML explanation file.

---

## Web Application Flow

The Flask app works in the following way:

1. User selects smartphone specifications in the web form
2. Input values are converted into model-ready feature vectors
3. The trained model predicts a price
4. The app compares the predicted value with similar phones in the dataset
5. The app generates SHAP and LIME explanations
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
- generating SHAP explanation plots
- generating LIME explanation text/ranking
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

This file is the model-building and XAI-generation script. It is the training pipeline for the project. It covers:

- dataset preprocessing
- feature encoding
- train/test split
- model training
- evaluation metrics
- XAI plot generation
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

### `xai_outputs/`

This folder stores visual explanations generated during model training and analysis. These files are useful for demonstration and reporting, but they are not required for the deployed prediction service itself.

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
- visual SHAP plots
- LIME-based reasoning summary

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

The SPP XAI project is a complete end-to-end example of building a practical machine learning application that goes beyond simple prediction. It not only estimates smartphone prices but also explains why the model made that estimate.

This makes it an excellent project for:

- learning ML model training
- working with Flask web apps
- applying explainable AI concepts
- demonstrating real-world AI in a user-facing environment

---

## License

This project is intended for educational and personal use unless otherwise specified by the repository owner.

---

## Author / Project Note

This project was developed as a Smartphone Price Prediction with XAI demonstration and is suitable for GitHub portfolio sharing, academic presentation, and further extension.
