# Housing Affordability Prediction

This project involves the analysis and prediction of population-related statistics using a crime dataset. The goal is to visualize important trends, clean and process the dataset, and apply machine learning models to predict population values.


## Project Overview

The project is divided into the following key sections:
1. **Data Preprocessing**
   - Reading and understanding the data.
   - Handling missing and duplicate values.
   - Dropping irrelevant columns.
   - Visualization of relationships between various features.

2. **Data Visualization**
   - Scatter plots, bar charts, and heatmaps to explore the data.
   - State-level comparisons and trend visualizations.

3. **Machine Learning**
   - Several regression models were implemented and tested, including:
     - Linear Regression
     - Lasso Regression
     - Decision Tree
     - Random Forest
     - Gradient Boosting
     - Support Vector Regression
     - K-Nearest Neighbors
   - Model evaluation using:
     - R²
     - Mean Absolute Error (MAE)
     - Mean Squared Error (MSE).

4. **Hyperparameter Tuning**
   - Fine-tuning the best-performing model using GridSearchCV to improve accuracy.


## Features Analyzed

The following columns were used from the dataset:
- `state`
- `fold`
- `population`
- `numbUrban`
- `NumUnderPov`
- `murders`
- `NumImmig`
- `PctUsePubTrans`
- And others...


## Data Cleaning & Visualization

### Key Steps
- Dropped rows with anomalies (e.g., max population).
- Used correlation heatmaps to identify key relationships.
- Visualized trends like murders by state and housing-related statistics.

### Example Plots
- Scatter plots between features (`population` vs. `NumUnderPov`, etc.).
- Statewise distribution of murders.
- Correlation heatmap to highlight the relationships between numerical features.


## Machine Learning Models

- **Best Performing Model**: After testing multiple models, the `Lasso Regression` model performed best.
- **Hyperparameter Tuning**:
  - Conducted using GridSearchCV.
  - Optimal parameters: `alpha` and `max_iter`.


## Setup Instructions

### Prerequisites
Ensure you have Python installed with the following libraries:
- `numpy`
- `pandas`
- `seaborn`
- `matplotlib`
- `scikit-learn`

### Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/prantikm07/Housing-Affordability-Prediction-based-on-socio-economic-factors.git
   cd your-repo-name
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```


## Results

- **Best Model**: Lasso Regression
  - Mean Squared Error:  (to be updated after last run)
  - Mean Absolute Error:  (to be updated after last run)
  - R² Score:  (to be updated after last run)


## Contact

If you have any questions, feel free to contact me via:
- Email: [prantik25m@gmail.com](mailto:prantik25m@gmail.com)
- LinkedIn: [Prantik Mukhopadhyay](https://www.linkedin.com/in/prantikm07/)