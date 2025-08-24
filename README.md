NASA MISSE Data Analysis App

This project is an interactive Streamlit web application designed to analyze data from NASA’s Materials International Space Station Experiment (MISSE). The app enables users to upload Excel datasets, map columns across different sheets, and visualize critical material degradation parameters such as erosion yield, mass loss, thickness, solar exposure, and AO fluence.

It also integrates regression and classification models for predictions, along with survival score analysis and visualization.

Features

Excel Upload & Sheet Selection: Import multi-sheet Excel files for analysis.

Dynamic Column Mapping: Select and map dataset columns interactively.

Data Visualization: Generate plots for erosion yield, mass loss, thickness, solar exposure, and AO fluence.

Predictive Modeling: Includes regression and classification models for analysis.

Survival Score Analysis: Compute and visualize material survival metrics.

Interactive Interface: Built using Streamlit for an intuitive and accessible experience.

Tech Stack

Python

Streamlit

Pandas

Scikit-learn

Matplotlib

Installation

Clone the repository and install the dependencies:

git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
pip install -r requirements.txt

Usage

Run the Streamlit app locally:

streamlit run app.py


Open your browser and go to http://localhost:8501 to use the app.

Deployment

The app is deployed on Streamlit Cloud for easy access.
You can view the live version here:
Deployed App Link

Project Structure
├── app.py              # Main Streamlit application  
├── NASAProject1.py     # Core logic, preprocessing, and ML models  
├── MergedExcel.xlsx    # Sample dataset (optional)  
├── requirements.txt    # Dependencies  
└── README.md           # Documentation  

Acknowledgments

Data sourced from NASA’s Materials International Space Station Experiment (MISSE).

Built for research and educational purposes to explore material degradation in space environments.
