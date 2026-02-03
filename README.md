# 🌍 Carbon Prediction System

![Python](https://img.shields.io/badge/-Python-blue?logo=python&logoColor=white)
![html](https://img.shields.io/badge/-HTML-yellow?logo=html&logoColor=yellow)
![css](https://img.shields.io/badge/-CSS-green?logo=css&logoColor=white)

## 📝 Overview

The Carbon Prediction System is a lightweight yet extensible framework designed to estimate and forecast carbon emissions from everyday activities, industrial processes, or organizational operations. By combining simple data inputs with modular prediction models, this project provides a clear, reproducible foundation for sustainability-focused applications.

Carbon emissions are one of the most pressing challenges of our time. Understanding and predicting them is critical for individuals, businesses, and policymakers who want to reduce their environmental footprint. This system is intentionally designed to be simple, transparent, and adaptable, making it suitable for:

    -> Students learning about climate data and sustainability.
    
    -> Researchers prototyping emission models.
    
    -> Developers integrating carbon prediction into dashboards, apps, or APIs.
    
    -> Organizations exploring lightweight tools for environmental reporting.
    
    -> The project emphasizes clarity over complexity, ensuring that anyone can run predictions, visualize results, and extend the system with minimal effort.

## 🛠️ Tech Stack

- 🐍 Python

## 📦 Key Dependencies

```
Flask
pandas
numpy
scikit-learn
tensorflow
joblib
```

## 📁 Project Structure

``` 
.
├── app.py
├── train_models.py
├── backend.py
├── requirements.txt
├── CO2 Emissions_Canada.csv (you can use your own dataset)
├── Data Description.csv
├── models
│   ├── (consists of ann, rf and svr model files)
├── static
│   └── style.css
└── templates
    └── index.html
    └── result.html
```

## 🛠️ Development Setup

### Python Setup
1. Install Python (v3.9+ recommended)
2. Create a virtual environment: `python -m venv venv`
3. Activate the environment:
   - Windows: `venv\Scripts\activate`
   - Unix/MacOS: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. add the dataset in the project folder your are running make sure your dataset has the parameters as in the input side (or use the dataset given in this project)
6. You can also refer the dataset description in the `data description.csv` file 
7. Run the `train_models.py` to check weather all the models are trained or not.
8. Run the `backend.py` to check the accuracy score of all the models (Random Forest, Support Vector Regression and ANN)
9. Finally run the `app.py` to run the frontend and give all the respective details and you can obtain the output.


## 👥 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork** the repository
2. **Clone** your fork: `git clone https://github.com/ambrose-kutti/Carbon-Prediction-System.git`
3. **Create** a new branch: `git checkout -b feature/your-feature`
4. **Commit** your changes: `git commit -am 'Add some feature'`
5. **Push** to your branch: `git push origin feature/your-feature`
6. **Open** a pull request

