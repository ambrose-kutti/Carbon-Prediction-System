# 🍃 Carbon Emission Prediction System

<div align="center">

<!-- TODO: Add project logo (e.g., an icon related to CO2 or a chart) -->

[![GitHub stars](https://img.shields.io/github/stars/ambrose-kutti/Carbon-Prediction-System?style=for-the-badge)](https://github.com/ambrose-kutti/Carbon-Prediction-System/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/ambrose-kutti/Carbon-Prediction-System?style=for-the-badge)](https://github.com/ambrose-kutti/Carbon-Prediction-System/network)
[![GitHub issues](https://img.shields.io/github/issues/ambrose-kutti/Carbon-Prediction-System?style=for-the-badge)](https://github.com/ambrose-kutti/Carbon-Prediction-System/issues)
[![GitHub license](https://img.shields.io/github/license/ambrose-kutti/Carbon-Prediction-System?style=for-the-badge)](LICENSE) <!-- TODO: Add LICENSE file -->

**A lightweight, modular system for estimating and predicting carbon emissions using Machine Learning models.**

</div>

## 📖 Overview

The Carbon Emission Prediction System is a web-based application designed to provide estimations and predictions of carbon emissions. Built with Python and Flask, this project integrates various machine learning models (such as Artificial Neural Networks, Random Forest, and Support Vector Regression) to offer insights into environmental impact. It serves as an excellent resource for learning about ML model deployment, Flask web development, and environmental data analysis, emphasizing clarity and reproducibility.

## ✨ Features

-   **Carbon Emission Prediction**: Utilizes trained machine learning models to predict CO2 emissions based on input parameters.
-   **Interactive Web Interface**: A user-friendly web application built with Flask and Jinja2 templates for inputting data and viewing predictions.
-   **Multiple ML Models**: Supports various predictive models including Artificial Neural Networks (ANN), Random Forest, and Support Vector Regression (SVR).
-   **Modular Design**: Separates concerns into distinct modules for data processing, model training, backend logic, and application serving.
-   **Data-Driven Insights**: Leverages real-world CO2 emission data (e.g., Canada's CO2 Emissions dataset) for training and prediction.
-   **Easy Setup**: Designed for straightforward installation and local execution, perfect for prototyping and educational purposes.

## 🖥️ Screenshots

<!-- TODO: Add actual screenshots of the web application and its prediction results -->
Home page with Toggle modee
<img width="1274" height="571" alt="Screenshot 2026-03-06 130251" src="https://github.com/user-attachments/assets/a158f0cc-9800-467a-b411-cca8a9f0afd8" />
Screenshot 2
path-to-screenshot-of-prediction-results.png

## 🛠️ Tech Stack

**Backend & Machine Learning:**
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)

**Frontend:**
![HTML5](https://img.shields.io/badge/HTML5-E34F26?style=for-the-badge&logo=html5&logoColor=white)
![CSS3](https://img.shields.io/badge/CSS3-1572B6?style=for-the-badge&logo=css3&logoColor=white)
![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black)

## 🚀 Quick Start

Follow these steps to get the Carbon Emission Prediction System up and running on your local machine.

### Prerequisites

-   **Python 3.9+**: Ensure you have a compatible version of Python installed.

### Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/ambrose-kutti/Carbon-Prediction-System.git
    cd Carbon-Prediction-System
    ```

2.  **Install dependencies**
    It's recommended to use a virtual environment.
    ```bash
    # Create a virtual environment
    python -m venv venv
    # Activate the virtual environment
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate

    # Install Python packages
    pip install -r requirements.txt
    ```

3.  **Train the Machine Learning Models**
    Before running the application, you need to train the models. This script will preprocess the data and save the trained models in the `model/` directory.
    ```bash
    python train_models.py
    ```

4.  **Start the Flask development server**
    ```bash
    python app.py
    ```

5.  **Open your browser**
    Visit `http://localhost:5000` (or the port indicated in your console output).

## 📁 Project Structure

```
Carbon-Prediction-System/
├── CO2 Emissions_Canada.csv    # Primary dataset for training
├── Data Description.csv        # Description of the dataset columns
├── app.py                      # Main Flask application entry point
├── backend.py                  # Core ML logic and prediction functions
├── model/                      # Directory to store trained ML models (.pkl files)
├── requirements.txt            # Python dependencies
├── static/                     # Static assets (CSS, JS, images)
│   └── style.css
│   └── script.js
├── templates/                  # HTML templates for the web interface
│   ├── index.html              # Input form for prediction
│   └── results.html            # Displays prediction results
└── train_models.py             # Script for data preprocessing and model training
```

## ⚙️ Configuration

### Environment Variables
This project does not currently rely on specific environment variables for its core functionality, making it easy to run. For production deployments, you might consider setting `FLASK_ENV=production`.

### Configuration Files
-   `requirements.txt`: Lists all Python packages required for the project.

## 🔧 Development

### Available Scripts
-   `python train_models.py`: Executes the model training and saving pipeline.
-   `python app.py`: Starts the Flask web application.

### Development Workflow
1.  Ensure all dependencies are installed using `pip install -r requirements.txt`.
2.  If you modify the ML models or training data, run `python train_models.py` to retrain and update the saved models.
3.  Run `python app.py` to start the Flask server. Any changes to Python files will usually require restarting the server (Flask's debug mode can auto-reload, but it's not explicitly configured here).
4.  Modify HTML, CSS, or JavaScript files in `templates/` and `static/` as needed for frontend changes.

## 🧪 Testing

While this repository does not include explicit unit tests, you can manually test the system by:
1.  Running `train_models.py` and verifying that model files are generated in the `model/` directory.
2.  Starting `app.py` and navigating to `http://localhost:5000`.
3.  Interacting with the web form, submitting different input values, and observing the prediction results.

## 🚀 Deployment

To deploy this Flask application to a production environment:

### Production Build
There isn't a "build" step in the traditional frontend sense. The Python code runs directly.

### Deployment Options
-   **WSGI Server**: For production, Flask applications are typically served with a WSGI server like Gunicorn or uWSGI, behind a reverse proxy like Nginx or Apache.
    ```bash
    # Example with Gunicorn (install with: pip install gunicorn)
    gunicorn -w 4 app:app
    ```
-   **Containerization (Docker)**: A Dockerfile could be added to containerize the application, making deployment consistent across environments.
-   **Cloud Platforms**: Services like Heroku, AWS Elastic Beanstalk, Google App Engine, or Azure App Service can host Flask applications.

## 📚 API Reference

The application exposes the following routes:

### `/` (GET)
-   **Description**: Renders the main input form for carbon emission prediction.
-   **Template**: `templates/index.html`

### `/predict` (POST)
-   **Description**: Receives user input from the form, processes it, performs carbon emission prediction using the trained ML model, and displays the result.
-   **Request Body**: Form data containing features required by the ML model.
-   **Template**: `templates/results.html`

## 🤝 Contributing

We welcome contributions to enhance this project! If you're interested in improving the system, please consider:
-   Adding more robust error handling.
-   Implementing user authentication (if extending to a multi-user system).
-   Integrating a database for storing predictions or user data.
-   Improving the frontend UI/UX.
-   Experimenting with new machine learning models or feature engineering techniques.

Please feel free to fork the repository, make your changes, and submit a pull request.

## 📄 License

This project is open-source and available under the [LICENSE_NAME](LICENSE) - see the `LICENSE` file for details. <!-- TODO: Add a LICENSE file (e.g., MIT, Apache 2.0) -->

## 🙏 Acknowledgments

-   **Flask Community**: For the flexible and lightweight web framework.
-   **Scikit-learn, Pandas, NumPy**: Essential libraries for machine learning and data manipulation.
-   **CO2 Emissions_Canada.csv dataset**: The data source used for training the models.

## 📞 Support & Contact

-   🐛 Issues: [GitHub Issues](https://github.com/ambrose-kutti/Carbon-Prediction-System/issues)
-   👤 Author: [Ambrose Kutti](https://github.com/ambrose-kutti)

---

<div align="center">

**⭐ Star this repo if you find it helpful!**

Made with ❤️ by [ambrose-kutti](https://github.com/ambrose-kutti)

</div>
