# 📊 Customer Churn Prediction Dashboard

A **Streamlit web application** that predicts **customer churn** using a pre-trained **XGBoost model**.  
The app provides **real-time churn probability scores** based on customer details and also includes **interactive visualizations** for exploring churn patterns.

---


## 🛠️ Tech Stack

- **Python 3.8+**
- **Streamlit**
- **Pandas**
- **Scikit-learn**
- **XGBoost**
- **Plotly (for visualizations)**

---

## 🚀 Features

- 📊 **Interactive Data Visualization** – Explore churn trends by contract type, internet service, payment method, and more.  
- 🔮 **Churn Prediction** – Enter customer details to get churn probability using a trained ML pipeline.  
- 🎨 **Modern UI/UX** – Clean dashboard with styled charts, icons, and progress bars.  
- ⚡ **Real-time Inference** – Predictions served instantly inside the Streamlit app.  

---

## 📂 Project Structure

```
.
├── app.py                        # Main Streamlit app
├── churn_model.pkl               # Trained ML model (generated from training script)
├── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Dataset for visualization
├── requirements.txt              # Dependencies
└── README.md                     # Project documentation
```

---

## 🛠️ Setup & Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/bits-blitzz/customerChurnFinalProject.git
cd <customerChurnFinalProject>
```

### 2️⃣ Create a Virtual Environment
```bash
# Create the environment
python -m venv venv

# Activate on Windows
.env\Scriptsctivate

# Activate on macOS/Linux
source venv/bin/activate
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Run the Streamlit App
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501/`.

---

## 📊 Dataset

The project uses the **Telco Customer Churn dataset** from IBM:  
[Telco Churn Dataset on Kaggle](https://www.kaggle.com/blastchar/telco-customer-churn)

---

## 🤝 Contributing

Contributions are welcome!  
- Fork the repo  
- Create a new branch (`feature/your-feature`)  
- Commit changes and push  
- Open a Pull Request 🎉  

---

## 📜 License

This project is licensed under the **MIT License**.  
Feel free to use and modify it as per your needs.

---

## 💡 Acknowledgements
- IBM Telco Churn Dataset
- Streamlit Team
- Scikit-learn & XGBoost
