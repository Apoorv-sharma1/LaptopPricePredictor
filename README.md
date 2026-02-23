# 💻 Laptop Price Predictor | Machine Learning Project

A Machine Learning-based web application that predicts the **price of a laptop** based on its specifications such as brand, RAM, storage, processor, GPU, screen size, and operating system.

---

## 🚀 Project Overview

Buying a laptop can be confusing due to the wide range of configurations and prices.  
This project solves that problem by using **Machine Learning algorithms** to predict the **approximate price of a laptop** based on its features.

The model is trained on a real-world dataset and deployed using a simple and interactive web interface.

---

## 🧠 Machine Learning Concepts Used

- Supervised Learning  
- Linear Regression / Multiple Linear Regression  
- Data Preprocessing  
- Feature Encoding  
- Train-Test Split  
- Model Evaluation  
- Prediction System  

---

## 🛠️ Tech Stack

| Category | Technologies |
|--------|-------------|
| Programming Language | Python 🐍 |
| Libraries | NumPy, Pandas, Scikit-learn |
| Visualization | Matplotlib, Seaborn |
| Web Framework | Flask |
| Frontend | HTML, CSS |
| Deployment | Localhost / (Optional: Render / Heroku) |

---

## 📊 Dataset Features

The dataset contains the following features:

- Brand
- Laptop Type
- RAM (GB)
- Storage (HDD/SSD)
- Processor
- GPU
- Operating System
- Screen Size
- Weight

🎯 **Target Variable:** Laptop Price

---

## ⚙️ How the Model Works

1. User enters laptop specifications
2. Input data is preprocessed & encoded
3. Trained ML model predicts the price
4. Predicted price is displayed to the user

---

## 📈 Model Performance

- Algorithm Used: **Linear Regression**
- Evaluation Metric: **Mean Squared Error (MSE)**
- Model trained using clean and optimized dataset

---

## 🖥️ Project Structure
Laptop-Price-Predictor/
│
├── dataset/
│ └── laptop_data.csv
│
├── model/
│ └── laptop_price_model.pkl
│
├── app.py
├── requirements.txt
├── templates/
│ └── index.html
│
├── static/
│ └── style.css
│
└── README.md


---

## ▶️ How to Run the Project

### 1️⃣ Clone the Repository
bash
git clone https://github.com/your-username/laptop-price-predictor.git

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Run the Application
python app.py

4️⃣ Open in Browser
http://127.0.0.1:5000/

🎯 Use Cases

Laptop price estimation

E-commerce price comparison

Machine learning learning project

College mini-project / major project

🌱 Future Enhancements

Use advanced algorithms (Random Forest, XGBoost)

Improve UI/UX

Add more laptop features

Deploy on cloud platform

Add comparison between multiple laptops

🤝 Contributing

Contributions are welcome!
Feel free to fork the repository and submit a pull request.

📜 License

This project is licensed under the MIT License.

⭐ Acknowledgements

Dataset Source: Kaggle

Libraries: Scikit-learn, Pandas, NumPy

Inspired by real-world price prediction systems

⭐ If you like this project, don't forget to star the repository!
