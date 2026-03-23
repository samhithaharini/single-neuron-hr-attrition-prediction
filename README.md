# Single Neuron HR Prediction

This project predicts whether an employee will leave the company using a **Single Neuron Model (Logistic Regression)**.

---

##  Objective
To understand how a single neuron works in Deep Learning by implementing it using Logistic Regression and validating it with a manual prediction function.

---

##  Concept

A single neuron computes:

z = w1x1 + w2x2 + ... + b  
y = sigmoid(z)

Where:
- w → weights  
- b → bias  
- sigmoid → activation function  

---

##  Dataset Features

- satisfaction_level  
- last_evaluation  
- number_project  
- average_monthly_hours  
- time_spend_company  
- Work_accident  
- promotion_last_5years  
- Department  
- salary  

---

##  Steps Performed

1. Data preprocessing (handling categorical variables using get_dummies)  
2. Train-test split  
3. Model training using Logistic Regression  
4. Extracting weights and bias  
5. Implementing manual sigmoid function  
6. Manual prediction using learned parameters  

---

##  Key Insight

Logistic Regression behaves like a **single neuron**:
- Linear combination of inputs  
- Sigmoid activation for probability output  

---

##  Output

- Predicts whether an employee will leave (1) or stay (0)  
- Provides probability using sigmoid function  

---

##  Technologies Used

- Python  
- Pandas  
- NumPy  
- Scikit-learn  

---

##  Conclusion

This project demonstrates the fundamental working of a neuron, forming the base for understanding deep learning models.

---
