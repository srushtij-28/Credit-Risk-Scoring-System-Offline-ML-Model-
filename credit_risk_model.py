import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Sample dataset
data = {
    "income": [50000, 60000, 30000, 80000, 20000, 90000],
    "debt": [5000, 10000, 15000, 2000, 18000, 1000],
    "credit_score": [700, 720, 600, 780, 580, 800],
    "approved": [1, 1, 0, 1, 0, 1]
}

df = pd.DataFrame(data)

X = df[["income", "debt", "credit_score"]]
y = df["approved"]

model = DecisionTreeClassifier()
model.fit(X, y)

print("🏦 Credit Risk Predictor\n")

income = float(input("Enter applicant income: "))
debt = float(input("Enter applicant debt: "))
score = float(input("Enter credit score: "))

prediction = model.predict([[income, debt, score]])[0]

if prediction == 1:
    print("\n✅ Loan Approved (Low Risk)")
else:
    print("\n❌ Loan Rejected (High Risk)")
