from django.shortcuts import render;
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

def home(request):
    return render(request, "home.html")
def predict(request):
    return render(request, "predict.html")
def result(request):
    data = pd.read_csv(r"E:\USA_Housing (1).csv")
    data=data.drop(['Address'], axis=1)
    x=data.drop('Price', axis=1)
    y=data['Price']
    x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.3)
    model=LinearRegression()
    model.fit(x_train, y_train)
    
    # var1=float(request.GET['n1'])
    # var2=float(request.GET['n2'])
    # var3=float(request.GET['n3'])
    # var4=float(request.GET['n4'])
    # var5=float(request.GET['n5'])
    
    # pred=model.predict(np.array([var1, var2, var3, var4, var5]).reshape(1,-1))
    # pred=round(pred[0])
    # price="The Predicted Price is $"+str(pred)
    
    try:
        var1 = float(request.GET.get('n1', 0) or 0)
        var2 = float(request.GET.get('n2', 0) or 0)
        var3 = float(request.GET.get('n3', 0) or 0)
        var4 = float(request.GET.get('n4', 0) or 0)
        var5 = float(request.GET.get('n5', 0) or 0)

        # Creating a NumPy array and reshaping it to match the model's input format
        input_data = np.array([var1, var2, var3, var4, var5]).reshape(1, -1)

        pred = model.predict(input_data)  # Making prediction
        pred = round(pred[0])  # Rounding off the prediction
        price = f"The Predicted Price is ${pred}"

    except ValueError:
        price = "Invalid input. Please enter valid numeric values."
    
    return render(request, "predict.html", {"result2":price})
