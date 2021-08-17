import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from tkinter import *
from tkmacosx import *
from tkinter import ttk

dataSet=pd.read_csv("insurance dataset.csv")
print(dataSet.head())
print("rows :",dataSet.shape[0],"columns :",dataSet.shape[1])
print("\nData Set Info\n")
dataSet.info()

print("\n\n",dataSet.describe())

# distribution of age valuer
sns.set()
plt.figure(figsize=(6,6))
sns.histplot(dataSet['age'])
plt.title('Age Distribution')
plt.show()

# Gender column
plt.figure(figsize=(6,6))
sns.countplot(x='sex', data=dataSet)
plt.title('Sex Distribution')
plt.show()

# bmi distribution
plt.figure(figsize=(6,6))
sns.histplot(dataSet['bmi'])
plt.title('BMI Distribution')
plt.show()
#Normal BMI Range --> 18.5 to 24.9

# children column
plt.figure(figsize=(6,6))
sns.countplot(x='children', data=dataSet)
plt.title('Children')
plt.show()

print("\nChildrens",dataSet['children'].value_counts(),sep="\n")

# smoker column
plt.figure(figsize=(6,6))
sns.countplot(x='smoker', data=dataSet)
plt.title('smoker')
plt.show()

print("\nSmokers",dataSet["smoker"].value_counts(),sep="\n")

# region column
plt.figure(figsize=(6,6))
sns.countplot(x='region', data=dataSet)
plt.title('region')
plt.show()

print("\nRegions",dataSet["region"].value_counts(),sep="\n")

# distribution of charges value
plt.figure(figsize=(6,6))
sns.histplot(dataSet['charges'])
plt.title('Charges Distribution')
plt.show()

# encoding sex column
dataSet.replace({'sex':{'male':0,'female':1}}, inplace=True)

# encoding 'smoker' column
dataSet.replace({'smoker':{'yes':1,'no':0}}, inplace=True)

# encoding 'region' column
dataSet.replace({'region':{'southeast':1,'southwest':2,'northeast':3,'northwest':4}}, inplace=True)

print(dataSet.head)

X = dataSet.drop(columns='charges', axis=1)
Y = dataSet['charges']

#Splitting the data into Training data & Testing Data

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# loading the Linear Regression model
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# prediction on training data
training_data_prediction =regressor.predict(X_train)

# R squared value
r2_train = metrics.r2_score(Y_train, training_data_prediction)
print('R squared vale : ', r2_train)

# prediction on test data
test_data_prediction =regressor.predict(X_test)

# R squared value
r2_test = metrics.r2_score(Y_test, test_data_prediction)
print('R squared vale : ', r2_test)

'''
input_data = (31,1,25.74,0,1,0)

# changing input_data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = regressor.predict(input_data_reshaped)
print(prediction)

print('The insurance cost is USD ', prediction[0])

'''


def prediction(root):
    Age = int(age.get())
    temp = sexOption.get()
    Sex = 1
    if temp == 'Male':
        Sex = 0
    Bmi = float(bmi.get())
    Children = int(children.get())
    temp = smokeOption.get()
    Smoke = 1
    if temp == 'No':
        Smoke = 0
    temp = regionOption.get()
    Region = 1
    if temp == "southeast":
        Region = 2
    elif temp == "northwest":
        Region = 3
    else:
        Region = 4
    input_data = (Age, Sex, Bmi, Children, Smoke, Region)
    # changing input_data to a numpy array

    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = regressor.predict(input_data_reshaped)
    # result.config(text='The insurance cost is USD '+ prediction[0])
    print('The insurance cost is USD ' , prediction[0])
    text=float(prediction[0])
    result = Label(root,text="The insurance cost is %.2f USD"%abs(text), font="helvetica 20 bold", fg='green', bg='black')
    result.place(x=400, y=300, anchor="center")



### end of training and testing process ###


root=Tk()
root.geometry("800x400")
root.configure(bg='#faefcf')
root.title("Medical Insurance Prediction")
Label(root,text="Please fill all the options to get your insurance prediction!",fg='black',bg='red',font='Verdana 24 bold').grid(row=0,column=0,columnspan=10)
Label(root,text="Age: ",fg='black',bg='yellow',font='Verdana 15 bold').grid(row=1,column=0,padx=5,pady=5)
age=StringVar()
Entry(root,fg='black',textvariable=age,font='Helvetica 15 bold').grid(row=1,column=1,padx=5,pady=5,columnspan=1)
Label(root,text='Sex: ',fg='black',bg='yellow',font='Verdana 15 bold').grid(row=1,column=2,padx=5,pady=5)

sexOption=StringVar()
temp=ttk.Combobox(root,width=20,textvariable=sexOption)
temp['values']=('Male','Female')
temp.grid(row=1,column=3,padx=5,pady=5)
temp.current()

Label(root,text='BMI: ',fg='black',bg='yellow',font='Verdana 15 bold').grid(row=2,column=0,padx=5,pady=5)
bmi=StringVar()
Entry(root,fg='black',textvariable=bmi,font='Helvetica 15 bold').grid(row=2,column=1,padx=5,pady=5)

Label(root,text='Children: ',fg='black',bg='yellow',font='Verdana 15 bold').grid(row=2,column=2,padx=5,pady=5)
children=StringVar()
Entry(root,fg='black',textvariable=children,font='Helvetica 15 bold').grid(row=2,column=3,padx=5,pady=5)

Label(root,text='Smoker: ',fg='black',bg='yellow',font='Verdana 15 bold').grid(row=3,column=0,padx=5,pady=5)
smokeOption=StringVar()
temp=ttk.Combobox(root,width=20,textvariable=smokeOption)
temp['values']=('Yes',"No")
temp.grid(row=3,column=1,padx=5,pady=5)
temp.current()

Label(root,text='Region: ',fg='black',bg='yellow',font='Verdana 15 bold').grid(row=3,column=2,padx=5,pady=5)
regionOption=StringVar()
temp=ttk.Combobox(root,width=20,textvariable=regionOption)
temp['values']=('southwest','southeast','northwest','northeast')
temp.grid(row=3,column=3,padx=5,pady=5)
temp.current()


Button(root, text="Get Insurance Cost", font="helvetica 25 bold", fg="green", bg="black", borderless=1,
       command=lambda:prediction(root)).place(x=400,y=180,anchor='center')


root.mainloop()







