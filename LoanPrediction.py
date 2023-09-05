import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score,train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.linear_model import LogisticRegression
from tkinter import *
from tkinter import messagebox  

def predict(s):
    data= pd.read_csv('loan.csv')
    data=pd.DataFrame(data)
    data['Gender'] = data['Gender'].fillna('Male')
    data['LoanAmount'] = data['LoanAmount'].fillna(data['LoanAmount'].median())
    data['Dependents'] = data['Dependents'].fillna('1')
    data['Self_Employed']= data['Self_Employed'].fillna('No')
    data['Credit_History'] = data['Credit_History'].fillna(1)
    data['Loan_Amount_Term'] = data['Loan_Amount_Term'].fillna(data['Loan_Amount_Term'].median())
    data['Married'] = data['Married'].fillna('No')
    data['Married'].replace({'No':0, 'Yes':1},inplace=True)
    data['Gender'].replace({'Male':0, 'Female':1},inplace=True)
    data['Education'].replace({'Graduate':1,'Not Graduate':0},inplace=True)
    data['Self_Employed'].replace({'No':0, 'Yes':1},inplace=True)
    data['Loan_Status'].replace({'N':0, 'Y':1},inplace=True)
    data['Property_Area'].replace({'Rural':1, 'Semiurban':3, 'Urban':2},inplace=True)
    data['Dependents'].replace({'3+':3,'2':2,'1':1,'0':0},inplace=True)
    
    x = data.drop(['Loan_Status','Loan_ID'],axis=1)
    y = data['Loan_Status']

    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=10)
    # nsamples, nx, ny = x_train.shape
    # x_train = train_dataset.reshape((nsamples,nx*ny))


    model= LogisticRegression(random_state=0,max_iter=1000).fit(x_train, y_train)
    y_pred=model.predict(x_test)
    accuracy=accuracy_score(y_test,y_pred)
    #m = np.array([x])
    # m = np.array([p])
    # result= model.predict(m)
    n=s.split(" ")
    l=[]
    for i in n:
        l.append(int(i))
    result=model.predict([l])    
    if(result== 1):
        messagebox.showinfo("Prediction","Yes,Person has Loan")
    else:
        messagebox.showinfo("Prediction","No,Person doesn't has Loan")


def clearFunc():
    loan_id.set("")
    gender.set(None)
    Marry.set(None)
    dependents.set(0)
    edu.set(None)
    emp.set(None)
    Aincome.set("")
    CAincome.set("")
    loanamt.set("")
    loanterm.set("")
    history.set("")
    Parea.set(None)


if __name__=="__main__":
    root=Tk()
    root.title("Loan Status Prediction")
    loan_id=StringVar()
    Label(root,text="Enter the Loan ID:",fg="blue",font=("Ariel",16)).place(x=12,y=10)
    Entry(root,width=20,textvariable=loan_id).place(x=200,y=15)


    Label(root,text="Select the Gender:",fg="blue",font=("Ariel",16)).place(x=12,y=60)
    gender= StringVar()
    Radiobutton(root, text='Male', variable=gender, value=0,font=("Ariel",16)).place(x=200,y=60)
    Radiobutton(root, text='Female', variable=gender, value=1,font=("Ariel",16)).place(x=200,y=90)

    Label(root,text="Select the Mariatal status:",fg="blue",font=("Ariel",16)).place(x=12,y=130)
    Marry= StringVar()
    Radiobutton(root, text='single', variable=Marry, value=0,font=("Ariel",16)).place(x=260,y=130)
    Radiobutton(root, text='Married', variable=Marry, value=1,font=("Ariel",16)).place(x=260,y=160)

    dependents=StringVar()
    Label(root,text="Enter the Dependents:",fg="blue",font=("Ariel",16)).place(x=12,y=190)
    Entry(root,width=20,textvariable=dependents).place(x=230,y=195)

    Label(root,text="Select the Education status:",fg="blue",font=("Ariel",16)).place(x=12,y=230)
    edu= StringVar()
    Radiobutton(root, text='Graduate', variable=edu, value=1,font=("Ariel",16)).place(x=280,y=230)
    Radiobutton(root, text='Not Graduate', variable=edu, value=0,font=("Ariel",16)).place(x=280,y=260)

    Label(root,text="Self-employeed ?",fg="blue",font=("Ariel",16)).place(x=12,y=290)
    emp= StringVar()
    Radiobutton(root, text='Yes', variable=emp, value=1,font=("Ariel",16)).place(x=200,y=290)
    Radiobutton(root, text='No', variable=emp, value=0,font=("Ariel",16)).place(x=200,y=320)

    Aincome=StringVar()
    Label(root,text="Enter the Applicants income:",fg="blue",font=("Ariel",16)).place(x=12,y=350)
    Entry(root,width=20,textvariable=Aincome).place(x=290,y=355)

    CAincome=StringVar()
    Label(root,text="Enter the Co-applicants income:",fg="blue",font=("Ariel",16)).place(x=12,y=390)
    Entry(root,width=20,textvariable=CAincome).place(x=320,y=395)

    loanamt=StringVar()
    Label(root,text="Enter the loan amount:",fg="blue",font=("Ariel",16)).place(x=12,y=430)
    Entry(root,width=20,textvariable=loanamt).place(x=320,y=435)

    loanterm=StringVar()
    Label(root,text="Enter the loan amount term:",fg="blue",font=("Ariel",16)).place(x=12,y=470)
    Entry(root,width=20,textvariable=loanterm).place(x=320,y=475)

    history=StringVar()
    Label(root,text="Enter the Credit History:",fg="blue",font=("Ariel",16)).place(x=12,y=510)
    Entry(root,width=20,textvariable=history).place(x=250,y=515)

    Parea=StringVar()
    Label(root,text="Enter the property area:",fg="blue",font=("Ariel",16)).place(x=12,y=550)
    Radiobutton(root, text='Rular', variable=Parea, value=1,font=("Ariel",16)).place(x=250,y=550)
    Radiobutton(root, text='Urban', variable=Parea, value=2,font=("Ariel",16)).place(x=250,y=580)
    Radiobutton(root, text='Semiurban', variable=Parea, value=3,font=("Ariel",16)).place(x=250,y=610)

    Button(root, text='Reset', width=10,height=2,font=("Ariel",16),fg="black", command=clearFunc).place(x=12,y=650)
    Button(root, text='Predict', width=10,height=2,font=("Ariel",16),fg="green", command=lambda:[predict(gender.get()+" "+Marry.get()+" "+dependents.get()+" "+edu.get()+" "+emp.get()+" "+Aincome.get()+" "+CAincome.get()+" "+loanamt.get()+" "+loanterm.get()+" "+history.get()+" "+Parea.get())]).place(x=230,y=650)
    Button(root, text='Stop', width=10,height=2,font=("Ariel",16),fg="red", command=root.destroy).place(x=450,y=650)

    root.geometry("600x750+500+150")
    root.resizable(0,0)
    root.mainloop()

    