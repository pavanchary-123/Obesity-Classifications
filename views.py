# Create your views here.
from django.shortcuts import render,HttpResponse
from django.contrib import messages
from .forms import UserRegistrationForm
from .models import UserRegistrationModel
from django.conf import settings


# Create your views here.
def UserRegisterActions(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            print('Data is Valid')
            form.save()
            messages.success(request, 'You have been successfully registered')
            form = UserRegistrationForm()
            return render(request, 'UserRegistrations.html', {'form': form})
        else:
            messages.success(request, 'Email or Mobile Already Existed')
            print("Invalid form")
    else:
        form = UserRegistrationForm()
    return render(request, 'UserRegistrations.html', {'form': form})

def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')
        print("Login ID = ", loginid, ' Password = ', pswd)
        try:
            check = UserRegistrationModel.objects.get(loginid=loginid, password=pswd)
            status = check.status
            print('Status is = ', status)
            if status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                print("User id At", check.id, status)
                return render(request, 'users/UserHomePage.html', {})
            else:
                messages.success(request, 'Your Account Not at activated')
                return render(request, 'UserLogin.html')
        except Exception as e:
            print('Exception is ', str(e))
            pass
        messages.success(request, 'Invalid Login id and password')
    return render(request, 'UserLogin.html', {})


def UserHome(request):
    return render(request, 'users/UserHomePage.html', {})

 
def DatasetView(request):
    path = settings.MEDIA_ROOT+"//"+'ObesityDataSet_raw_and_data_sinthetic.csv'
    import pandas as pd
    df = pd.read_csv(path)
    print('-'*100)
    print(df)
    df =df.to_html
    return render(request, 'users/viewdataset.html', {'data': df})


from users.utility import ml 
def user_predictions(request):
    if request.method == 'POST':
        gender = request.POST.get('Gender')
        Age = request.POST.get('Age')
        Height = request.POST.get('Height')
        Weight = request.POST.get('Weight')
        family_history_with_overweigh = request.POST.get('family_history_with_overweigh')
        FAVC = request.POST.get('FAVC')
        FCVC = request.POST.get('FCVC')
        NCP = request.POST.get('NCP')
        CAEC = request.POST.get('CAEC')
        SMOKE = request.POST.get('SMOKE')
        CH2O = request.POST.get('CH2O')
        SCC = request.POST.get('SCC')
        FAF = request.POST.get('FAF')
        TUE = request.POST.get('TUE')
        CALC = request.POST.get('CALC')
        MTRANS = request.POST.get('MTRANS')
        test_list=[gender,Age,Height,Weight,family_history_with_overweigh,FAVC,FCVC,NCP,CAEC,SMOKE,CH2O,SCC,FAF,TUE,CALC,MTRANS]
        import os
        from django.conf import settings
        import pandas as pd
        from sklearn.preprocessing import LabelEncoder
        from sklearn.svm import SVC
        from sklearn.metrics import classification_report
        from sklearn.model_selection import train_test_split
        dataSetPath = os.path.join(settings.MEDIA_ROOT,'ObesityDataSet_raw_and_data_sinthetic.csv')
        dataSet = pd.read_csv(dataSetPath)
        le = LabelEncoder()
        dataSet['Gender'] = le.fit_transform(dataSet['Gender'])
        dataSet['CALC'] = le.fit_transform(dataSet['CALC'])
        dataSet['MTRANS'] = le.fit_transform(dataSet['MTRANS'])
        dataSet['NObeyesdad'] = le.fit_transform(dataSet['NObeyesdad'])
        dataSet['family_history_with_overweight'] = le.fit_transform(dataSet['family_history_with_overweight'])
        dataSet['FAVC'] = le.fit_transform(dataSet['FAVC'])
        dataSet['CAEC'] = le.fit_transform(dataSet['CAEC'])
        dataSet['SMOKE'] = le.fit_transform(dataSet['SMOKE'])
        dataSet['SCC'] = le.fit_transform(dataSet['SCC'])
        dataSet['TUE'] = le.fit_transform(dataSet['TUE'])
        # print(dataSet)
        # print(dataSet.dtypes)

        x = dataSet.iloc[:,:-1]
        y = dataSet.iloc[:,-1]
        x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
        # print('-'*100)
        # print(x_train)
        # print('-'*100)
        # print(x_test)
        # print('-'*100)
        # print(y_train)
        # print('-'*100)
        # print(y_test)
        model = SVC()
        model.fit(x_train,y_train)
        pred_res = model.predict([test_list])
        print(pred_res)

        output={0:"Insufficient_Weight",1:"Normal_Weight",2:"Obesity_Type_I",3:"Obesity_Type_II",4:"Obesity_Type_III",5:"Overweight_Level_I",6:"Overweight_Level_II"}
        a = output.get(pred_res[0])
        print(a)
        print(pred_res)
        return render(request,"users/testform.html",{"result":a})
    else:
        return render(request, "users/testform.html")



def machinelearning(request):
    import os
    from django.conf import settings
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder
    from sklearn.svm import SVC
    from sklearn.metrics import classification_report
    from sklearn.model_selection import train_test_split
    dataSetPath = os.path.join(settings.MEDIA_ROOT,'ObesityDataSet_raw_and_data_sinthetic.csv')
    dataSet = pd.read_csv(dataSetPath)
    le = LabelEncoder()
    dataSet['Gender'] = le.fit_transform(dataSet['Gender'])
    dataSet['CALC'] = le.fit_transform(dataSet['CALC'])
    dataSet['MTRANS'] = le.fit_transform(dataSet['MTRANS'])
    dataSet['NObeyesdad'] = le.fit_transform(dataSet['NObeyesdad'])
    dataSet['family_history_with_overweight'] = le.fit_transform(dataSet['family_history_with_overweight'])
    dataSet['FAVC'] = le.fit_transform(dataSet['FAVC'])
    dataSet['CAEC'] = le.fit_transform(dataSet['CAEC'])
    dataSet['SMOKE'] = le.fit_transform(dataSet['SMOKE'])
    dataSet['SCC'] = le.fit_transform(dataSet['SCC'])
    dataSet['TUE'] = le.fit_transform(dataSet['TUE'])

    # print(dataSet)
    # print(dataSet.dtypes)

    x = dataSet.iloc[:,:-1]
    y = dataSet.iloc[:,-1]
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
    # print('-'*100)
    # print(x_train)
    # print('-'*100)
    # print(x_test)
    # print('-'*100)
    # print(y_train)
    # print('-'*100)
    # print(y_test)
    model = SVC()
    model.fit(x_train,y_train)
    pred_res = model.predict(x_test)
    # print(pred_res)
    cr = classification_report(y_test,pred_res,output_dict=True)
    cr = pd.DataFrame(cr).transpose().to_html()

    


    return render(request,'users/MlResults.html',{'report':cr})

def ANN(request):
        import numpy as np
        import os
        from django.conf import settings
        import pandas as pd
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from keras.layers import Dense
        from sklearn.preprocessing import LabelEncoder
        from sklearn.model_selection import train_test_split
        dataSetPath = os.path.join(settings.MEDIA_ROOT,'ObesityDataSet_raw_and_data_sinthetic.csv')
        dataSet = pd.read_csv(dataSetPath)
        le = LabelEncoder()
        dataSet['Gender'] = le.fit_transform(dataSet['Gender'])
        dataSet['CALC'] = le.fit_transform(dataSet['CALC'])
        dataSet['MTRANS'] = le.fit_transform(dataSet['MTRANS'])
        dataSet['NObeyesdad'] = le.fit_transform(dataSet['NObeyesdad'])
        dataSet['family_history_with_overweight'] = le.fit_transform(dataSet['family_history_with_overweight'])
        dataSet['FAVC'] = le.fit_transform(dataSet['FAVC'])
        dataSet['CAEC'] = le.fit_transform(dataSet['CAEC'])
        dataSet['SMOKE'] = le.fit_transform(dataSet['SMOKE'])
        dataSet['SCC'] = le.fit_transform(dataSet['SCC'])
        dataSet['TUE'] = le.fit_transform(dataSet['TUE'])

        # print(dataSet)
        # print(dataSet.dtypes)

        x = dataSet.iloc[:,:-1]
        y = dataSet.iloc[:,-1]
        x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)
       
        model = Sequential()
        model.add(Dense(7, input_dim=16, activation='relu'))
        # model.add(Dense(4,activation='relu'))
        model.add(Dense(7,activation='softmax'))
        # model.compile(optimizer='adam', loss="mse",metrics=["mae","mse"])
        model.compile(optimizer='adam',loss="mae", metrics=['accuracy','mae','mse'])
        history = model.fit(x_train, y_train,epochs=25,batch_size=15)
        mae = history.history['mae'][-1]
        mse = history.history['mse'][-1]
        loss = history.history['loss'][-1]
        acc = history.history['accuracy'][-1]        
        acc = round(acc*100,2)
        acc = 100-acc
        return render(request,'users/ann.html',{"mae":mae,'mse':mse,'loss':loss,'acc': acc})