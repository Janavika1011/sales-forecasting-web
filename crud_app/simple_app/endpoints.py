import logging

from flask_pymongo import pymongo
from flask import jsonify, request
import pandas as pd

con_string ="mongodb+srv://januanandan21:januanand@cluster0.ge8jes2.mongodb.net/?retryWrites=true&w=majority"

client = pymongo.MongoClient(con_string)

db = client.get_database('janu')

user_collection = pymongo.collection.Collection(db, 'janucollection') #(<database_name>,"<collection_name>")
print("MongoDB connected Successfully")


def project_api_routes(endpoints):
    @endpoints.route('/hello', methods=['GET'])
    def hello():
        res = 'Hello world'
        print("Hello world")
        return res

    @endpoints.route('/register-user', methods=['POST'])
    def register_user():
        resp = {}
        try:
            req_body = request.json
            # resp['hello'] = hello_world
            # req_body = req_body.to_dict()
            user_collection.insert_one(req_body)            
            print("User Data Stored Successfully in the Database.")
            status = {
                "statusCode":"200",
                "statusMessage":"User Data Stored Successfully in the Database."
            }
        except Exception as e:
            print(e)
            status = {
                "statusCode":"400",
                "statusMessage":str(e)
            }
        resp["status"] =status
        return resp


   

    @endpoints.route('/read-users',methods=['GET'])
    def read_users():
        resp = {}
        try:
            users = user_collection.find({})
            print(users)
            users = list(users)
            status = {
                "statusCode":"200",
                "statusMessage":"User Data Retrieved Successfully from the Database."
            }
            output = [{'Name' : user['name'], 'Email' : user['email']} for user in users]   #list comprehension
            resp['data'] = output
        except Exception as e:
            print(e)
            status = {
                "statusCode":"400",
                "statusMessage":str(e)
            }
        resp["status"] =status
        return resp
    

    @endpoints.route('/login',methods=["POST"])
    def login():
        resp={}
        email=request.json['email']
        password=request.json['password']
        if user_collection.find_one({'email':email,'password':password}):
            status = {
                "statusCode":"200",
                "statusMessage":"sucsess"
            }
        else:
            status = {
                "statusCode":"400",
                "statusMessage":"ërror"
            }
        
        return jsonify(status)

    @endpoints.route('/update-users',methods=['PUT'])
    def update_users():
        resp = {}
        try:
            req_body = request.json
            # req_body = req_body.to_dict()
            user_collection.update_one({"id":req_body['id']}, {"$set": req_body['updated_user_body']})
            print("User Data Updated Successfully in the Database.")
            status = {
                "statusCode":"200",
                "statusMessage":"User Data Updated Successfully in the Database."
            }
        except Exception as e:
            print(e)
            status = {
                "statusCode":"400",
                "statusMessage":str(e)
            }
        resp["status"] =status
        return resp    

    @endpoints.route('/delete',methods=['DELETE'])
    def delete():
        resp = {}
        try:
            delete_id = request.args.get('delete_id')
            user_collection.delete_one({"id":delete_id})
            status = {
                "statusCode":"200",
                "statusMessage":"User Data Deleted Successfully in the Database."
            }
        except Exception as e:
            print(e)
            status = {
                "statusCode":"400",
                "statusMessage":str(e)
            }
        resp["status"] =status
        return resp
    
    @endpoints.route('perrin-freres-monthly-champagne- (1).csv',methods=['POST'])
    def file_upload():
        import numpy as np
        import pandas as pd
        import pickle

        import matplotlib.pyplot as plt
        df.head()
        df.tail()
        df.columns=["Month","Sales"]
        df.head()
        df.drop(106,axis=0,inplace=True)
        df.tail()
        df.drop(105,axis=0,inplace=True)
        df.tail()
        df['Month']=pd.to_datetime(df['Month'])
        df.head()
        df.set_index('Month',inplace=True)
        df.head()
        df.describe()
        df.plot()
        from statsmodels.tsa.stattools import adfuller
        test_result
        def adfuller_test(sales):
            adfuller_test(df['Sales'])
            df['Sales First Difference'] = df['Sales'] - df['Sales'].shift(1)

            df['Sales First Difference'] = df['Sales'] - df['Sales'].shift(1)
            df['Sales']
            df['Sales'].shift(1)
            adfuller_test(df['Sales First Difference'].dropna())
            df['Seasonal First Difference']=df['Sales']-df['Sales'].shift(12)
            df['Sales'].shift(12)
            df.head(14)
            adfuller_test(df['Seasonal First Difference'].dropna())
            df['Seasonal First Difference'].plot()
        from pandas.plotting import autocorrelation_plot
        autocorrelation_plot(df['Sales'])
        plt.show()
        from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
        import statsmodels.api as sm
        fig = plt.figure(figsize=(12,8))
        ax1 = fig.add_subplot(211)
        fig = sm.graphics.tsa.plot_acf(df['Seasonal First Difference'].iloc[13:],lags=40,ax=ax1)
        ax2 = fig.add_subplot(212)
        fig = sm.graphics.tsa.plot_pacf(df['Seasonal First Difference'].iloc[13:],lags=40,ax=ax2)
        from statsmodels.tsa.arima.model import ARIMA
        model=ARIMA(df['Sales'],order=(1,1,1))
        model_fit=model.fit()
        model_fit.summary()
        df['forecast']=model_fit.predict(start=90,end=103,dynamic=True)
        df[['Sales','forecast']].plot(figsize=(12,8))
        import statsmodels.api as sm
        model=sm.tsa.statespace.SARIMAX(df['Sales'],order=(1, 1, 1),seasonal_order=(1,1,1,12))
        results=model.fit()
        df['forecast']=results.predict(start=90,end=103,dynamic=True)
        df[['Sales','forecast']].plot(figsize=(12,8))
        from pandas.tseries.offsets import DateOffset
        future_dates=[df.index[-1]+ DateOffset(months=x)for x in range(0,48)]
        future_datest_df=pd.DataFrame(index=future_dates[1:],columns=df.columns)
        future_datest_df.tail()
        future_df=pd.concat([df,future_datest_df])
        future_df
        future_df['forecast'] = results.predict(start = 104, end = 175, dynamic= True)  
        future_df[['Sales', 'forecast']].plot(figsize=(12, 8))
        print("Hello") 















       
        resp = {}
        try:
            req = request.form
            file = request.files.get('file')
            df = pd.read_csv(file)
            print(df)
            print(df.head)
            print(df.columns())
            status = {
                "statusCode":"200",
                "statusMessage":"File uploaded Successfully."
            }
        except Exception as e:
            print(e)
            status = {
                "statusCode":"400",
                "statusMessage":str(e)
            }
        resp["status"] =status
        return resp


    return endpoints
