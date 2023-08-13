#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install xgboost')
get_ipython().system('pip install lightgbm')
get_ipython().system('pip install shap')
get_ipython().system('pip install pmdarima')


# In[ ]:


import pandas as pd
import numpy as np
from pyspark.sql.functions import *
from pyspark.sql.types import *
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.feature_selection import SelectKBest,mutual_info_regression,SelectFromModel,VarianceThreshold
from sklearn.linear_model import Ridge,Lasso,ElasticNet
from sklearn.feature_selection import f_regression
from sklearn.model_selection import train_test_split,cross_val_score,RandomizedSearchCV,GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error,mean_absolute_percentage_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import shap
from statsmodels.tsa.holtwinters import ExponentialSmoothing
#import lime


# In[ ]:


ctg_seg=spark.read.table("categorysegment.final_rsr_product_data_tier1")
#ctg_seg.display()


# In[ ]:


ctg_seg=ctg_seg.toPandas()
ctg_seg_UK=ctg_seg[ctg_seg.Market=='UNITED KINGDOM']
ctg_seg_AUS=ctg_seg[ctg_seg.Market=='AUSTRALIA']
ctg_seg_GER=ctg_seg[ctg_seg.Market=='GERMANY']
ctg_seg_SPAIN=ctg_seg[ctg_seg.Market=='Spain/Balearic Islands']


# In[ ]:


ctg_seg_UK.display()


# In[ ]:


#ctg_seg_UK[(ctg_seg_UK.Period=='2020-04-01')]
ctg_seg_UK=ctg_seg_UK[ctg_seg_UK.Global_Price_Segment.isnull()==False]
ctg_seg_UK.Period=pd.to_datetime(ctg_seg_UK.Period)
ctg_seg_UK.insert(0,'Date',ctg_seg_UK.Period)
ctg_seg_UK.drop('Period',axis=1,inplace=True)


# In[ ]:



ctg_seg_UK['Format_price_Prod']=ctg_seg_UK['Stick_Format']+"_"+ctg_seg_UK['Global_Price_Segment']+"_"+ctg_seg_UK['Product_Group']
#ctg_seg_UK_RYO=ctg_seg_UK[ctg_seg_UK.Product_Group=='RYO']
#ctg_seg_UK_Cig=ctg_seg_UK[ctg_seg_UK.Product_Group=='Cigarettes']


# In[ ]:



#df_Segment_RYO=pd.DataFrame(ctg_seg_UK_RYO.sort_values(['Date']).groupby(['Date','Format_price_Prod'])['Volume_WSE'].sum()).reset_index()
#df_Segment_Cig=pd.DataFrame(ctg_seg_UK_Cig.sort_values(['Date']).groupby(['Date','Format_price_Prod'])['Volume_WSE'].sum()).reset_index()
df_Segment=pd.DataFrame(ctg_seg_UK.sort_values(['Date']).groupby(['Date','Format_price_Prod'])['Volume_WSE'].sum()).reset_index()


# In[ ]:


df_Segment.display()


# In[ ]:


df_slim=df_Segment[df_Segment.Format_price_Prod=='Super Kings/100s_Low_Cigarettes']


# In[ ]:


df_Segment.groupby('Format_price_Prod').count()


# In[ ]:


df_slim.drop('Format_price_Prod',axis=1,inplace=True)


# In[ ]:


df_slim.set_index('Date',inplace=True)


# In[ ]:


df_slim.plot()


# In[ ]:


train_ratio=int(len(df_slim)*0.85)
df_train=df_slim
df_test=df_slim.iloc[train_ratio:]


# In[ ]:


def grid_param(df_train,df_test):
    param={}

    for trend in ['add','mul']:
        for seasonal in ['add','mul']:
            met={}


            # contrived dataset
            # fit model
            df_train.index.freq='MS'
            model = ExponentialSmoothing(df_train,trend=trend,seasonal=seasonal)
            model_fit = model.fit()
            # make prediction
            yhat = model_fit.forecast(len(df_test))

            score=r2_score(df_test,yhat)
            met['trend']=trend
            met['seasonal']=seasonal
            param[score]=met
    print(param)
    best_param=param[sorted(param)[-1]]
    return best_param


# In[ ]:


from statsmodels.tsa.holtwinters import ExponentialSmoothing
best_param=grid_param(df_slim.iloc[:train_ratio],df_test)
print(best_param)
#df_train=df_sterling_RYO_VOL
df_train.index.freq='MS'
model = ExponentialSmoothing(df_train,trend=best_param['trend'],seasonal=best_param['seasonal'])
model_fit = model.fit()
# make prediction
yhat = model_fit.forecast(24)
df_slim.plot()
df_test.plot()
yhat.plot()
plt.show()


# In[ ]:


df_forecast_Vol=pd.DataFrame()
Brand_pack_var=[]
brand_pack_var_less_data=[]
for i in df_Segment.Format_price_Prod.unique():
    try:
        df_slim=df_Segment[df_Segment.Format_price_Prod==i]
        df_slim.drop('Format_price_Prod',axis=1,inplace=True)
        df_slim.set_index('Date',inplace=True)
        train_ratio=int(len(df_slim)*0.85)
        df_train=df_slim
    #df_train=df_sterling_RYO_VOL.iloc[:train_ratio]
        df_test=df_slim.iloc[train_ratio:]
        
            #print(len(df_train))
        best_param=grid_param(df_slim.iloc[:train_ratio],df_test)
        #print(best_param)
        #df_train=df_sterling_RYO_VOL
        df_train.index.freq='MS'
        model = ExponentialSmoothing(df_train,trend=best_param['trend'],seasonal=best_param['seasonal'])
        model_fit = model.fit()
        # make prediction
        yhat = model_fit.forecast(36)
        yhat=yhat.to_frame(name='Volume_WSE').reset_index()
        yhat.rename(columns={'index':'Date'},inplace=True)
        yhat=df_slim.reset_index().append(yhat)
        yhat['Format_price_Prod']=i
        #print(yhat)
        df_forecast_Vol=df_forecast_Vol.append(yhat)
        #print(len(df_forecast))
        #print(future)
        #future.plot()

    except ValueError:
        #print(brand_pack_variant)
        pass
df_forecast_Vol.loc[df_forecast_Vol.Volume_WSE < 0,'Volume_WSE'] = 0


# In[ ]:


df_forecast_Vol['Stick_Format']=df_forecast_Vol.Format_price_Prod.apply(lambda x : x.split('_')[0])
df_forecast_Vol['Price_Segment']=df_forecast_Vol.Format_price_Prod.apply(lambda x : x.split('_')[1])
df_forecast_Vol['Product_Group']=df_forecast_Vol.Format_price_Prod.apply(lambda x : x.split('_')[2])
df_forecast_Vol.Volume_WSE=df_forecast_Vol.Volume_WSE.astype('int')
#df_forecast_Vol.drop('Format_price_Prod',axis=1,inplace=True)
df_forecast_Vol.display()


# In[ ]:


df_forecast_Vol.display()


# In[ ]:


for seg in df_forecast_Vol.Format_price_Prod.unique():
    print(seg)
    df_forecast_Vol[df_forecast_Vol.Format_price_Prod==seg].drop('Format_price_Prod',axis=1).set_index('Date').plot(figsize=(12,6),title=seg)


# In[ ]:


get_ipython().system('pip install Prophet')


# In[ ]:


from prophet import Prophet
from sklearn.model_selection import ParameterGrid


# In[ ]:


from sklearn.model_selection import ParameterGrid
def hyper_tune(Train_prophet,Test_prophet):
    
    params_grid = {'seasonality_mode':('multiplicative','additive'),
               'changepoint_prior_scale':[0.1,0.2,0.3,0.4,0.5],
              'n_changepoints' : [100,150,200,250]}
    grid = ParameterGrid(params_grid)
    cnt = 0
    for p in grid:
        cnt = cnt+1

    print('Total Possible Models',cnt)

    import random
    strt=str(Test_prophet.ds.min())
    end=str(Test_prophet.ds.max())

    model_parameters = pd.DataFrame(columns = ['MAPE','Parameters'])
    for p in grid:
        test = pd.DataFrame()
        #print(p)
        random.seed(0)
        train_model =Prophet(changepoint_prior_scale = p['changepoint_prior_scale'],
                             n_changepoints = p['n_changepoints'],
                             seasonality_mode = p['seasonality_mode'],
                             interval_width=0.95)
        #train_model.add_country_holidays(country_name='US')
        train_model.fit(Train_prophet)
        test_forecast = train_model.predict(Test_prophet)
        test=test_forecast[['ds','yhat']]
        Actual = Test_prophet[(Test_prophet['ds']>=strt) & (Test_prophet['ds']<=end)]
        #print(Actual['y'],test['yhat'])
        MAPE = mean_absolute_percentage_error(Actual['y'],test['yhat'])
        #print('Mean Absolute Percentage Error(MAPE)------------------------------------',MAPE)
        model_parameters = model_parameters.append({'MAPE':MAPE,'Parameters':p},ignore_index=True)
        
    parameters = model_parameters.sort_values(by=['MAPE'])
    parameters = parameters.reset_index(drop=True)
    best_param=parameters.Parameters[0]
    return best_param


# In[ ]:


def prophet(Format_price_Seg):
    df_slim=df_Segment[df_Segment.Format_price_Seg==Format_price_Seg]
    df_slim.drop('Format_price_Seg',axis=1,inplace=True)


    model_prophet=Prophet()
    model_prophet.add_seasonality(name='monthly',period=30,fourier_order=5)
    train_ratio=int(len(df_slim)*0.8)
    #df_train=df_sterling_RYO.iloc[:train_ratio]
    df_train=df_slim
    df_test=df_slim.iloc[train_ratio:]
    Train_prophet=df_slim.iloc[:train_ratio].copy(deep=True)
    Test_prophet=df_test.copy(deep=True)
    Train_prophet=Train_prophet.rename(columns={'Date':'ds','Volume_WSE':'y'})
    Test_prophet=Test_prophet.rename(columns={'Date':'ds','Volume_WSE':'y'})

    from datetime import datetime

    Train_prophet.ds=Train_prophet['ds'].dt.tz_localize(None)
    Test_prophet.ds=Test_prophet['ds'].dt.tz_localize(None)
    best_param=hyper_tune(Train_prophet,Test_prophet)
    model_prophet=Prophet(**best_param)
    Train_prophet=df_train
    Train_prophet=Train_prophet.rename(columns={'Date':'ds','Volume_WSE':'y'})
    Train_prophet.ds=Train_prophet['ds'].dt.tz_localize(None)
    model_prophet.fit(Train_prophet)

    future_data = model_prophet.make_future_dataframe( periods=24, freq='m', include_history=False)
    forecast_data=model_prophet.predict(future_data)

    forecast_data[['ds', 'yhat', 'yhat_lower','yhat_upper']].plot(x='ds',y='yhat',figsize=(10,8))
    ax = (Test_prophet.plot(x='ds',y='y',figsize=(10,5),title='Actual Vs Forecast'))
    forecast_data.plot(x='ds',y='yhat',figsize=(10,5),title='Actual vs Forecast', ax=ax)
    return forecast_data


# In[ ]:


Forecast_data=prophet('Not Applicable_Mid')


# In[ ]:


Forecast_data[['ds','yhat']].plot()


# In[ ]:




