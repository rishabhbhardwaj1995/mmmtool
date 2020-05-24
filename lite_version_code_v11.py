import pandas as pd
import re
import numpy as np
import statsmodels.formula.api as sm
import scipy.special as ssp

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import statsmodels.api as smm
#import plotly.plotly as py
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import json
import datetime
import operator
import os
import base64
import io
import sqlite3





############################################# Functions ##############################################


adstock_prefix_conditions = ['tv', 'adx']

adstock_suffix_conditions = []



monthly = {'adstock_1':['adstock_1',0.0625], 'adstock_2':['adstock_2',0.25], 'adstock_3':['adstock_3',0.3969],
                 'adstock_4':['adstock_4',0.5], 'adstock_5':['adstock_5',0.5743], 'adstock_6':['adstock_6',0.63],
                 'adstock_7':['adstock_7',0.6730], 'adstock_8':['adstock_8',0.7071]}

weekly = {'adstock_1':['adstock_1',0.5], 'adstock_2':['adstock_2',0.71], 'adstock_3':['adstock_3',0.79],
                'adstock_4':['adstock_4',0.84], 'adstock_5':['adstock_5',0.87], 'adstock_6':['adstock_6',0.89],
                 'adstock_7':['adstock_7',0.91], 'adstock_8':['adstock_8',0.92]}


daily = {'adstock_1':['adstock_1',0.91], 'adstock_2':['adstock_2',0.95], 'adstock_3':['adstock_3',0.97],
                 'adstock_4':['adstock_4',0.98], 'adstock_5':['adstock_5',0.98], 'adstock_6':['adstock_6',0.98],
                 'adstock_7':['adstock_7',0.99], 'adstock_8':['adstock_8',0.99]}



adstock_dict = {}



def urlify(s):
    s = re.sub(r"[^\w\s]", '', s)
    s = re.sub(r"\s+", '_', s)
    return s

def lag(df, n):
    new_columns = ["{}_lag{:02d}".format(variable, n) for variable in df.columns]
    new_df = df.shift(n)
    new_df.columns = new_columns
    return new_df

def lagged_dataframe(df, lags=1):
    data_frames = [df]
    data_frames.extend([lag(df, i) for i in range(1, lags + 1)])
    return pd.concat(data_frames, axis=1)




def diffdata(df):
    n = len(df)
    newcolumns = ["{}_diff01".format(variable) for variable in df.columns]
    newdf=pd.DataFrame(np.diff(df, axis = 0))
    kdata = newdf.append(pd.Series(), ignore_index = True)
    kdata["new"]=range(1, len(kdata)+1)
    kdata.ix[n, 'new']=0
    kdata = kdata.sort_values('new').reset_index(drop='True')
    kdata.drop("new", axis = 1, inplace = True)
    kdata.columns=newcolumns
    return kdata


def diff_lag(df,k):
    dt=diffdata(df)
    finaldata=lagged_dataframe(dt,lags=k)
    return finaldata

def lag_diff(df,k):
    dt=lagged_dataframe(df, lags=k)
    finaldata=diffdata(dt)
    return finaldata


def completedata(df,k):
    n= len(df)
    ldata=lagged_dataframe(df, lags=k)
    newcolumns = ["{}_diff01".format(variable) for variable in df.columns]
    newdf=pd.DataFrame(np.diff(df, axis = 0))
    kdata = newdf.append(pd.Series(), ignore_index = True)
    kdata["new"]=range(1, len(kdata)+1)
    kdata.ix[n, 'new']=0
    kdata = kdata.sort_values('new').reset_index(drop='True')
    kdata.drop("new", axis = 1, inplace = True)
    kdata.columns=newcolumns
    #ldata=lagged_dataframe(df, lags=k)
    data_frame= ldata.join(kdata)
    return data_frame


def unusual_columns(df):
    df.fillna(0, inplace= True)
    cols=list(df.columns)
    new_cols = list(map(urlify, cols))
    new_cols = [x.lower() for x in new_cols]
    df.columns=new_cols
    type = pd.DataFrame(df.dtypes)
    type['col_names'] = df.columns
    unusual_cols =[]
    for i in range(len(type)):
        if (type.iloc[i,0]!='float64' and type.iloc[i,0]!='int64'):
            unusual_cols.append(type.iloc[i,1])        
    
    return unusual_cols


def adstock_dataframe(df, adstock_lambda,adstock_prefix_conditions, adstock_suffix_conditions):
    df.fillna(0, inplace= True)
    cols=list(df.columns)
    new_cols = list(map(urlify, cols))
    new_cols = [x.lower() for x in new_cols]
    df.columns=new_cols
    names_cols = list(df.columns)
    adstock_dict = {}
    unusual_cols = unusual_columns(df)
    for i in unusual_cols:
        names_cols.remove(i)
    adstock_1 = []
    for i in names_cols:
        for j in adstock_prefix_conditions:
            if (j in i):
                adstock_1.append(i)
    adstock_final_cols = []
    
    if len(adstock_suffix_conditions) != 0 :
        for i in adstock_1:
            for j in adstock_suffix_conditions:
                if (j in i):
                    adstock_final_cols.append(i)
    else:
        adstock_final_cols = adstock_1 
    
    adstock_df= df[adstock_final_cols]
    
    #addataframe = pd.DataFrame(columns = adstock_final_cols)
    
    for j in range(len(adstock_lambda.columns)):
        keyn = str(adstock_lambda.iloc[0,j])
        new_columns = list(adstock_df.columns)
        addataframe = pd.DataFrame(columns = new_columns)
        for i in range(len(adstock_df)):
            if i == 0:
                val= adstock_df.iloc[i,:]
                addataframe = addataframe.append(val, ignore_index = True)
                print('yes')
            else:
                val = (val*float(adstock_lambda.iloc[1,j])) + adstock_df.iloc[i,:]
                addataframe = addataframe.append(val, ignore_index = True)
        
        new_columns_up = ["{}_{:s}".format(variable, str(adstock_lambda.iloc[0,j])) for variable in adstock_df.columns]
        addataframe.columns = new_columns_up
        addataframe['tmp'] = [_ for _ in range(len(addataframe))]
        adstock_dict[keyn] = addataframe
    
    key_names = list(adstock_dict.keys())
    row = [i for i in range(len(df))]
    final_adstock_df = pd.DataFrame()
    final_adstock_df['tmp'] = row
    
    for i in key_names:
        data=adstock_dict[i]
        final_adstock_df = pd.merge(final_adstock_df, data, on = ['tmp'])
        
    return final_adstock_df

def logit(df):
    dfa = df.copy()
    dfaa =np.power((sum(np.square(np.array(dfa)))/(len(dfa)-1)), 0.5)
    for i in range(len(dfaa)):
        dfa.iloc[:,i] = dfa.iloc[:,i]/dfaa[i]
    dfa = ssp.expit(dfa)
    return dfa
    

def merged_data(df,n):
    if str(n) == 'daily':
        stock_details = daily
        lg = 30
        print(stock_details)
        
    elif str(n) == 'weekly':
        stock_details = weekly
        lg = 8
        print(stock_details)
        
    elif str(n) == 'monthly':
        lg = 2
        stock_details = monthly
        print(stock_details)
        
    adstock_lambda = pd.DataFrame(stock_details)
    unusual_col_names=unusual_columns(df)
    columns_for_lag_diff = [i for i in list(df.columns) if i not in unusual_col_names]
    df_diff_lag = df[columns_for_lag_diff]
    diff_lag_data = completedata(df_diff_lag, lg)
    diff_lag_data['tmp'] = [_i for _i in range(len(df))]
    adstock_data = adstock_dataframe(df,adstock_lambda ,adstock_prefix_conditions, adstock_suffix_conditions)
    final_master_data = pd.merge(adstock_data, diff_lag_data, on = ['tmp'])
    final_master_data.fillna(0, inplace= True)
    final_master_data.drop(columns = ['tmp'], inplace = True)
#    final_master_data.to_csv('final_actual_data.csv')
    final_actual = final_master_data.copy()
    final_logit_data = logit(final_actual)
#    final_logit_data.to_csv('final_logit_data.csv')
    return final_logit_data, final_master_data



def summary_stats(df, tg):
    raw_df = df 
    if tg == 'daily':
        n = 365
    elif tg == 'weekly':
        n = 52
    elif tg == 'monthly':
        n = 12
    df.columns = df.columns.str.lower()
    date_columns = [i for i in df.columns if 'date' in i]
    date_columns = df[date_columns]       
    df,dfa = merged_data(df,tg)    
    l2 = np.square(dfa)
    l2 = pd.DataFrame(l2.sum())
    l2.columns = ['k_value']
    l2['k_value'] = l2['k_value']/(len(dfa)-1)
    l2['k_value'] = np.power(l2['k_value'], 0.5)
    a = pd.DataFrame(df.max())
    a.columns = ['max']
    b = pd.DataFrame(df.min())
    b.columns=['min']
    c = pd.DataFrame(df.mean())
    c.columns= ['overall_average']
    d = pd.DataFrame(df.sum())
    d.columns= ['total_sum']
    d1 = pd.DataFrame(dfa.sum())
    d1.columns= ['total_sum_absolute']
    e = pd.DataFrame(dfa.astype(bool).sum(axis=0))
    e.columns = ['non_zero']
    f = pd.concat([d,e], axis = 1)
    f1 = pd.DataFrame()
    f1['active_average'] = (f['total_sum']-(0.5*(len(df)- f['non_zero'])))/f['non_zero']
    
    df_ap_last_year = df.tail(n)
    df_app_last_year = dfa.tail(n)    
    g = pd.DataFrame(df_ap_last_year.max())
    g.columns = ['max_last_year']
    h = pd.DataFrame(df_ap_last_year.min())
    h.columns=['min_last_year']
    i = pd.DataFrame(df_ap_last_year.mean())
    i.columns= ['overall_average_last_year']
    j = pd.DataFrame(df_ap_last_year.sum())
    j.columns= ['total_sum_last_year']
    j1 = pd.DataFrame(df_app_last_year.sum())
    j1.columns= ['total_sum_last_year_absolute']
    k = pd.DataFrame(df_app_last_year.astype(bool).sum(axis=0))
    k.columns = ['non_zero_last_year']
    l = pd.concat([j,k], axis = 1)
    l1 = pd.DataFrame()
    l1['active_average_last_year'] = (l['total_sum_last_year']-(0.5*(len(df_ap_last_year)- l['non_zero_last_year'])))/l['non_zero_last_year']
    summary= pd.concat([b,c,f1,i,l1,e,g,a,h,j,j1,k,d,d1,l2], axis = 1)
    summary.loc['Intercept'] = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    summary['var_names'] = [str(i) for i in summary.index]
#    summary.to_csv('summary.csv')
    dfa = pd.concat([dfa, date_columns], axis = 1)
    df['null'] = 0.5
    return summary ,df,dfa, raw_df 





def indx(par_df):
    list_con = ['_lag01', '_lag04', '_lag02', '_lag03', '_diff01', '_adstock01', '_adstock1', '_adstock02', '_adstock2', '_adstock03', '_adstock3', '_adstock04', '_adstock4', '_adstock05', '_adstock5', '_adstock06', '_adstock6', '_adstock07', '_adstock7', '_adstock08', '_adstock8', '_ad1', '_ad2', '_ad3', '_ad4', '_ad5', '_ad6', '_ad7','_ad8', '_adstock_1', '_adstock_2',
          '_adstock_3', '_adstock_4', '_adstock_5', '_adstock_6', '_adstock_7', '_adstock_8']

    new_idx = []
    for i in range(len(par_df)):
        k = str(par_df.index[i])
        print(k)
        for j in list_con:
            if j in k:
                k = k[:(-1*len(j))]
                new_idx.append(k)            
    old_idx = [par_df.index[l] for l in range(len(par_df))]
    for j in range(len(old_idx)):
        for i in new_idx:
            if i in (old_idx[j]):
                old_idx[j] = i
    return old_idx
    




def indx2(par_df, col_name):
    list_con = ['_lag01', '_lag04', '_lag02', '_lag03', '_diff01', '_adstock01', '_adstock1', '_adstock02', '_adstock2', '_adstock03', '_adstock3', '_adstock04', '_adstock4', '_adstock05', '_adstock5', '_adstock06', '_adstock6', '_adstock07', '_adstock7', '_adstock08', '_adstock8', '_ad1', '_ad2', '_ad3', '_ad4', '_ad5', '_ad6', '_ad7','_ad8', '_adstock_1', '_adstock_2',
          '_adstock_3', '_adstock_4', '_adstock_5', '_adstock_6', '_adstock_7', '_adstock_8']

    new_col = []
    tr_list = par_df[str(col_name)]
    for i in range(len(par_df)):
        k = str(tr_list[i])
        print(k)
        for j in list_con:
            if j in k:
                k = k[:(-1*len(j))]
                new_col.append(k)            
    old_idx = [tr_list[l] for l in range(len(par_df))]
    for j in range(len(old_idx)):
        for i in new_col:
            if i in (old_idx[j]):
                old_idx[j] = i
    return old_idx




def contribution(par_df, summ_df ,dependent_variable, p_val, custom_index, base_form):    
    old_idx = indx(par_df)
    par_df['new_idx'] = old_idx
    par_df.set_index('new_idx', inplace = True)    
    summary = summ_df
    kval = summary.loc[str(dependent_variable), 'k_value']
    print(kval)
    par_df['logit_min'] = [summary.loc[i,'min'] for i in (par_df.index) ]
    par_df['logit_overall_average'] = [summary.loc[i,'overall_average'] for i in (par_df.index) ]
    par_df['logit_active_average'] = [summary.loc[i,'active_average'] for i in (par_df.index) ]
    par_df['logit_overall_average_last_year'] = [summary.loc[i,'overall_average_last_year'] for i in (par_df.index) ]
    par_df['logit_active_average_last_year'] = [summary.loc[i,'active_average_last_year'] for i in (par_df.index) ]


    
    x=0
    
    for i in range(len(par_df)):
        j = custom_index[i]
        x += par_df.iloc[i,0]*par_df.iloc[i,j]

    
    par_df['predicted_value'] = x
    predicted_oa = par_df['predicted_value']    

    predicted_condition = []
    for i in range(len(par_df)):
        k=custom_index[i]
        val = par_df.iloc[i,0]*par_df.iloc[i,1]
        val1 = par_df.iloc[i,0]*par_df.iloc[i,k]
        val2 = sum(par_df.iloc[:,0]*par_df.iloc[:,2])
        val2 = val2 - val1 + val
        predicted_condition.append(val2)
    val3 = sum(par_df.iloc[:,0]*par_df.iloc[:,1])
    predicted_condition[0] = val3
    
    par_df['predicted_condition'] = predicted_condition
    
    predicted_oa_actual = []
    for i in range(len(par_df)):
        v = np.log(predicted_oa[i]/(1-predicted_oa[i]))*kval
        predicted_oa_actual.append(v)
    
    par_df['predicted_oa_actual']= predicted_oa_actual
    
    predicted_condition_actual = []
    for i in range(len(par_df)):
        v = np.log(predicted_condition[i]/(1-predicted_condition[i]))*kval
        predicted_condition_actual.append(v)
    par_df['predicted_condition_actual']= predicted_condition_actual
        
    percentage_unormalized = []
    for i in range(len(par_df)):              
        val = (par_df.iloc[i,8]-par_df.iloc[i,9])/par_df.iloc[i,9]
        percentage_unormalized.append(val)
        
    par_df['percentage_unormalized'] = percentage_unormalized
    par_df['percentage_normalized'] =percentage_unormalized/np.sum(percentage_unormalized)
    
    if base_form == 1:
        percentage_unormalized[0] = (par_df.iloc[0,8]-par_df.iloc[0,9])/par_df.iloc[0,9]
    elif base_form == 2:
        percentage_unormalized[0] = (1-(par_df.iloc[0,8]-par_df.iloc[0,9])/par_df.iloc[0,9])
    elif base_form == 3:
        percentage_unormalized[0] = (par_df.iloc[0,8]-par_df.iloc[0,9])/par_df.iloc[0,8]
    
    par_df['percentage_normalized'] =(percentage_unormalized/np.sum(percentage_unormalized))*100
    old_idxx = indx(p_val)
    p_val['new_idx'] = old_idxx
    p_val.set_index('new_idx', inplace = True)
    par_df = pd.concat([par_df, p_val], axis = 1)
    par_df['variables'] = [str(i) for i in par_df.index]
    par_df.to_csv('cal_file.csv')
    par_dff = par_df.iloc[:,[-1, 0, -2, -3]]
    par_dff = pd.DataFrame(par_dff)
    par_dff.columns = ['var_names', 'estimates', 'p-values', 'contribution']
    del par_dff.index.name      
    return par_dff, par_df



def media_mix_summ(depen ,df):
    cols = df.columns
    spends_cols = [i for i in cols if 'spend' in i]
    spends_cols.append(str(depen))
    spends_df = df[spends_cols]
    total = sum(spends_df.values)
    column_names = pd.DataFrame(spends_cols, columns = ['variables'])
    total = pd.DataFrame(total, columns = ['total_spends'])
    final_df = pd.concat([column_names, total], axis = 1)
    return (final_df)




def roi (dfa, roi_list, ind_l, depen, par, typ, ref):
    ref.to_csv('ref.csv')
    roi_df = pd.DataFrame(roi_list, columns = ['variables'])
    ind_df = pd.DataFrame(ind_l, columns = ['Contributing Variables'])
    comb_df = pd.concat([roi_df, ind_df], axis = 1)
    comb_df['Contributing Variables'] = indx2(comb_df, 'Contributing Variables')
    roi_data =  comb_df.merge(ref, on = 'variables', how = 'left')
    par['Contributing Variables'] = par['variables']
    par = pd.DataFrame(par.loc[:,['Contributing Variables', 'percentage_normalized']])
    par.to_csv('par1.csv')
    roi_data.to_csv('roi_data1.csv')
    roi_data = roi_data.merge(par, on = 'Contributing Variables', how = 'left')
    main_kpi = float(ref.iloc[-1, 0]) 
    roi_val = []
    roi_data.to_csv('roi_data2.csv')
    if typ == 1:
        for i in range(len(roi_data)):
            contri = float(roi_data.iloc[i,-1])
            spends = float(roi_data.iloc[i,-2])
            val = (contri*main_kpi)/(100*spends)
            roi_val.append(val)
    if typ == 2:
        for i in range(len(roi_data)):
            contri = float(roi_data.iloc[i,-1])
            spends = float(roi_data.iloc[i,-2])
            val = 1/((contri*main_kpi)/(100*spends))
            roi_val.append(val)
    roi_data['ROI/CPA'] = roi_val
    return (roi_data)






############################################ Application ############################################







########################################### Layout ##################################################


app = dash.Dash(__name__, 
                external_stylesheets= [dbc.themes.LUX],
                suppress_callback_exceptions=True)



app.layout = html.Div([
    
    html.Div([html.H1('Market Mix Modelling Tool')],style={'textAlign': 'center'}),
    html.Br(),
    html.Hr(),
    html.Br(),
    html.Div([dash_table.DataTable(id='table',persistence = True, persistence_type = 'session' ,persisted_props = 'data')], style={'display': 'none'}),    
    html.Div([dash_table.DataTable(id='table2',persistence = True, persistence_type = 'session' ,persisted_props = 'data')], style={'display': 'none'}),
    html.Div([dash_table.DataTable(id='table3',persistence = True, persistence_type = 'session' ,persisted_props = 'data')], style={'display': 'none'}),
    html.Div([dash_table.DataTable(id='table4',persistence = True, persistence_type = 'session' ,persisted_props = 'data')], style={'display': 'none'}),
    html.Div([dash_table.DataTable(id='table5',persistence = True, persistence_type = 'session' ,persisted_props = 'data')], style={'display': 'none'}),
    html.Div([dash_table.DataTable(id='table6',persistence = True, persistence_type = 'session' ,persisted_props = 'data')], style={'display': 'none'}),


html.Br(),
html.H4('File Upload'),

html.Div([

    
    dbc.Row([
             dbc.Col(
        
            dcc.Upload(
            id='upload-data',
            children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '50%',
            'height': '30px',
            'lineHeight': '30px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        multiple=False)),
    
   dbc.Col(dcc.Dropdown(
           id = 'timegrain',
           options=[{'label': 'daily', 'value': 'daily'},
                    {'label': 'weekly', 'value': 'weekly'},
                    {'label': 'monthly', 'value': 'monthly'}],
           placeholder = 'Time Level',
           style = {'width': '50%'}         
           ))]),
        ]),
        
 
    
html.Div( [
               html.Br(),
               html.H4('Media Mix'),
            
            
                    dash_table.DataTable(
                    id='media_mix',
                    style_cell={'text-align':'center'},
                    style_table={'overflowY': 'scroll'})], 
            style={'padding': '40px 5px'}

             ),       
    
        


 html.Div([  
        html.Button(
        id='propagate-button2',
        n_clicks=0,
        children='Populate Columns')
        ]),
    
    html.Div([
                
         dbc.Row([
                 
            dbc.Col(
                
                html.Div([
                        dcc.Dropdown(
                                id='plot_dependent',
                                placeholder = 'Select Variable to Plot',
                                persistence = True,
                                persistence_type = 'session'
                                
                                )])),
   
             
            dbc.Col(
                
                html.Div([
                        dcc.Dropdown(
                                id='plot_independent',
                                placeholder = 'Select Variable to Plot',
                                persistence = True,
                                persistence_type = 'session'
                                
                                )])),

                 ])]),
     


 

           html.Div([
                   
                   dcc.Graph(
                           
                     id='crossfilter-indicator-scatter'
                     )], style={'display': 'inline-block',
                                'width' : '49%'}),
        
        
           html.Div([
                                      
                   dcc.Graph(
                     
                     id='crossfilter-indicator-scatter2'
                     )], style={'display': 'inline-block',
                                'width' : '49%',}),
   
        
        
        
        
        
        
    html.Br(),
    html.Br(),
    html.Hr(),
    html.Br(),

        
        
        html.Div([html.H4('Regression Equation'),
                
         dbc.Row([
                 
            dbc.Col(
                
                html.Div([
                        dcc.Dropdown(
                                id='dependent-column',
                               # options=[{'label': i, 'value': i} for i in available_indicators],
                                placeholder = 'Dependent Variable',
                                persistence = True,
                                persistence_type = 'session'
                                )
                      ] )),
             
            dbc.Col(
                
                html.Div([
                        dcc.Dropdown(
                                id='base',
                                options=[{'label': 'standard', 'value': 1},
                                         {'label': '1-standard', 'value': 2},
                                         {'label': 'drop from average', 'value': 3}],
                                placeholder = 'baseline formula',
                                value = 1,
                                persistence = True,
                                persistence_type = 'session'
                                )])),

                 ]),

        dbc.Row([
                 
            dbc.Col(
                
                html.Div([
                        dcc.Dropdown(
                                id='tv-column',
                                #options=[{'label': i, 'value': i} for i in available_indicators],
                                placeholder = 'Select Inependent Variable',
                                value = 'null',
                                persistence = True,
                                persistence_type = 'session'
                                )])),
            dbc.Col(
                
                html.Div([
                        dcc.Dropdown(
                                id='l1',
                                options=[{'label': 'overall average', 'value': 2},
                                         {'label': 'active average', 'value': 3},
                                         {'label': 'l1Y average', 'value': 4},
                                         {'label': 'l1y active average', 'value': 5}],
                                placeholder = 'logit choice',
                                value = 2,
                                persistence = True,
                                persistence_type = 'session'
                                )])),

                 ]),
         
        dbc.Row([
                 
            dbc.Col(
                
                html.Div([
                        dcc.Dropdown(
                                id='var1',
                               # options=[{'label': i, 'value': i} for i in available_indicators],
                                placeholder = 'Select Inependent Variable',
                                value = 'null',
                                persistence = True,
                                persistence_type = 'session'
                                )])),
            dbc.Col(
                
                html.Div([
                        dcc.Dropdown(
                                id='l2',
                                options=[{'label': 'overall average', 'value': 2},
                                         {'label': 'active average', 'value': 3},
                                         {'label': 'l1Y average', 'value': 4},
                                         {'label': 'l1y active average', 'value': 5}],
                                placeholder = 'logit choice',
                                value = 2,
                                persistence = True,
                                persistence_type = 'session'
                                )])),

                 ]),

       dbc.Row([
                 
            dbc.Col(
                
                html.Div([
                        dcc.Dropdown(
                                id='var2',
                                #options=[{'label': i, 'value': i} for i in available_indicators],
                                placeholder = 'Select Inependent Variable',
                                value = 'null',
                                persistence = True,
                                persistence_type = 'session'
                                )])),
            dbc.Col(
                
                html.Div([
                        dcc.Dropdown(
                                id='l3',
                                options=[{'label': 'overall average', 'value': 2},
                                         {'label': 'active average', 'value': 3},
                                         {'label': 'l1Y average', 'value': 4},
                                         {'label': 'l1y active average', 'value': 5}],
                                placeholder = 'logit choice',
                                value = 2,
                                persistence = True,
                                persistence_type = 'session'
                                )])),

                 ]),
       
       dbc.Row([
                 
            dbc.Col(
                
                html.Div([
                        dcc.Dropdown(
                                id='var3',
                                #options=[{'label': i, 'value': i} for i in available_indicators],
                                placeholder = 'Select Inependent Variable',
                                value = 'null',
                                persistence = True,
                                persistence_type = 'session'
                                )])),
            dbc.Col(
                
                html.Div([
                        dcc.Dropdown(
                                id='l4',
                                options=[{'label': 'overall average', 'value': 2},
                                         {'label': 'active average', 'value': 3},
                                         {'label': 'l1Y average', 'value': 4},
                                         {'label': 'l1y active average', 'value': 5}],
                                placeholder = 'logit choice',
                                value = 2,
                                persistence = True,
                                persistence_type = 'session'
                                )])),

                 ]),

       dbc.Row([
                 
            dbc.Col(
                
                html.Div([
                        dcc.Dropdown(
                                id='var4',
                                #options=[{'label': i, 'value': i} for i in available_indicators],
                                placeholder = 'Select Inependent Variable',
                                value = 'null',
                                persistence = True,
                                persistence_type = 'session'
                                )])),
            dbc.Col(
                
                html.Div([
                        dcc.Dropdown(
                                id='l5',
                                options=[{'label': 'overall average', 'value': 2},
                                         {'label': 'active average', 'value': 3},
                                         {'label': 'l1Y average', 'value': 4},
                                         {'label': 'l1y active average', 'value': 5}],
                                placeholder = 'logit choice',
                                value = 2,
                                persistence = True,
                                persistence_type = 'session'
                                )])),

                 ]),
        
        dbc.Row([
                 
            dbc.Col(
                
                html.Div([
                        dcc.Dropdown(
                                id='var5',
                                #options=[{'label': i, 'value': i} for i in available_indicators],
                                placeholder = 'Select Inependent Variable',
                                value = 'null',
                                persistence = True,
                                persistence_type = 'session'
                                )])),
            dbc.Col(
                
                html.Div([
                        dcc.Dropdown(
                                id='l6',
                                options=[{'label': 'overall average', 'value': 2},
                                         {'label': 'active average', 'value': 3},
                                         {'label': 'l1Y average', 'value': 4},
                                         {'label': 'l1y active average', 'value': 5}],
                                placeholder = 'logit choice',
                                value = 2,
                                persistence = True,
                                persistence_type = 'session'
                                )])),

                 ]),
        
        dbc.Row([
                 
            dbc.Col(
                
                html.Div([
                        dcc.Dropdown(
                                id='var6',
                                #options=[{'label': i, 'value': i} for i in available_indicators],
                                placeholder = 'Select Inependent Variable',
                                value = 'null',
                                persistence = True,
                                persistence_type = 'session'
                                )])),
            dbc.Col(
                
                html.Div([
                        dcc.Dropdown(
                                id='l7',
                                options=[{'label': 'overall average', 'value': 2},
                                         {'label': 'active average', 'value': 3},
                                         {'label': 'l1Y average', 'value': 4},
                                         {'label': 'l1y active average', 'value': 5}],
                                placeholder = 'logit choice',
                                value = 2,
                                persistence = True,
                                persistence_type = 'session'
                                )])),

                 ]),
        
        dbc.Row([
                 
            dbc.Col(
                
                html.Div([
                        dcc.Dropdown(
                                id='var7',
                                #options=[{'label': i, 'value': i} for i in available_indicators],
                                placeholder = 'Select Inependent Variable',
                                value = 'null',
                                persistence = True,
                                persistence_type = 'session'
                                )])),
            dbc.Col(
                
                html.Div([
                        dcc.Dropdown(
                                id='l8',
                                options=[{'label': 'overall average', 'value': 2},
                                         {'label': 'active average', 'value': 3},
                                         {'label': 'l1Y average', 'value': 4},
                                         {'label': 'l1y active average', 'value': 5}],
                                placeholder = 'logit choice',
                                value = 2,
                                persistence = True,
                                persistence_type = 'session'
                                )])),

                 ]),
        
        dbc.Row([
                 
            dbc.Col(
                
                html.Div([
                        dcc.Dropdown(
                                id='var8',
                                #options=[{'label': i, 'value': i} for i in available_indicators],
                                placeholder = 'Select Inependent Variable',
                                value = 'null',
                                persistence = True,
                                persistence_type = 'session'
                                )])),
            dbc.Col(
                
                html.Div([
                        dcc.Dropdown(
                                id='l9',
                                options=[{'label': 'overall average', 'value': 2},
                                         {'label': 'active average', 'value': 3},
                                         {'label': 'l1Y average', 'value': 4},
                                         {'label': 'l1y active average', 'value': 5}],
                                placeholder = 'logit choice',
                                value = 2,
                                persistence = True,
                                persistence_type = 'session'
                                )])),

                 ]),
        
        dbc.Row([
                 
            dbc.Col(
                
                html.Div([
                        dcc.Dropdown(
                                id='var9',
                                #options=[{'label': i, 'value': i} for i in available_indicators],
                                placeholder = 'Select Inependent Variable',
                                value = 'null',
                                persistence = True,
                                persistence_type = 'session'
                                )])),
            dbc.Col(
                
                html.Div([
                        dcc.Dropdown(
                                id='l10',
                                options=[{'label': 'overall average', 'value': 2},
                                         {'label': 'active average', 'value': 3},
                                         {'label': 'l1Y average', 'value': 4},
                                         {'label': 'l1y active average', 'value': 5}],
                                placeholder = 'logit choice',
                                value = 2,
                                persistence = True,
                                persistence_type = 'session'
                                )])),

                 ]),
             

          ]),
    
                 

    html.Div( [
               html.H4('Regression Results'),
            
            
                    dash_table.DataTable(
                    id='datatable-interactivity',
                    style_cell={'text-align':'center'},
                    style_table={'overflowY': 'scroll'})], 
            style={'padding': '20px 5px'}

             ),


   
    html.Div([
            html.Br(),
            html.H4('Correlation Matrix of Model Variables'),
        
                   dash_table.DataTable(
                   id='correlation_matrix',
                   style_cell={'text-align':'center'},
                   style_table={'overflowY': 'scroll'}
                   ) 
           
            ], 
            style={'padding': '30px 5px'}),
  
html.Br(), 
html.Hr(),
html.Br(),    

 html.Div([html.H4('ROI Calculation'),
         
         dbc.Row([
                 
                 dbc.Col(
                         html.Button(id='propagate-button',
                                     n_clicks=0,
                                     children='Populate Columns'),
                                     
                                    ),
                 dbc.Col(dcc.Dropdown(
                                id='roi_typ',
                                options=[{'label': 'ROI', 'value': 1},
                                         {'label': 'Cost Per Acquisition', 'value': 2}],
                                placeholder = 'ROI/CPA',
                                persistence = True,
                                persistence_type = 'session',
                                style = {'width': '50%'}
                                ))]),
                
            
        dbc.Row([
                 
            dbc.Col(
                
                html.Div([
                        dcc.Dropdown(
                                id='roi_independent1',
                                placeholder = 'Select Variable to Plot',
                                persistence = True,
                                persistence_type = 'session'
                                
                                )])),
   
             
            dbc.Col(
                
                html.Div([
                        dcc.Dropdown(
                                id='spends_independent1',
                                placeholder = 'Select Variable to Plot',
                                persistence = True,
                                persistence_type = 'session'
                                
                                )])),

                 ]),
                    
         dbc.Row([
                 
            dbc.Col(
                
                html.Div([
                        dcc.Dropdown(
                                id='roi_independent2',
                                placeholder = 'Select Variable to Plot',
                                persistence = True,
                                persistence_type = 'session'
                                
                                )])),
   
             
            dbc.Col(
                
                html.Div([
                        dcc.Dropdown(
                                id='spends_independent2',
                                placeholder = 'Select Variable to Plot',
                                persistence = True,
                                persistence_type = 'session'
                                
                                )])),

                 ]),
                    
        dbc.Row([
                 
            dbc.Col(
                
                html.Div([
                        dcc.Dropdown(
                                id='roi_independent3',
                                placeholder = 'Select Variable to Plot',
                                persistence = True,
                                persistence_type = 'session'
                                
                                )])),
   
             
            dbc.Col(
                
                html.Div([
                        dcc.Dropdown(
                                id='spends_independent3',
                                placeholder = 'Select Variable to Plot',
                                persistence = True,
                                persistence_type = 'session'
                                
                                )])),
                 ]),
                    
            html.Br(),
            
            dash_table.DataTable(
                   id='roi_table',
#                   editable=True,
#                   filter_action="native",
#                   sort_action="native",
#                   sort_mode="multi",
#                   page_action="native",
                   style_cell={'text-align':'center'},
                   style_table={'overflowY': 'scroll'}
                   
                   )
            
                    
                    
                    
                    
        ], style={'padding': '30px 5px'}),


 html.Br(),
 html.Hr(),
 html.Br(),         
            
      
       
          html.Div([
        
                   dash_table.DataTable(
                   id='datatable-interactivity2',
                   editable=True,
                   filter_action="native",
                   sort_action="native",
                   sort_mode="multi",
                   page_action="native",
                   style_cell={'text-align':'center'},
                   style_table={'overflowY': 'scroll'}
                   
                   ) 
           
            ], style={'padding': '30px 5px'}),
                    
        


            
   ])

html.Footer('Copyright &copy; 2022')


##################################### Call Backs #####################################################


# file upload function

# Functions

# file upload function
                   
            
def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))

    except Exception as e:
        print(e)
        return None

    return df






# callback table creation
@app.callback ([Output('table', 'data'),
                Output('table2', 'data'),
                Output('table3', 'data'),
                Output('table4', 'data')],
              [Input('upload-data', 'contents'),
               Input('upload-data', 'filename'),
               Input('timegrain', 'value')])
def update_output(contents, filename, tg):
    if contents is not None:
        df = parse_contents(contents, filename)
        if df is not None:
            summary, dfl, dfa, df_raw = summary_stats(df,str(tg))
            return  dfl.to_dict('records'), summary.to_dict('records'), dfa.to_dict('records'), df_raw.to_dict('records')  
        else:
            return [{}]
    else:
        return [{}]



@app.callback(
    [Output('table6', 'columns'),
     Output('table6', 'data')],
    [Input('table4', 'data'),
     Input('plot_dependent', 'value')])

def media_mix_raw(dat, indep):
    dat = pd.DataFrame(dat)
    dff = media_mix_summ(str(indep) ,dat)
    dff.to_csv('media_mix1.csv')
    columns=[
            {"name": i, "id": i, "deletable": True, "selectable": True} for i in dff.columns
        ]
    data=dff.to_dict('records')     
    return columns, data 




@app.callback(
    [Output('media_mix', 'columns'),
     Output('media_mix', 'data')],
    [Input('table6', 'data')])

def media_mix(dat):
    dat = pd.DataFrame(dat)
    dat = dat.sort_values(['total_spends'], ascending = False)
    dat['total_spends'] = ["{:,}".format(int(i)) for i in dat['total_spends']]
    cols = list(dat.columns)
    cols = [cols[-1]] + cols[:-1]
    dat = dat[cols]
    columns=[
            {"name": i, "id": i, "deletable": True, "selectable": True} for i in dat.columns
        ]
    data=dat.to_dict('records')     
    return columns, data 


@app.callback(Output('dependent-column', 'options'),
              [Input('propagate-button2', 'n_clicks'),
               Input('table', 'data')])
def update_filter_column_options1(n_clicks_update, tablerows):
    if n_clicks_update < 1:
        print ("df empty")
        return []

    else:
        dff = pd.DataFrame(tablerows)
        print ("updating... dff empty?:", dff.empty) #result is True, labels stay empty

        return [{'label': i, 'value': i} for i in sorted(list(dff))]




@app.callback(Output('tv-column', 'options'),
              [Input('propagate-button2', 'n_clicks'),
               Input('table', 'data')])
def update_filter_column_options2(n_clicks_update, tablerows):
    if n_clicks_update < 1:
        print ("df empty")
        return []

    else:
        dff = pd.DataFrame(tablerows)
        print ("updating... dff empty?:", dff.empty) #result is True, labels stay empty

        return [{'label': i, 'value': i} for i in sorted(list(dff))]



@app.callback(Output('var1', 'options'),
              [Input('propagate-button2', 'n_clicks'),
               Input('table', 'data')])
def update_filter_column_options3(n_clicks_update, tablerows):
    if n_clicks_update < 1:
        print ("df empty")
        return []

    else:
        dff = pd.DataFrame(tablerows)
        return [{'label': i, 'value': i} for i in sorted(list(dff))]


@app.callback(Output('var2', 'options'),
              [Input('propagate-button2', 'n_clicks'),
               Input('table', 'data')])
def update_filter_column_options4(n_clicks_update, tablerows):
    if n_clicks_update < 1:
        print ("df empty")
        return []

    else:
        dff = pd.DataFrame(tablerows)
        return [{'label': i, 'value': i} for i in sorted(list(dff))]


@app.callback(Output('var3', 'options'),
              [Input('propagate-button2', 'n_clicks'),
               Input('table', 'data')])
def update_filter_column_options5(n_clicks_update, tablerows):
    if n_clicks_update < 1:
        print ("df empty")
        return []

    else:
        dff = pd.DataFrame(tablerows)
        return [{'label': i, 'value': i} for i in sorted(list(dff))]




@app.callback(Output('var4', 'options'),
              [Input('propagate-button2', 'n_clicks'),
               Input('table', 'data')])
def update_filter_column_options6(n_clicks_update, tablerows):
    if n_clicks_update < 1:
        print ("df empty")
        return []

    else:
        dff = pd.DataFrame(tablerows)        
        return [{'label': i, 'value': i} for i in sorted(list(dff))]


@app.callback(Output('var5', 'options'),
              [Input('propagate-button2', 'n_clicks'),
               Input('table', 'data')])
def update_filter_column_options7(n_clicks_update, tablerows):
    if n_clicks_update < 1:
        print ("df empty")
        return []

    else:
        dff = pd.DataFrame(tablerows)        
        return [{'label': i, 'value': i} for i in sorted(list(dff))]

@app.callback(Output('var6', 'options'),
              [Input('propagate-button2', 'n_clicks'),
               Input('table', 'data')])
def update_filter_column_options8(n_clicks_update, tablerows):
    if n_clicks_update < 1:
        print ("df empty")
        return []

    else:
        dff = pd.DataFrame(tablerows)        
        return [{'label': i, 'value': i} for i in sorted(list(dff))]

@app.callback(Output('var7', 'options'),
              [Input('propagate-button2', 'n_clicks'),
               Input('table', 'data')])
def update_filter_column_options9(n_clicks_update, tablerows):
    if n_clicks_update < 1:
        print ("df empty")
        return []

    else:
        dff = pd.DataFrame(tablerows)        
        return [{'label': i, 'value': i} for i in sorted(list(dff))]

@app.callback(Output('var8', 'options'),
              [Input('propagate-button2', 'n_clicks'),
               Input('table', 'data')])
def update_filter_column_options10(n_clicks_update, tablerows):
    if n_clicks_update < 1:
        print ("df empty")
        return []

    else:
        dff = pd.DataFrame(tablerows)        
        return [{'label': i, 'value': i} for i in sorted(list(dff))]

@app.callback(Output('var9', 'options'),
              [Input('propagate-button2', 'n_clicks'),
               Input('table', 'data')])
def update_filter_column_options11(n_clicks_update, tablerows):
    if n_clicks_update < 1:
        print ("df empty")
        return []

    else:
        dff = pd.DataFrame(tablerows)        
        return [{'label': i, 'value': i} for i in sorted(list(dff))]




@app.callback(Output('plot_dependent', 'options'),
              [Input('propagate-button2', 'n_clicks'),
               Input('table3', 'data')])
def update_filter_tab_2(n_clicks_update, tablerows):
    if n_clicks_update < 1:
        print ("df empty")
        return []

    else:
        dff = pd.DataFrame(tablerows)
        
        return [{'label': i, 'value': i} for i in sorted(list(dff))]


@app.callback(Output('plot_independent', 'options'),
              [Input('propagate-button2', 'n_clicks'),
               Input('table3', 'data')])
def update_filter_tab_2_2(n_clicks_update, tablerows):
    if n_clicks_update < 1:
        print ("df empty")
        return []

    else:
        dff = pd.DataFrame(tablerows)
        
        return [{'label': i, 'value': i} for i in sorted(list(dff))]


@app.callback(Output('roi_independent1', 'options'),
              [Input('propagate-button', 'n_clicks'),
               Input('datatable-interactivity', 'data')])
def roi_dropdown1(n_clicks_update, tablerows):
    if n_clicks_update < 1:
        print ("df empty")
        return []
    else:
        dff = pd.DataFrame(tablerows)
        li =  [dff.iloc[i,3] for i in range(1,len(dff))]
        print(li)
        return [{'label': i, 'value': i} for i in sorted(li)]




@app.callback(Output('spends_independent1', 'options'),
              [Input('propagate-button', 'n_clicks'),
               Input('media_mix', 'data')])
def roi_dropdown2(n_clicks_update, tablerows):
    if n_clicks_update < 1:
        print ("df empty")
        return []

    else:
        dff = pd.DataFrame(tablerows)
        li =  [dff.iloc[i,1] for i in range(1,len(dff))]
        print(li)
        return [{'label': i, 'value': i} for i in sorted(li)]
    

@app.callback(Output('roi_independent2', 'options'),
              [Input('propagate-button', 'n_clicks'),
               Input('datatable-interactivity', 'data')])
def roi_dropdown3(n_clicks_update, tablerows):
    if n_clicks_update < 1:
        print ("df empty")
        return []
    else:
        dff = pd.DataFrame(tablerows)
        li =  [dff.iloc[i,3] for i in range(1,len(dff))]
        print(li)
        return [{'label': i, 'value': i} for i in sorted(li)]


@app.callback(Output('spends_independent2', 'options'),
              [Input('propagate-button', 'n_clicks'),
               Input('media_mix', 'data')])
def roi_dropdown4(n_clicks_update, tablerows):
    if n_clicks_update < 1:
        print ("df empty")
        return []

    else:
        dff = pd.DataFrame(tablerows)
        li =  [dff.iloc[i,1] for i in range(1,len(dff))]
        print(li)
        return [{'label': i, 'value': i} for i in sorted(li)]


@app.callback(Output('roi_independent3', 'options'),
              [Input('propagate-button', 'n_clicks'),
               Input('datatable-interactivity', 'data')])
def roi_dropdown5(n_clicks_update, tablerows):
    if n_clicks_update < 1:
        print ("df empty")
        return []

    else:
        dff = pd.DataFrame(tablerows)
        li =  [dff.iloc[i,3] for i in range(1,len(dff))]
        return [{'label': i, 'value': i} for i in sorted(li)]


@app.callback(Output('spends_independent3', 'options'),
              [Input('propagate-button', 'n_clicks'),
               Input('media_mix', 'data')])
def roi_dropdown6(n_clicks_update, tablerows):
    if n_clicks_update < 1:
        print ("df empty")
        return []

    else:
        dff = pd.DataFrame(tablerows)
        li =  [dff.iloc[i,1] for i in range(1,len(dff))]
        print(li)
        return [{'label': i, 'value': i} for i in sorted(li)]


@app.callback([Output('roi_table', 'columns'),
               Output('roi_table', 'data')],
              [Input('roi_independent1', 'value'),
               Input('roi_independent2', 'value'),
               Input('roi_independent3', 'value'),
               Input('spends_independent1', 'value'),
               Input('spends_independent2', 'value'),
               Input('spends_independent3', 'value'),
               Input('table5', 'data'),
               Input('table3', 'data'),
               Input('dependent-column', 'value'),
               Input('table6', 'data'),
               Input('roi_typ', 'value')])

def roi_table(r1, r2, r3, s1, s2, s3, dat1, dat2, depen, med, typ):
    rol_l = [str(s1), str(s2), str(s3)]
    ind_l = [str(r1), str(r2), str(r3)]
    dfa = pd.DataFrame(dat2)
    par = pd.DataFrame(dat1)
    media_mix = pd.DataFrame(med)
    roi_df = roi(dfa, rol_l, ind_l, str(depen), par, int(typ), media_mix)
    roi_df = pd.DataFrame(roi_df)
    columns=[
            {"name": i, "id": i, "deletable": True, "selectable": True} for i in roi_df.columns
        ]
    data= roi_df.to_dict('records')
    return columns, data
    







@app.callback(
    [Output('datatable-interactivity', 'columns'),
     Output('datatable-interactivity', 'data'),
     Output('table5', 'columns'),
     Output('table5', 'data')],
    [Input('tv-column', 'value'),
     Input('dependent-column', 'value'),
     Input('var1', 'value'),
     Input('var2', 'value'),
     Input('var3', 'value'),
     Input('var4', 'value'),
     Input('l1', 'value'),
     Input('l2', 'value'),
     Input('l3', 'value'),
     Input('l4', 'value'),
     Input('l5', 'value'),
     Input('base', 'value'),
     Input('table', 'data'),
     Input('table2', 'data'),
     Input('var5', 'value'),
     Input('var6', 'value'),
     Input('var7', 'value'),
     Input('var8', 'value'),
     Input('var9', 'value'),
     Input('l6', 'value'),
     Input('l7', 'value'),
     Input('l8', 'value'),
     Input('l9', 'value'),
     Input('l10', 'value'),
     ])

def regression(tv_var, depen, var1, var2, var3, var4, l1, l2, l3, l4, l5, base, dat, summ, var5, var6, var7, var8, var9, l6, l7, l8, l9, l10):
#    dat = pd.read_json(dat, orient='split') # <- problem! dff stays empty even though table was uploaded
#    dat = pd.DataFrame(dat)
    dat = pd.DataFrame(dat)
    summ = pd.DataFrame(summ)
    summ.set_index('var_names', inplace = True) 
    del summ.index.name
    logit_index = [2, int(l1), int(l2), int(l3), int(l4), int(l5), int(l6), int(l7), int(l8), int(l9), int(l10)]
    string_list = [str(tv_var), str(var1), str(var2), str(var3), str(var4), str(var5), str(var6), str(var7), str(var8), str(var9)]
    string_list = [i for i in string_list if i != 'null']
    form1 = " + ".join(string_list)
    formula = str(depen)+ ' ' + '~' + form1  
    results = sm.ols(formula, data=dat).fit()
    df_p = pd.DataFrame(results.pvalues)
    print(df_p)
    df_e = pd.DataFrame(results.params)
    con, con2 = contribution(df_e, summ , str(depen), df_p, logit_index, base)
    resu = pd.DataFrame(con)
    columns=[
            {"name": i, "id": i, "deletable": True, "selectable": True} for i in resu.columns
        ]
    data=resu.to_dict('records') 
    resu2 = pd.DataFrame(con2)
    columns2=[
            {"name": i, "id": i, "deletable": True, "selectable": True} for i in resu2.columns
        ]
    data2=resu2.to_dict('records') 
    return columns, data, columns2, data2 
    



@app.callback(
    [Output('correlation_matrix', 'columns'),
     Output('correlation_matrix', 'data')],
    [Input('tv-column', 'value'),
     Input('dependent-column', 'value'),
     Input('var1', 'value'),
     Input('var2', 'value'),
     Input('var3', 'value'),
     Input('var4', 'value'),
     Input('table', 'data'),
     Input('var5', 'value'),
     Input('var6', 'value'),
     Input('var7', 'value'),
     Input('var8', 'value'),
     Input('var9', 'value')
     ])

def matrix(tv_var, depen, var1, var2, var3, var4, dat, var5, var6, var7, var8, var9):
    string_list = [depen , str(tv_var), str(var1), str(var2), str(var3), str(var4), str(var5), str(var6), str(var7), str(var8), str(var9)]
    string_list = [i for i in string_list if i != 'null']
    df= pd.DataFrame(dat)
    data_for_matrix = df[string_list]  
    matrix = data_for_matrix.corr()
    matrix['variable'] = [i for i in matrix.index]
    cols = list(matrix.columns)
    cols = [cols[-1]] + cols[:-1]
    matrix = matrix[cols]
    columns=[
            {"name": i, "id": i, "deletable": True, "selectable": True} for i in matrix.columns
        ]
    data=matrix.to_dict('records')    
    return columns, data 




# Tab 2 callback
    
    
    
@app.callback(
    dash.dependencies.Output('crossfilter-indicator-scatter', 'figure'),    
    [Input('plot_dependent', 'value'),
     Input('table3', 'data')]
    )

def plot(depen, data):
    # Create figure with secondary y-axis
    
    #dfa = pd.DataFrame(data)
#    dfa = pd.read_json(data, orient='split') # <- problem! dff stays empty even though table was uploaded
    dfa = pd.DataFrame(data)

    
    fig1 = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add traces
    
    fig1.add_trace(
    go.Scatter(x= dfa['date'], y=dfa[str(depen)], name="yaxis data"),
    secondary_y=False,
    )
    
    # Margins
    fig1.update_layout(
    margin=dict(l=10, r=0, t= 40, b= 0))

    # Set x-axis title
    fig1.update_xaxes(title_text="Date", tickfont = dict(size = 10))
    
    # Set y-axes titles
    fig1.update_yaxes(title_text="Y axis", secondary_y=False)
    #fig_code = fig.to_json()
    
    return(fig1)


@app.callback(
    dash.dependencies.Output('crossfilter-indicator-scatter2', 'figure'),    
    [Input('plot_independent', 'value'),
     Input('table3', 'data')]
    )

def plot2(indepen, data):
    # Create figure with secondary y-axis
    
    dfa = pd.DataFrame(data)
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add traces
    
    fig.add_trace(
    go.Scatter(x= dfa['date'], y=dfa[str(indepen)], name="yaxis data"),
    secondary_y=False,
    )
    
    
    # Margins
    fig.update_layout(
    margin=dict(l=10, r=0, t= 40, b= 40))
    
    # Set x-axis title
    fig.update_xaxes(title_text="Date", tickfont = dict(size = 10))
    
    # Set y-axes titles
    fig.update_yaxes(title_text="Y axis", secondary_y=False)
    #fig_code = fig.to_json()
    
    return(fig) 



@app.callback(
    [Output('datatable-interactivity2', 'columns'),
     Output('datatable-interactivity2', 'data')],    
    [Input('plot_dependent', 'value'),
     Input('table', 'data')]
    )

def cormatrix(depen, data):
    df = pd.DataFrame(data)
    mat = df.corr()
    mat['var'] = [i for i in mat.index]
    mat = mat[['var', str(depen)]]
    mat = mat.sort_values([str(depen)], ascending = False)
#    mat = pd.DataFrame(mat[str(depen)]
    columns=[
            {"name": i, "id": i, "deletable": True, "selectable": True} for i in mat.columns
        ]
    data=mat.to_dict('records')     
    return columns, data 


#@app.callback(
#    [Output('roi_table', 'columns'),
#     Output('roi_table', 'data')],    
#    [Input('plot_dependent', 'value'),
#     Input('table', 'data')]
#    )
#
#def roi_update(depen, data):
#    df = pd.DataFrame(data)
#    mat = df.corr()
#    mat['var'] = [i for i in mat.index]
#    mat = mat[['var', str(depen)]]
#    mat = mat.sort_values([str(depen)], ascending = False)
##    mat = pd.DataFrame(mat[str(depen)]
#    columns=[
#            {"name": i, "id": i, "deletable": True, "selectable": True} for i in mat.columns
#        ]
#    data=mat.to_dict('records')     
#    return columns, data 




if __name__ == '__main__':
    app.run_server(debug=False)