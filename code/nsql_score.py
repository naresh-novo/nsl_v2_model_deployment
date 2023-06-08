#!/usr/bin/env python
# coding: utf-8

# In[35]:


# load libraries -------------------------------------------------------------------------------------------------------
# path variables
import sys
project_path = '/Users/naresh/Downloads/DS/growth/nsl_v2/nsl_v2_deployment/'
sys.path.insert(0, project_path+'conf')

# core libraries
import warnings
warnings.filterwarnings("ignore")
import json
import pickle
import datetime
import pandas as pd
import numpy as np


# In[36]:


# load models ----------------------------------------------------------------------------------------------------------
def load_models():
    # deposit Intent Model
    file_name = "nsql_model_v2.pkl"
    path = project_path+"models/"
    model = pickle.load(open(path+file_name, "rb"))

    # return models
    return model


# In[37]:


def convert_nulls_to_one_format(df:pd.DataFrame):
    for col in df.columns:
        idx = df.index[df[col].isnull()].tolist()
        idx.extend(df.index[df[col].isna()].tolist())
        idx.extend(df.index[df[col] == ''].tolist())
        idx.extend(df.index[df[col] == '[]'].tolist())
        idx = list(set(idx))
        df.loc[idx, col] = None
    return df


# In[38]:


def impute_nulls_zeros(df:pd.DataFrame):
    ################## APPLICATIONS ##################
    df['estimated_monthly_revenue'] = df['estimated_monthly_revenue'].fillna('$1k +')
    df['incoming_ach_payments'] = df['incoming_ach_payments'].fillna('<$1k')
    df['outgoing_ach_and_checks'] = df['outgoing_ach_and_checks'].fillna('<$1k')
    df['check_deposit_amount'] = df['check_deposit_amount'].fillna('<$1k')
    df['outgoing_wire_transfers'] = df['outgoing_wire_transfers'].fillna('$0')
    df['incoming_wire_transfer'] = df['incoming_wire_transfer'].fillna('$0')
    df['business_type'] = df['business_type'].fillna('llc')
    df['email_domain'] = df['email_domain'].fillna('gmail.com')
    df['current_bank'] = df['current_bank'].fillna('novo-is-first')
    df['industry_category_name'] = df['industry_category_name'].fillna('retail trade')
    ################## ALLOY ##################
    df['iovation_device_type'] = df['iovation_device_type'].fillna('iphone')
    df['iovation_device_timezone'] = df['iovation_device_timezone'].fillna('300')
    df['carrier'] = df['carrier'].fillna('t-mobile usa')
    df['socure_sigma'] = df['socure_sigma'].fillna(0.557)
    df['socure_phonerisk'] = df['socure_phonerisk'].fillna(0.408)
    df['socure_emailrisk'] = df['socure_emailrisk'].fillna(0.879)
    df['socure_reason_code'] = df['socure_reason_code'].fillna("""[\n  "I610",\n  "R566",\n  "I711",\n  "I632",\n  "I705",\n  "I625",\n  "I611",\n  "I614",\n  "I636",\n  "R561",\n  "I630",\n  "I708",\n  "I618",\n  "I555",\n  "I707",\n  "I602"\n]""")
    df['socure_phonerisk_reason_code'] = df['socure_phonerisk_reason_code'].fillna("""[\n  "I610",\n  "I632",\n  "I620",\n  "I625",\n  "I611",\n  "I614",\n  "I636",\n  "I630",\n  "I618",\n  "I602"\n]""")
    df['socure_emailrisk_reason_code'] = df['socure_emailrisk_reason_code'].fillna("""[\n  "I520",\n  "I553",\n  "I556",\n  "I555"\n]""")
    ################## SEGMENT ##################
    df['screen_width_mean'] = df['screen_width_mean'].fillna(1092.0)
    df['screen_width_mean'] = np.where(df['screen_width_mean']==0, 1092.0, df['screen_width_mean'])
    df['screen_height_mean'] = df['screen_height_mean'].fillna(751.0)
    df['screen_height_mean'] = np.where(df['screen_height_mean']==0, 751.0, df['screen_height_mean'])
    
    return df
    


# In[39]:


# derived variables ----------------------------------------------------------------------------------------------------
def derive_variables(df:pd.DataFrame):
   
    ################## APPLICATIONS ##################
    # estimated business numbers
    estimated_cols = ['estimated_monthly_revenue', 'incoming_ach_payments', 'outgoing_ach_and_checks', 
                      'check_deposit_amount', 'outgoing_wire_transfers', 'incoming_wire_transfer']

    # grouping all responses into 5K+ and 5K-
    for col in estimated_cols:
        df[col] = df[col].str.lower()
        df[col] = np.where(df[col].isin(['$5k +', '$50k +']), 1, 0)

    # business type
    df['business_type'] = df['business_type'].str.lower()    
    df['business_group'] = np.where(df['business_type'] == 'sole_proprietorship', 0, 1)

    # current bank
    df['current_bank'] = df['current_bank'].str.lower()    
    hdb_group = ['bluevine', 'other-national-bank', 'td-ank', 'chase', 'usaa']
    df['current_bank_group'] = np.where(df['current_bank'].isin(hdb_group), 1, 0)

    # email domain
    email_domain_group = ['gmail.com', 'yahoo.com', 'outlook.com', 'icloud.com', 'protonmail.com',
                          'ymail.com', 'me.com', 'hotmail.com', 'aol.com', 'msn.com', 'gmx.com', 'rocketmail.com', 
                          'comcast.net', 'mac.com', 'pm.me', 'mail.com', 'att.net', 'smartmomstravelagents.com', 
                          'live.com', 'proton.me', 'kw.com', 'usa.com', 'exprealty.com', 'verizon.net', 'email.com', 
                          'zohomail.com', 'bellsouth.net', 'sbcglobal.net']
    df['email_domain'] = df['email_domain'].str.lower()    
    df['email_domain_bucket'] = np.where(df['email_domain'].isin(email_domain_group), 0, 1)
        
    # industry type
    df['industry_category_name'] = df['industry_category_name'].str.lower()
    
    df['industry_category_name_1'] = np.where(df['industry_category_name'
                                                ]=='professional, scientific, and technical services', 1, 0)
    df['industry_category_name_2'] = np.where(df['industry_category_name']=='real estate rental and leasing', 1, 0)
    df['industry_category_name_3'] = np.where(df['industry_category_name']=='retail trade', 1, 0)
    df['industry_category_name_4'] = np.where(df['industry_category_name']=='manufacturing', 1, 0)
    df['industry_category_name_5'] = np.where(
        df['industry_category_name']=='administrative and support and waste management and remediation services', 1, 0)

    
    ################## ALLOY ##################
    # iovation_device_type
    df['iovation_device_type'] = df['iovation_device_type'].str.lower()
    df['iovation_device_type_mac'] = np.where(df['iovation_device_type']=='mac', 1, 0)
    df['iovation_device_type_android'] = np.where(df['iovation_device_type']=='android', 1, 0)
    
    # iovation_device_timezone
    df['iovation_device_timezone'] = df['iovation_device_timezone'].str.lower()
    df['iovation_device_timezone_480'] = np.where(df['iovation_device_timezone']=='480', 1, 0)

    # carrier
    df['carrier'] = df['carrier'].str.lower()
    df['carrier_tmobile'] = np.where(df['carrier'].str.contains('t-mobile*'), 1, 0)

    # socure reason codes
    df['socure_emailrisk_reason_code_i553'] = np.where(df['socure_emailrisk_reason_code'
                                                         ].str.contains("i553", case=False, na=False), 1, 0)
    df['socure_emailrisk_reason_code_i566'] = np.where(df['socure_emailrisk_reason_code'
                                                         ].str.contains("i566", case=False, na=False), 1, 0)
    df['socure_emailrisk_reason_code_r561'] = np.where(df['socure_emailrisk_reason_code'
                                                         ].str.contains("r561", case=False, na=False), 1, 0)
    df['socure_phonerisk_reason_code_i630'] = np.where(df['socure_phonerisk_reason_code'
                                                         ].str.contains("i630", case=False, na=False), 1, 0)
    df['socure_phonerisk_reason_code_i614'] = np.where(df['socure_phonerisk_reason_code'
                                                         ].str.contains("i614", case=False, na=False), 1, 0)
    df['socure_phonerisk_reason_code_r616'] = np.where(df['socure_phonerisk_reason_code'
                                                         ].str.contains("r616", case=False, na=False), 1, 0)
    df['socure_reason_code_r207'] = np.where(df['socure_reason_code'].str.contains("r207", case=False,na=False), 1, 0)
    

    ################## SEGMENT ##################
    df['sh_sw_ratio_mean'] = df['screen_height_mean']/df['screen_width_mean']
    
    # return dataframe
    return df


# In[40]:


# get predictions ------------------------------------------------------------------------------------------------------
def get_predictions(df:pd.DataFrame):
    #load model
    nsql_model = load_models()
    
    # get derived variables
    df = convert_nulls_to_one_format(df)
    df = impute_nulls_zeros(df)
    df = derive_variables(df)
    
    # predictors
    independent_variables = [
                             'estimated_monthly_revenue',
                             'incoming_ach_payments',
                             'screen_width_mean',
                             'socure_emailrisk_reason_code_i553',
                             'iovation_device_type_mac',
                             'industry_category_name_1',
                             'sh_sw_ratio_mean',
                             'outgoing_ach_and_checks',
                             'business_group',
                             'industry_category_name_2',
                             'socure_sigma',
                             'socure_emailrisk',
                             'industry_category_name_3',
                             'iovation_device_type_android',
                             'check_deposit_amount',
                             'socure_phonerisk_reason_code_i630',
                             'socure_emailrisk_reason_code_r561',
                             'socure_emailrisk_reason_code_i566',
                             'outgoing_wire_transfers',
                             'socure_phonerisk',
                             'socure_phonerisk_reason_code_i614',
                             'socure_reason_code_r207',
                             'iovation_device_timezone_480',
                             'industry_category_name_4',
                             'current_bank_group',
                             'incoming_wire_transfer',
                             'socure_phonerisk_reason_code_r616',
                             'carrier_tmobile',
                             'industry_category_name_5',
                             'email_domain_bucket'
                            ]
    # get probabilities
    y_prob = nsql_model.predict_proba(df[independent_variables])
    
    # return score
    return y_prob[:,1:].flatten()


# In[41]:


# main -----------------------------------------------------------------------------------------------------------------
def main(argv1):
    
    # get input json
    with open(project_path + '/data/' + argv1, "r") as read_file:
        input_data = pd.DataFrame([json.load(read_file)])
    
    # get prediction
    input_data['nsql_prob'] = get_predictions(input_data)

    # return prediction
    output_data = input_data[['application_id', 'nsql_prob']].to_dict('records')[0]
    with open(project_path + 'data/nsql_sample_output.json', 'w') as fp:
        json.dump(output_data, fp)
        


# In[ ]:


if __name__ == '__main__':
    main()

