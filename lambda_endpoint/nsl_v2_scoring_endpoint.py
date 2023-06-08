import pickle
import re
# from nltk.corpus import stopwords
# import spacy
import numpy as np
import pandas as pd
import xgboost
import json

# stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",
#                "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 
#                "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 
#                'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 
#                'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 
#                'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 
#                'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 
#                'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 
#                'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 
#                'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 
#                'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', 
#                "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', 
#                "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', 
#                "won't", 'wouldn', "wouldn't"]
# nlp = spacy.load("en_core_web_sm")


def load_models():
    file_name = "nsql_model_v2.json"
    model = xgboost.XGBClassifier()
    model.load_model(file_name)

    model.save_model("nsql_model_v2_copy.json")

    return model

def convert_nulls_to_one_format(df:pd.DataFrame):
    for col in df.columns:
        idx = df.index[df[col].isnull()].tolist()
        idx.extend(df.index[df[col].isna()].tolist())
        idx.extend(df.index[df[col] == ''].tolist())
        idx.extend(df.index[df[col] == '[]'].tolist())
        idx = list(set(idx))
        df.loc[idx, col] = None
    return df

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


def get_predictions(df):
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
    print(nsql_model)
    print(nsql_model.predict(df[independent_variables]))
    y_prob = nsql_model.predict_proba(df[independent_variables])
    print(y_prob)
    
    # return score
    return y_prob[0,1]

def lambda_handler(event, context):
    data = json.loads(event['body'])
    
    needed_keys = ['application_id', 'estimated_monthly_revenue', 'incoming_ach_payments', 'outgoing_ach_and_checks', 
     'check_deposit_amount', 'outgoing_wire_transfers', 'incoming_wire_transfer', 'business_type', 
     'email_domain', 'current_bank', 'industry_category_name', 'iovation_device_type', 'iovation_device_timezone', 
     'carrier', 'socure_sigma', 'socure_phonerisk', 'socure_emailrisk', 'socure_reason_code', 'socure_phonerisk_reason_code', 
     'socure_emailrisk_reason_code', 'screen_width_mean', 'screen_height_mean']
    
    for key in needed_keys:
        if key not in data.keys():
            print("Key Missing")
            return {
                'statusCode': 400,
                'body': json.dumps({
                    'error': 'All parameters needed for scoring not present',
                    'message': f"please include {needed_keys}" 
                })
            }
    
    if 'application_id' in data.keys():
        try:
            score = get_predictions(pd.DataFrame.from_dict(data, orient='index').T)
        except:
            print("Scoring Failed")
            return {
                'statusCode': 400,
                'body': json.dumps({
                    'error': "Data not in expected format",
                    'message': """Check Values sent for scoring - sample values - 
                            {"application_id":"35b9598f-1d7b-4e03-b52d-5a3c7a883ada","estimated_monthly_revenue":"$5K +",
                            "incoming_ach_payments":"$5K +","outgoing_ach_and_checks":"$5K +",
                            "check_deposit_amount":"$5K +","outgoing_wire_transfers":"$0",
                            "incoming_wire_transfer":"<$1K","business_type":"llc","email_domain":"gmail.com",
                            "current_bank":"WELLS-FARGO","industry_category_name":"Manufacturing",
                            "iovation_device_type":"IPHONE","iovation_device_timezone":"300","carrier":"AT&T",
                            "socure_sigma":0.818,"socure_phonerisk":0.583,"socure_emailrisk":0.705,
                            "socure_reason_code":"[\n  \"I610\",\n  \"I626\",\n  \"I711\",\n  \"R559\",\n  \"I632\",\n  \"I705\",\n  \"I631\",\n  \"I553\",\n  \"I611\",\n  \"I614\",\n  \"R610\",\n  \"I636\",\n  \"I630\",\n  \"I708\",\n  \"I618\",\n  \"I555\",\n  \"I707\",\n  \"I602\"\n]",
                            "socure_phonerisk_reason_code":"[\n  \"I610\",\n  \"I626\",\n  \"I632\",\n  \"I620\",\n  \"I631\",\n  \"I611\",\n  \"I614\",\n  \"I636\",\n  \"I630\",\n  \"I618\",\n  \"I602\"\n]","socure_emailrisk_reason_code":"[\n  \"R559\",\n  \"I520\",\n  \"I553\",\n  \"I555\"\n]",
                            "screen_width_mean":414.0,"screen_height_mean":776.0}"""
                })
            }
    else:
        score = 0
    
    print(f"score: {score}")
    return {
        'statusCode': 200,
        'body': json.dumps({'probability': str(score)})
    }

if __name__ == '__main__':
    event = {
        'body': json.dumps({"application_id":"35b9598f-1d7b-4e03-b52d-5a3c7a883ada","estimated_monthly_revenue":"$5K +",
                            "incoming_ach_payments":"$5K +","outgoing_ach_and_checks":"$5K +",
                            "check_deposit_amount":"$5K +","outgoing_wire_transfers":"$0",
                            "incoming_wire_transfer":"<$1K","business_type":"llc","email_domain":"gmail.com",
                            "current_bank":"WELLS-FARGO","industry_category_name":"Manufacturing",
                            "iovation_device_type":"IPHONE","iovation_device_timezone":"300","carrier":"AT&T",
                            "socure_sigma":0.818,"socure_phonerisk":0.583,"socure_emailrisk":0.705,
                            "socure_reason_code":"[\n  \"I610\",\n  \"I626\",\n  \"I711\",\n  \"R559\",\n  \"I632\",\n  \"I705\",\n  \"I631\",\n  \"I553\",\n  \"I611\",\n  \"I614\",\n  \"R610\",\n  \"I636\",\n  \"I630\",\n  \"I708\",\n  \"I618\",\n  \"I555\",\n  \"I707\",\n  \"I602\"\n]",
                            "socure_phonerisk_reason_code":"[\n  \"I610\",\n  \"I626\",\n  \"I632\",\n  \"I620\",\n  \"I631\",\n  \"I611\",\n  \"I614\",\n  \"I636\",\n  \"I630\",\n  \"I618\",\n  \"I602\"\n]","socure_emailrisk_reason_code":"[\n  \"R559\",\n  \"I520\",\n  \"I553\",\n  \"I555\"\n]",
                            "screen_width_mean":414.0,"screen_height_mean":776.0})
    }

    print(lambda_handler(event, {}))


