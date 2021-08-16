import pandas                      as pd
import numpy                       as np


def mimic_cleanup(df, live_prediction=False, postcode_csv='australian_postcodes.csv',df_mode_path='df_mode.csv', verbose=True):
    '''
    preprocessing of the data frames
    
    Parameters
    ----------
    df : Pandas dataframe
        Input dataframe
    
    live_prediction : bool 
        indicates whether we are preparing data for prediction purpose the model or not
        
    postcode_csv : str 
        path to the australian post code mapping table   
        
    df_mode_path : str
        path to the most frequency columns data .csv
   
    verbose : bool
        Displaying process completion steps

    Returns
    -------
    df : pandas dataframe
        preprocessed Dataframe  
    '''
    steps = 11
    current_step = 0  
    # list of columns which are selected as the One hot encoding scheme.
    one_hot_cols    = ['csource', 'cadmitclass', 'state']   
    
    
    # -----------------------------------------
    #            renaming the columns 
    # -----------------------------------------  
    rename_dict = {'recordTime': 'dt', 'cVisitNumber': 'visit','cAdmissionNumber':'visit','URNumber': 'urno', 'Age': 'age','Sex': 'sex', 
                   'Indig': 'indig','cAdmitDateTime': 'cadttext', 'cWard': 'cward','cBed': 'cbed', 'cUnit': 'cunit','cSource': 'csource',
                    'cAdmitClass': 'cadmitclass','cPvtHealth': 'cpvthlth', 'cAdmitMBS': 'cadmitmbs','cExptStay': 'cexpstay',
                    'eVisitNumber': 'evisit','eAdmissionNumber':'evisit','eComplaintCode': 'ecomplaintcode','eComplaint': 'ecomplaint',
                    'pUnit': 'punit','eTriageCat': 'etriage','cICU': 'cicu','cTheatre': 'ctheatre', 'pVisitNumber': 'pvisit','pAdmissionDate':'padmdtext',
                    'pLengthOfStay': 'plos','IndexOfComorbidity': 'pcomorbid', 'ageonadmission': 'ageadmission','AgeRelatedRisk': 'agerisk',
                    'CombinedScore': 'combinedscore','Estimated10YearSurvival': 'survive10', 'c_MI': 'c_mi','c_chf': 'c_chf', 'c_pvd': 'c_pvd',
                    'c_cvd': 'c_cvd','c_dementia': 'c_dementia', 'c_cpd': 'c_cpd','c_ctd': 'c_ctd', 'c_pud': 'c_pud',
                    'c_mld': 'c_mld','c_dmnc': 'c_dmnc', 'c_dmcc': 'c_dmcc','c_hemi': 'c_hemi', 'c_renal': 'c_renal',
                    'c_cancer': 'c_cancer','c_ld': 'c_ld', 'c_metca': 'c_metca','c_aids': 'c_aids', 'pP1': 'pdiag',
                    'pP1Desc': 'pdiagd','pProc1': 'pproc', 'pProcDesc': 'pprocd','pCall': 'pcall', 'pICU': 'picu','pH': 'ph', 'BiCarb': 'bicarb',
                    'Lactate': 'lactate','Hb': 'hb', 'WCC': 'wcc','Creatinine': 'creatinine', 'Platelets': 'platelets',
                    'PostCode': 'postcode','AdvantageAndDisadvantageDecile': 'sadd', 'RelativeSocioEconomicDisadvantageDecile': 'srsd',
                    'EconomicResourcesDecile': 'ser','EducationAndOccupationDecile': 'seo', 'ipInLast1Years': 'ip12m',                    
                    'ipInLast2Years': 'ip24m','EDInLast1Years': 'ed12m', 'EDInLast2Years': 'ed24m','isAtRisk': 'atrisk', 'uniqueID': 'id',
                    'pFrailtyScore': 'frailty'
                     }
    # rename the columns to what the model was learned
    df.rename(columns=rename_dict,inplace=True)
    if verbose: 
        current_step +=1
        print(f'({current_step}/{steps}) Completed renaming columns')  
    
    # -----------------------------------------
    # keeping only the admission record for each visit ID
    # -----------------------------------------    
    df.drop_duplicates(subset=['visit'], keep = 'first', inplace = True)
        
    if verbose: 
        current_step +=1
        print(f'({current_step}/{steps}) Completed keeping the admission record only ')
        
    # -----------------------------------------
    # preparing data for training purpose only 
    # -----------------------------------------    
    if not live_prediction:
        # update values of cb and met columns to indicate whether an emergency call (MET / CB) occurred or not.
        df.loc[df['met'] > 0, 'met'] = 1
        df.loc[df['cb'] > 0, 'cb'] = 1

        # Feature Engineering for Length of Stay (LOS) target variable
        # Convert admission and discharge times to datatime type
        # Convert timedelta type into float 'days', 86400 seconds in a day
        df['admitdt'] = pd.to_datetime(df['admitdt'])
        df['dischargedt'] = pd.to_datetime(df['dischargedt'])
        df['LOS'] = (df['dischargedt'] - df['admitdt']).dt.total_seconds()/86400

        # Mark admissions where patients died in boolean column    
        df['died'] = df['died'].map(dict(Yes=1, No=0))
        prediction_output_list = ['met','cb', 'anycall','died', 'LOS']
        if verbose: 
            current_step +=1
            print(f'({current_step}/{steps}) Completed training purpose data engineering')   
            
    # -----------------------------------------
    #            filtering the columns 
    # -----------------------------------------  
    # filtering the columns based on live data columns
    columns_filter_list = ['dt','visit','urno','age','sex','indig','cadttext','cward','cbed','cunit','csource','cadmitclass',
                           'cpvthlth','cadmitmbs','cexpstay','evisit','ecomplaintcode','ecomplaint','etriage','cicu',
                           'ctheatre','pvisit','padmdtext','punit','plos','pcomorbid','ageadmission','agerisk',
                           'combinedscore','survive10','c_mi','c_chf','c_pvd','c_cvd','c_dementia','c_cpd',
                           'c_ctd','c_pud','c_mld','c_dmnc','c_dmcc','c_hemi','c_renal','c_cancer','c_ld',
                           'c_metca','c_aids','pdiag','pdiagd','pproc','pprocd','pcall','picu','ph','bicarb',
                           'lactate','hb','wcc','creatinine','platelets','postcode','sadd','srsd','ser','seo',
                           'ip12m','ip24m','ed12m','ed24m','atrisk','frailty']
    
    # following list are columns that have too much missing values in live data
    live_data_miss_columns = ['ed24m','ed12m','ip24m','ip12m','c_cpd','c_chf','c_pvd',
                                'c_cvd','c_dementia','c_ld','c_pud','c_metca','c_ctd',
                                'c_cancer','c_renal','c_hemi','c_dmcc','c_dmnc','c_aids','c_mi','c_mld']
    
    columns_filter_list = [col for col in columns_filter_list if col not in live_data_miss_columns]
    
    if not live_prediction:
        columns_filter_list += prediction_output_list
    df = df.filter(columns_filter_list)
    if verbose: 
        current_step +=1
        print(f'({current_step}/{steps}) Completed filtering columns')  
        
    # -----------------------------------------
    #     postcode (Feature Engineering) 
    # -----------------------------------------
    # Import CSV tables
    df_pc = pd.read_csv(postcode_csv)  
    df_pc = df_pc.drop_duplicates(subset=['postcode'])  # you can use take_last=True

    
    df['postcode'] = df['postcode'].fillna(0)
    df['postcode'] = df['postcode'].astype('int')
    
    # Merge postcode data with hospital data
    # delete postcode dataframe which not needed anymore
    df = df.merge(df_pc[['postcode','state', 'sa4','sa4name','Lat_precise','Long_precise']], on="postcode", how='left')
    del df_pc
    if verbose: 
        current_step +=1
        print(f'({current_step}/{steps}) Completed postcode (Feature Engineering).')
    
    # -----------------------------------------
    #          Feature Engineering 
    # -----------------------------------------
    # sex/male feature engineering
    conditions = [
    df['sex'].eq('F'),
    df['sex'].eq('M')
    ]

    choices = [0,1]

    df['male'] = np.select(conditions, choices, default=0) # [R, f, W, X, G, I, m] --> are considered as Female    
    
    # Admission Type (Feature Engineering)   
    # handle missing data based on elective info
    col = 'etriage'    
    if col in df.columns:
        if 'elective' in df.columns.tolist():
            df[col] = df[col][df.elective==0].fillna(0) # ******  Not elective but missed  value! **************
            df[col] = df[col].fillna(6)                 # elective
            
        else:
            # etriage missing value handling
            df[col] = df[col].fillna(0)
            
            # Adding 'elective' column
            # create a list of our conditions
            conditions = [
                (df[col] == 0),
                (df[col] > 0)
                ] 
            
            # create a list of the values we want to assign for each condition
            values = [0, 1]
            
            # create a new column and use np.select to assign values to it using our lists as arguments
            df['elective'] = np.select(conditions, values)
            
    # ICD-10-AM Code (Feature Engineering) 
    if 'icd10' in df.columns.tolist():           
        df['icd10Chapter'] = df['icd10'].astype(str).str[0]
        sub_chapte = df['icd10'].astype(str).str[1:3].astype('int')
        # replace 'neoplasms'
        df['icd10Chapter'] = np.where(df['icd10Chapter'] =='C'  , 'C/D', df['icd10Chapter'])
        df['icd10Chapter'] = np.where((df['icd10'].astype(str).str[0]=='D') & (sub_chapte<49) , 'C/D', df['icd10Chapter'])
        # replace 'Infectious'
        df['icd10Chapter'] = np.where((df['icd10'].astype(str).str[0]=='A') | (df['icd10'].astype(str).str[0]=='B') , 'AB', df['icd10Chapter'])
        # replace 'injury'
        df['icd10Chapter'] = np.where((df['icd10'].astype(str).str[0]=='S') | (df['icd10'].astype(str).str[0]=='T') , 'S/T', df['icd10Chapter'])
        # replace 'ex morbility'
        df['icd10Chapter'] = np.where((df['icd10'].astype(str).str[0]=='V') | (df['icd10'].astype(str).str[0]=='Y') , 'V-Y', df['icd10Chapter'])

        # Associated category names
        diag_dict = {'AB': 'Infectious', 'C/D': 'Neoplasms', 'D': 'Blood',
                     'E': 'Endocrine', 'F': 'Mental', 'G': 'Nervous', 'H': 'Eye/Ear',
                     'I': 'Circulatory', 'J': 'Respiratory', 'K': 'Digestive', 'L': 'Skin', 
                     'M': 'Muscular', 'N': 'Genitourinary', 'O': 'Pregnancy', 'P': 'Prenatal',
                     'Q': 'Congenital', 'R': 'Symptoms', 'S/T': 'Injury',
                     'U-V': 'Morbidity', 'Z': 'misc'}
        df['ICD10AM'] = df['icd10Chapter'].replace(diag_dict)
        
    # Age (Feature Engineering)
    bins = [0, 20, 30, 40, 50, 60, 70, 80, 90, 120]
    labels = [1,2,3,4,5,6,7,8,9] #['<20','20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89','+90']
    df['age_range'] = pd.cut(df.age, bins, labels = labels,include_lowest = True)
    
    
    # -----------------------------------------
    # 'outlier' column data engineering
    # -----------------------------------------
    # create a new column filled with ones
    #df['outlier'] = 1
    
    # create a list of our conditions
    not_outlier_conditions = [
        (df['cward'] == "IC1") | (df['cward'] == "EMU" ) | (df['cward'] == "EOU") | (df['cward'] == "DP1") | (df['cward'] == "SUB"),
        (df['cunit'] == "ACE") & ((df['cward'] == "8E" ) | (df['cward'] == "8W" )),
        (df['cunit'] == "BRE") &  (df['cward'] == "7E" ),
        (df['cunit'] == "CAR") & ((df['cward'] == "4W" ) | (df['cward'] == "CCU") | (df['cward'] == "4E")), 
        (df['cunit'] == "CTS") &  (df['cward'] == "4E" ),
        (df['cunit'] == "DER") &  (df['cward'] == "10E"),
        (df['cunit'] == "DOS") & ((df['cward'] == "7E" ) | (df['cward'] == "7W" )), 
        (df['cunit'] == "END") & ((df['cward'] == "9E" ) | (df['cward'] == "9W" )),
        (df['cunit'] == "ENT") &  (df['cward'] == "5W" ),
        (df['cunit'] == "ESO") & ((df['cward'] == "9E" ) | (df['cward'] == "9W" )),
        (df['cunit'] == "ESS") & ((df['cward'] == "EMU") | (df['cward'] == "EOU")),
        (df['cunit'] == "FMS") & ((df['cward'] == "9W" ) | (df['cward'] == "5W" )),
        (df['cunit'] == "GAS") & ((df['cward'] == "7E" ) | (df['cward'] == "7W" )),
        (df['cunit'] == "GMA") & ((df['cward'] == "8E" ) | (df['cward'] == "8W" )),
        (df['cunit'] == "GMB") & ((df['cward'] == "8E" ) | (df['cward'] == "8W" )),
        (df['cunit'] == "GMC") & ((df['cward'] == "8E" ) | (df['cward'] == "8W" )),
        (df['cunit'] == "GMD") & ((df['cward'] == "8E" ) | (df['cward'] == "8W" )),
        (df['cunit'] == "GME") & ((df['cward'] == "8E" ) | (df['cward'] == "8W" )),
        (df['cunit'] == "GS1") & ((df['cward'] == "7W" ) | (df['cward'] == "7E" )),
        (df['cunit'] == "GS3") & ((df['cward'] == "7W" ) | (df['cward'] == "7E" )),
        (df['cunit'] == "HAE") &  (df['cward'] == "6W"),
        (df['cunit'] == "IDU") & ((df['cward'] == "8E" ) | (df['cward'] == "8W" )),
        (df['cunit'] == "N/L") &  (df['cward'] == "10E"),
        (df['cunit'] == "N/S") &  (df['cward'] == "10W"),
        (df['cunit'] == "ONC") &  (df['cward'] == "6W" ),
        (df['cunit'] == "ORT") & ((df['cward'] == "9E" ) | (df['cward'] == "9W" )),
        (df['cunit'] == "PLA") &  (df['cward'] == "5W" ),
        (df['cunit'] == "PSN") &  (df['cward'] == "PSN"),
        (df['cunit'] == "REN") &  (df['cward'] == "10E"),
        (df['cunit'] == "RES") & ((df['cward'] == "4E" ) | (df['cward'] == "4W" )),
        (df['cunit'] == "RHU") & ((df['cward'] == "9E" ) | (df['cward'] == "9W" )),
        (df['cunit'] == "STR") &  (df['cward'] == "10E"),
        (df['cunit'] == "URO") &  (df['cward'] == "7W" ),
        (df['cunit'] == "VAS") &  (df['cward'] == "5W" )
        ]

    # create a list of the values we want to assign for each condition
    not_outlier_values = [0] * len(not_outlier_conditions)

    # create a new column and use np.select to assign values to it using our lists as arguments
    df['outlier'] = np.select(not_outlier_conditions, not_outlier_values, default=1)
    # fill nan with 1
    #df['outlier'] = df['outlier'].fillna(1)

    if verbose: 
        current_step +=1
        print(f'({current_step}/{steps}) Completed Feature Engineering')
    
    # -----------------------------------------
    #        Droping useless columns 
    # -----------------------------------------
    columns_to_remove = ['sex', 'pproc','pdiagd','pprocd','ecomplaint','elective2', 'diagnosis',                                        # redundante       -'pdiag' ,'nhin','statin','transin', ,'cunit'
                         #'vascular','stroke','respiratory','renal','nsurg','genmed','cts','gensurg','ort','ent','card',                # redundante in (unitcode)
                         'sa4name', 'etriage_cat','icd10Chapter','age',                                                                 # redundante
                         'cadmitmbs','icd10', 'ICD10AM',                                                                                # codded     -           
                         'drg', 'wies', 'period', 'outcome', 'arrest',                                                                  # outcome/leakage  -'died', 'survive10', 'agerisk', 'cexpstay'
                         'anyarrest', 'call','atrisk',                                                                                  # outcome/leakage  -hiout, 'met', 'cb', 'anycall', 'cicu','ctheatre',  
                         'cbed','postcode',                                                                                             # location
                         'dt','cadttext','padmdtext','rdt','cadt','padt','admitdt','adt','dischargedt','ddt', 'sequencelos',            # date/time
                         'dobt', 'dodt', 'ad','dd' ,'dob', 'dod', 'eventdate','ageadmission',                                           # date/time
                         'id', 'Unnamed: 0','sequence','pvisit','urno','visit',                                                         # id                ,  
                         'phc','bicarbc','lactatec','hbc','wccc','creatininec','plateletsc','phc2','bicarbc2','lactatec2','hbc2',       # useless
                         'wccc2','creatininec2','plateletsc2',                                                                          # useless
                         'evisit','punit','plos' ,'Lat_precise','Long_precise','pcall','frail2' ,'frail3',                              # useless                   
                        ]
    new_columns = [col for col in df.columns.tolist() if col not in columns_to_remove]
    df = df[new_columns]
    
    if verbose: 
        current_step +=1
        print(f'({current_step}/{steps}) Completed Droping useless columns')
    
    # -----------------------------------------
    #        Handling Missing Values  
    # -----------------------------------------
    #   
    fillna_list = ['survive10','frailty','pcomorbid','combinedscore','sadd','srsd','ser','seo',
                   'ecomplaintcode', 'sa4','state', 'pdiag','diagnosis', 'cadmitmbs','icd10',
                  'agerisk']
    for col in fillna_list:
        if col in df.columns.tolist():
            df[col] = df[col].fillna(-1)

    if 'hiout' in df.columns.tolist():
        df = df[df['hiout'].notna()]

    # Recategorization
    binary_cols = ['frail2', 'hiout', 'edvisit']#,'outlier']
    for col in binary_cols:
        if col in df.columns.tolist():
            if col != 'frail2':
                df[col] = df[col].map({'Yes': 1, 'No': 0})
            else:
                df[col] = df[col].map({'Frail': 1, 'Normal': 0})    
                
    # handle other missing values with most frequency columns data
    df_mode =  pd.read_csv(df_mode_path, low_memory=False)
    mode_values = {k:df_mode[k][0] for k in df_mode.columns.tolist() if k in df.columns.tolist()} 
    df.fillna(value=mode_values, inplace=True)  
    
    if verbose: 
        current_step +=1
        print(f'({current_step}/{steps}) Completed missing value handeling') 
        
    
    
    # -----------------------------------------
    #        One hot encoding
    # -----------------------------------------
    df = pd.get_dummies(df, columns=one_hot_cols)
    
    one_hot_columns_name = ['csource_E', 'csource_HME', 'csource_HWL', 'csource_N', 'csource_R',
                            'csource_S', 'csource_T', 'csource_TRC', 'csource_zPN', 'cadmitclass_EAD',
                            'cadmitclass_EMD', 'cadmitclass_O', 'cadmitclass_S', 'state_-1', 'state_ACT',
                            'state_NSW', 'state_NT', 'state_QLD', 'state_SA', 'state_TAS', 'state_VIC', 'state_WA']
    # add zero filled columns if the 'one_hot_columns_name' columns not exist in the df 
    not_in_list = np.setdiff1d(one_hot_columns_name,df.columns.values)
    for col in not_in_list:
        df[col] = 0
    
    if verbose: 
        current_step +=1
        print(f'({current_step}/{steps}) Completed data type conversion')
    
    # -----------------------------------------
    # Select numerical and categorical columns
    # -----------------------------------------
    # Select numerical and categorical columns
    numerical_ix   = df.select_dtypes(include=['int64', 'float64', 'category', 'uint8']).columns
    categorical_ix = df.select_dtypes(include=['object', 'bool']).columns
    # reorder the columns name
    reordered_cols =  list(categorical_ix)+list(numerical_ix)
    df = df[reordered_cols]
    if verbose: 
        current_step +=1
        print(f'({current_step}/{steps}) Completed data type conversion')    
        
    # -----------------------------------------
    #        data type conversion
    # -----------------------------------------
    # Convert dtype of some columns to int
    int_columns = ['etriage', 'pcomorbid','agerisk','combinedscore','sa4', 'combinedscore', 'srsd', 'ser', 'seo','sadd']
    df[int_columns] = df[int_columns].astype('int')
    '''df_dtypes={'cward': np.array([object()]).dtype, 'cunit': np.array([object()]).dtype, 'ecomplaintcode': np.array([object()]).dtype,
               'pdiag': np.array([object()]).dtype,'indig': np.int64, 'cpvthlth': np.int64,
               'cexpstay': np.int64, 'etriage': np.int64, 'cicu': np.int64,
               'ctheatre': np.int64,'pcomorbid': np.int64,'agerisk': np.int64,
               'combinedscore': np.int64,'survive10': np.float64,'picu': np.int64,
               'ph': np.float64,'bicarb': np.int64,'lactate': np.float64,
               'hb': np.int64,'wcc': np.float64,'creatinine': np.int64,
               'platelets': np.int64,'sadd': np.int64,'srsd': np.int64,
               'ser': np.int64,'seo': np.int64,'frailty': np.float64,
               'met': np.int64,'cb': np.int64,'anycall': np.int64,
               'died': np.int64, 'LOS': np.float64,'sa4': np.int64,
               'male': np.int64,'elective': np.int64,
               'outlier': np.int64,'csource_E': np.uint8,'csource_HME': np.uint8,
               'csource_HWL': np.uint8,'csource_N': np.uint8,'csource_R': np.uint8,
               'csource_S': np.uint8,'csource_T': np.uint8,'csource_TRC': np.uint8,
               'csource_zPN': np.uint8,'cadmitclass_EAD': np.uint8,'cadmitclass_EMD': np.uint8,
               'cadmitclass_O': np.uint8,'cadmitclass_S': np.uint8,'state_-1': np.uint8,
               'state_ACT': np.uint8,'state_NSW': np.uint8,'state_NT': np.uint8,
               'state_QLD': np.uint8,'state_SA': np.uint8,'state_TAS': np.uint8,
               'state_VIC': np.uint8,'state_WA': np.uint8} 
    for key in df_dtypes.keys():
        if key in df.columns.tolist():
            print(key)
            df[key] = df[key].astype(df_dtypes[key])'''
    
    if verbose: 
        current_step +=1
        print(f'({current_step}/{steps}) Completed data type conversion')
    
    if verbose: 
        print('Data Preprocessing complete.')
    
    print('Total Number of remained missing values --> {}'.format(df.isna().sum().sum()))
    print('Dimensions of the final dataset: {}\n'.format(df.shape))
    '''data_mode = pd.DataFrame(df.value_counts().idxmax()).T
    data_mode.columns = df.columns
    data_mode.to_csv('df_mode.csv', index=False)'''
    return df, categorical_ix