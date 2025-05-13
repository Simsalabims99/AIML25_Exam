# This is the processing of the baselines filters provided by Diabetesforeningen.
# The filters will be applied to the same dataset as the ML & AI models.

import pandas as pd

dataframe = pd.read_csv('Data/cleaned_contacts_for_filtering.csv')

def filter_1(dataframe):
    dataframe = dataframe[dataframe['Status_Aarsag'] != 778210004]
    dataframe = dataframe[dataframe['Medlemskabsnummer'].notnull()]
    dataframe = dataframe[dataframe['Kontakt_OK'] != 0]
    dataframe = dataframe[dataframe['Alder'] > 18]
    dataframe = dataframe[dataframe['Alder'] < 120]
    dataframe = dataframe[dataframe['Medlem'] == 0]
    dataframe = dataframe[dataframe['Status'] == 0]
    dataframe = dataframe[dataframe['Medlem_Status'] == 0]
    dataframe = dataframe[dataframe['Antal_Aktiv'] > 1]

    return dataframe

def filter_2(dataframe):
    dataframe = dataframe[dataframe['Status_Aarsag'] != 778210004]
    dataframe = dataframe[dataframe['Medlemskabsnummer'].notnull()]    
    dataframe = dataframe[dataframe['Kontakt_OK'] != 0]
    dataframe = dataframe[dataframe['Medlem'] == 1]
    dataframe = dataframe[dataframe['Alder'] > 18]
    dataframe = dataframe[dataframe['Alder'] < 120]
    dataframe = dataframe[dataframe['ap_modtagermedlemsblad'] != 778210003]
    dataframe = dataframe[dataframe['Medlem_Status'] == 0]
    dataframe = dataframe[dataframe['Antal_Aktiv'] > 1]

    return dataframe

def filter_3(dataframe):
    dataframe = dataframe[dataframe['Status_Aarsag'] != 778210004]
    dataframe = dataframe[dataframe['Medlemskabsnummer'].notnull()]
    dataframe = dataframe[dataframe['Kontakt_OK'] != 0]
    dataframe = dataframe[dataframe['Medlem'] == 1]
    dataframe = dataframe[dataframe['Alder'] > 18]
    dataframe = dataframe[dataframe['Alder'] < 120]
    dataframe = dataframe[dataframe['ap_modtagermedlemsblad'] == 778210003]
    dataframe = dataframe[dataframe['Status'] == 0]
    dataframe = dataframe[dataframe['Medlem_Status'] == 0]
    dataframe = dataframe[dataframe['Antal_Aktiv'] > 1]
    
    return dataframe

def filter_4(dataframe):
    dataframe = dataframe[dataframe['Status_Aarsag'] != 778210004]
    dataframe = dataframe[dataframe['Medlemskabsnummer'].notnull()]
    dataframe = dataframe[dataframe['Kontakt_OK'] != 0]
    dataframe = dataframe[dataframe['Alder'] > 18]
    dataframe = dataframe[dataframe['Alder'] < 120]
    dataframe = dataframe[dataframe['Medlem'] == 1]
    dataframe = dataframe[dataframe['ap_modtagermedlemsblad'] != 778210003]
    dataframe = dataframe[dataframe['dbf_segment'] == "DM 2024 Potentiel"]
    dataframe = dataframe[dataframe['dbf_dmlotteripausetprdato'].isnull()]


    return dataframe

def filter_5(dataframe):
    dataframe['dbf_indmeldelsesdatomedlemskab'] = pd.to_datetime(dataframe['dbf_indmeldelsesdatomedlemskab'], errors='coerce')

    dataframe = dataframe[dataframe['Status_Aarsag'] != 778210004]
    dataframe = dataframe[dataframe['Medlemskabsnummer'].notnull()]
    dataframe = dataframe[dataframe['Kontakt_OK'] != 0]
    dataframe = dataframe[dataframe['Alder'] > 18]
    dataframe = dataframe[dataframe['Alder'] < 120]
    dataframe = dataframe[dataframe['Medlem'] == 1]
    dataframe = dataframe[dataframe['dbf_dmlotteripausetprdato'].isnull()]
    dataframe = dataframe[dataframe['dbf_indmeldelsesdatomedlemskab'] >= '2021-01-01']
    dataframe = dataframe[dataframe['dbf_indmeldelsesdatomedlemskab'] <= '2024-10-16']
    dataframe = dataframe[dataframe['Antal'] == 0]
  

    return dataframe

def filter_6(dataframe):
    dataframe = dataframe[dataframe['Status_Aarsag'] != 778210004]
    dataframe = dataframe[dataframe['Medlemskabsnummer'].notnull()]
    dataframe = dataframe[dataframe['Kontakt_OK'] != 0]
    dataframe = dataframe[dataframe['Alder'] > 18]
    dataframe = dataframe[dataframe['Alder'] < 120]
    dataframe = dataframe[dataframe['Medlem'] == 1]
    dataframe = dataframe[dataframe['Medlem_Status'] == 0]
    dataframe = dataframe[dataframe['Antal'] > 1]
    dataframe = dataframe[dataframe['Antal_Aktiv'] < 1]

    return dataframe

def filter_7(dataframe):
    dataframe['dbf_indmeldelsesdatomedlemskab'] = pd.to_datetime(dataframe['dbf_indmeldelsesdatomedlemskab'], errors='coerce')

    dataframe = dataframe[dataframe['Status_Aarsag'] != 778210004]
    dataframe = dataframe[dataframe['Medlemskabsnummer'].notnull()]
    dataframe = dataframe[dataframe['Kontakt_OK'] != 0]
    dataframe = dataframe[dataframe['Alder'] > 18]
    dataframe = dataframe[dataframe['Alder'] < 120]
    dataframe = dataframe[dataframe['Medlem'] == 0]
    dataframe = dataframe[dataframe['Status'] == 0]
    dataframe = dataframe[dataframe['dbf_indmeldelsesdatomedlemskab'] >= '2024-10-17']
    dataframe = dataframe[dataframe['dbf_indmeldelsesdatomedlemskab'] <= '2025-02-15']
    dataframe = dataframe[dataframe['Medlem_Status'] == 0]

    return dataframe

def combine_filters(dataframe):
    # Applying all filters to the dataframe
    dataframe_1 = filter_1(dataframe)
    dataframe_2 = filter_2(dataframe)
    dataframe_3 = filter_3(dataframe)
    dataframe_4 = filter_4(dataframe)
    dataframe_5 = filter_5(dataframe)
    dataframe_6 = filter_6(dataframe)
    dataframe_7 = filter_7(dataframe)

    # Concatenating all filtered dataframes
    combined_dataframe = pd.concat([dataframe_1, dataframe_2, dataframe_3, dataframe_4, dataframe_5, dataframe_6, dataframe_7], ignore_index=True)

    # Removing duplicates
    combined_dataframe = combined_dataframe.drop_duplicates(subset=['Kontakt_ID'])

    # Removing rows with missing values in 'Kontakt_ID'
    combined_dataframe = combined_dataframe.dropna(subset=['Kontakt_ID'])

    combined_dataframe = combined_dataframe[['Kontakt_ID']]

    combined_dataframe.to_csv('Results/filtered_results.csv', index=False)
    print("Filtered results saved to 'Results/filtered_results.csv'")

# ChatGPT generated code using model chatgpt-4o at htttps://chat.openai.com/
def compare_results():
    # Comparing the filtered results with the results from the ML & AI models
    filtered_results = pd.read_csv('Results/filtered_results.csv')
    model_results = pd.read_csv('Results/predictions.csv')

    filtered_results['Kontakt_ID'] = filtered_results['Kontakt_ID'].astype(str)
    model_results['Kontakt_ID'] = model_results['Kontakt_ID'].astype(str)
    model_results_positive = model_results[model_results['Resultat'] == 1]

    merged_results = pd.merge(filtered_results, model_results_positive, on='Kontakt_ID', how='outer', indicator=True)

    # Classify comparison results
    merged_results['Comparison'] = merged_results['_merge'].map({
        'both': 'Agree',
        'left_only': 'Company',
        'right_only': 'Model'
    })

    # Summary overview of comparison
    summary = merged_results['Comparison'].value_counts()
    print("Summary:")
    print(summary)

    # 🔍 Investigate mismatches
    model_only = merged_results[merged_results['Comparison'] == 'Model']
    company_only = merged_results[merged_results['Comparison'] == 'Company']

    # Examine company-only results with model confidence scores
    company_only_scores = model_results[model_results['Kontakt_ID'].isin(company_only['Kontakt_ID'])]
    low_confidence = company_only_scores[company_only_scores['Sandsynlighed'] < 0.2]
    print("Company only low confidence results:")
    print(low_confidence['Kontakt_ID'].count())
    

    # Examine model-only results
    model_only = model_only[model_only['Sandsynlighed'] > 0.7]
    print("Model only high confidence results:")
    print(model_only['Kontakt_ID'].count())
    

combine_filters(dataframe)
compare_results()