import pandas as pd

# This file is used to clean the data from the CSV files and merge them into a single cleaned csv file that can be used for training models.

# Intial load of dataframes from CSV files
df_participants = pd.read_csv('Data/lottery_participants.csv')
df_contacts = pd.read_csv('Data/contacts.csv', dtype={'ap_ingenkontaktoverhovedet': str, 'address1_composite': str})
df_memberships = pd.read_csv('Data/memberships.csv')
df_membership_types = pd.read_csv('Data/membership_types.csv')
df_payments = pd.read_csv('Data/payments.csv')
df_lotteries = pd.read_csv('Data/lottery.csv')

# Function for clean up of participation data
def clean_participants(df_participants, df_payments, df_lotteries):
    # Removing unnecessary columns
    df_participants = df_participants.drop(columns=['@odata.etag'])

    # Renaming columns for better readability
    df_participants = df_participants.rename(columns={
        '_dbf_deltager_value': 'Kontakt_ID',
        'dbf_name': 'Navn',
        '_dbf_skrabelotteri_value': 'Skrabelotteri_ID',
        'dbf_lotterideltagerid': 'Deltager_ID'
    })

    # Filtering out rows linked to lottiries not associated with DM (Business rule - multiple ids)
    df_lotteries = df_lotteries[df_lotteries['dbf_afsendelsesmetode'] == 865500000]
    
    df_participants = df_participants[df_participants['Skrabelotteri_ID'].isin(df_lotteries['dbf_skrabelotteriid'])]

    # Adds the payment column to the dataframe (indicating if they have actively participated)
    df_payments = df_payments[df_payments['cen_amountpaid'] > 0]
    df_participants_active = df_participants.merge(df_payments, left_on='_dbf_betaling_value', right_on='cen_paymentid', how='inner')
    df_participants['Aktiv'] = df_participants['_dbf_betaling_value'].isin(df_participants_active['cen_paymentid']).astype(int)

    # Using GroupBy to aggregate the data and count the number of participations and active participations
    df_participants = df_participants.groupby('Kontakt_ID').agg({
        'Aktiv': 'sum',
        'Deltager_ID': 'count'
    }).reset_index()

    df_participants = df_participants.rename(columns={
        'Aktiv': 'Antal_Aktiv',
        'Deltager_ID': 'Antal'
    })

    # Sorting based on the number of active participations with highest first
    df_participants = df_participants.sort_values(by='Antal_Aktiv', ascending=False)
    df_participants['Aktiv_Deltager'] = (df_participants['Antal_Aktiv'] > 0).astype(int)
    df_participants['Aktiv_Deltager'] = df_participants['Aktiv_Deltager'].replace({1: 'Ja', 0: 'Nej'})

    df_participants.to_csv('Data/cleaned_participants.csv', index=False)

    return df_participants

# Function for cleaning the memberships data
def clean_memberships(df_memberships, df_membership_types):
    # Removing unnecessary columns
    df_memberships = df_memberships.drop(columns=['@odata.etag'])

    # Renaming columns for better readability
    df_memberships = df_memberships.rename(columns={
        '_ap_kontaktid_value': 'Kontakt_ID',
        'dbf_startdato': 'Startdato',
        '_ap_medlemstypeid_value': 'Medlemstype_ID',
        'statecode': 'Medlem_Status'
    })

    df_membership_types = df_membership_types.rename(columns={
        'ap_navn': 'Medlemstype'
    })

    # Calculating the duration of the membership
    df_memberships['Startdato'] = pd.to_datetime(df_memberships['Startdato'], errors='coerce').dt.year
    df_memberships['Medlem_Tid'] = pd.to_datetime('today').year - df_memberships['Startdato']
    

    # Merging with membership types to get the name of the membership type
    df_memberships = df_memberships.merge(df_membership_types, left_on='Medlemstype_ID', right_on='ap_medlemstypeid', how='left')

    # Removing unnecessary columns
    df_memberships = df_memberships.drop(columns=['ap_medlemstypeid', '@odata.etag', 'Medlemstype_ID', 'ap_medlemskabid'])

    # Removing rows with missing values in Kontakt_ID or duplicate Kontakt_IDs
    df_memberships = df_memberships.dropna(subset=['Kontakt_ID'])
    df_memberships = df_memberships.drop_duplicates(subset=['Kontakt_ID'])

    return df_memberships

# Function for cleaning the contacts data - pre-requisite that other dataframes are cleaned first
def clean_contacts(df_contacts, df_memberships, df_participants):
    # Removing unnecessary columns
    df_contacts = df_contacts.drop(columns=['@odata.etag', '_ap_medlemskabid_value', 'address1_composite'])

    # Renaming columns for better readability
    df_contacts = df_contacts.rename(columns={
        'contactid': 'Kontakt_ID',
        'ap_medlemskabsnummer': 'Medlemskabsnummer',
        'ap_alder': 'Alder',
        'statuscode': 'Status_Aarsag',
        'statecode': 'Status',
        'ap_medlem': 'Medlem',
        'ap_ingenkontaktoverhovedet': 'Kontakt_OK'
    })

    # Merging previously cleaned dataframes with contacts dataframe
    df_contacts = df_contacts.merge(df_memberships, on='Kontakt_ID', how='left')
    df_contacts = df_contacts.merge(df_participants, on='Kontakt_ID', how='left')

    # Mapping the postal codes to regions
    df_contacts['address1_postalcode'] = pd.to_numeric(df_contacts['address1_postalcode'], errors='coerce')
    df_contacts['Region'] = pd.cut(df_contacts['address1_postalcode'], bins=[0, 4999, 6000, 8999], labels=['Sjælland', 'Fyn', 'Jylland'], right=False)
    df_contacts['Region'] = df_contacts['Region'].astype(str)

    # Handling missing values in the dataset
    df_contacts['Antal_Aktiv'] = df_contacts['Antal_Aktiv'].fillna(0)
    df_contacts['Antal_Aktiv'] = df_contacts['Antal_Aktiv'].astype(int)

    df_contacts['Antal'] = df_contacts['Antal'].fillna(0)
    df_contacts['Antal'] = df_contacts['Antal'].astype(int)

    df_contacts['Aktiv_Deltager'] = df_contacts['Aktiv_Deltager'].fillna('Nej')

    df_contacts['Medlemstype'] = df_contacts['Medlemstype'].fillna('N/A')
    df_contacts['Medlem_Tid'] = df_contacts['Medlem_Tid'].fillna(0).astype(int)
    df_contacts['Medlem_Status'] = df_contacts['Medlem_Status'].fillna('N/A')
    df_contacts['Startdato'] = df_contacts['Startdato'].fillna('N/A')

    df_contacts.to_csv('Data/cleaned_contacts_for_filtering.csv', index=False)

    # Applying custom order to columns
    df_contacts = df_contacts[['Kontakt_ID', 'Alder', 'Region', 'Status_Aarsag', 'Status', 'Kontakt_OK', 'Medlem', 'Medlemskabsnummer', 'Medlemstype', 'Medlem_Tid', 'Medlem_Status' ,'Startdato','Antal', 'Antal_Aktiv', 'Aktiv_Deltager']]

    return df_contacts

# Function for final preprocessing - application of business rules and filters
def business_rules(df_contacts):
    # Applying business generic rules and filters
    df_contacts = df_contacts[df_contacts['Status_Aarsag'] != 778210004]
    df_contacts = df_contacts[df_contacts['Medlemskabsnummer'].notnull()]
    df_contacts = df_contacts[df_contacts['Kontakt_OK'] != 0]
    df_contacts = df_contacts[df_contacts['Alder'] > 18]
    df_contacts = df_contacts[df_contacts['Alder'] < 120]

    df_contacts = df_contacts.drop(columns=['Medlemskabsnummer'])

    df_contacts = df_contacts.sort_values(by='Antal', ascending=False)
    # Saving final cleaned data to CSV
    df_contacts.to_csv('Data/cleaned_data.csv', index=False)


# Execution order of code
cleaned_parcipants = clean_participants(df_participants, df_payments, df_lotteries)
cleaned_memberships = clean_memberships(df_memberships, df_membership_types)
cleaned_contacts = clean_contacts(df_contacts, cleaned_memberships, cleaned_parcipants)
business_rules(cleaned_contacts)