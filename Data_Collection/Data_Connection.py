# Setting up connection to the database

import pandas as pd
import requests
import msal
from decouple import config

# Connection Information
CLIENT_ID = config('client_id') 
TENANT_ID = config('tenant_id')
AUTHORITY_URL = f'https://login.microsoftonline.com/{TENANT_ID}'  # Replace with your tenant ID
SCOPE =  config('scope')
DATAVERSE_URL = config('dataverse_url') 

# Initialize the MSAL app
app = msal.PublicClientApplication(client_id=CLIENT_ID, authority=AUTHORITY_URL)

# Acquire a token interactively
result = app.acquire_token_interactive(scopes=SCOPE)

# Check if the token was acquired successfully
if 'access_token' in result:
    access_token = result['access_token']
    print("Access token acquired successfully.")
else:
    print("Failed to acquire access token.")
    print(result.get("error"))
    print(result.get("error_description"))
    print(result.get("correlation_id"))


# Set API headers
headers = {
    'Authorization': f'Bearer {access_token}',
    'Accept': 'application/json',
    'OData-MaxVersion': '4.0',
    'OData-Version': '4.0'
}

def load_contacts():
    # Get list of Contacts and add to DataFrame
    contacts_url = config('contacts_url')

    #contacts = fetch_first_page(contacts_url, headers=headers)
    contacts = fetch_all_records(contacts_url, headers=headers)
    if contacts.empty:
        print("No contacts found.")
        return
    
    # Save to CSV
    contacts.to_csv('Data/contacts.csv', index=False)
    print("Contacts list saved to 'Data/contacts.csv'")
    
def load_lottery_participants():
    # Get list of Participants in lottery and add to DataFrame
    participants_url = config('participants_url')

    #participants = fetch_first_page(participants_url, headers=headers)
    participants = fetch_all_records(participants_url, headers=headers)

    if participants.empty:
        print("No participants found.")
        return
    
    # Save to CSV
    participants.to_csv('Data/lottery_participants.csv', index=False)
    print("Lottery participants list saved to 'Data/lottery_participants.csv'")


def load_memberships():
    # Get list of memberships
    memberships_url = config('memberships_url')

    #df_memberships =fetch_first_page(memberships_url, headers=headers)
    df_memberships = fetch_all_records(memberships_url, headers=headers)
    if df_memberships.empty:
        print("No memberships found.")
        return

    # Save to CSV
    df_memberships.to_csv('Data/memberships.csv', index=False)
    print("Memberships list saved to 'Data/memberships.csv'")
    

def load_membership_types():
    # Get list of membership types
    membership_types_url = config('membership_types_url')

    df_membership_types = fetch_all_records(membership_types_url, headers)
    if df_membership_types.empty:
        print("No membership types found.")
        return
    
    # Save to CSV
    df_membership_types.to_csv('Data/membership_types.csv', index=False)
    print("Membership types list saved to 'Data/membership_types.csv'")

    return df_membership_types

def load_payments():
    # Get list of payments
    payments_url = config('payments_url')

    #df_payments = fetch_first_page(payments_url, headers=headers)
    df_payments = fetch_all_records(payments_url, headers=headers)
 
    if df_payments.empty:
        print("No payments found.")
        return

    # Save to CSV
    df_payments.to_csv('Data/payments.csv', index=False)
    print("Payments list saved to 'Data/payments.csv'")

    return df_payments

def load_lottery():
    # Get list of lotteries
    lottery_url = config('lottery_url')

    df_lottery = fetch_first_page(lottery_url, headers=headers)
    #df_lottery = fetch_all_records(lottery_url, headers=headers)
    if df_lottery.empty:
        print("No lotteries found.")
        return

    # Save to CSV
    df_lottery.to_csv('Data/lottery.csv', index=False)
    print("Lottery list saved to 'Data/lottery.csv'")

    return df_lottery

def fetch_all_records(url, headers):
    all_records = []
    while url:
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            data = response.json().get('value', [])
            all_records.extend(data)
            # Check for next link
            url = response.json().get('@odata.nextLink')
        else:
            print(f"Failed to retrieve data. Status Code: {response.status_code}")
            print(f"Error Message: {response.text}")
            break
    
    return pd.DataFrame(all_records)

def fetch_first_page(url, headers):
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json().get('value', [])
        return pd.DataFrame(data)
    else:
        print(f"Failed to retrieve data. Status Code: {response.status_code}")
        print(f"Error Message: {response.text}")
        return pd.DataFrame()
    

# Uncomment the following lines to load data

#load_contacts()
#load_lottery_participants()
#load_payments()
#load_memberships()
#load_membership_types()
#load_lottery()