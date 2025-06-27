import openai
import requests
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Prepare headers
headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}

# Set date range (start of month to today)
start_date = datetime.today().replace(day=1).strftime('%Y-%m-%d')
end_date = datetime.today().strftime('%Y-%m-%d')

# API call to fetch usage
url = f"https://api.openai.com/v1/dashboard/billing/usage?start_date={start_date}&end_date={end_date}"
response = requests.get(url, headers=headers)

# Print usage
print("Usage this month (in $):", response.json().get("total_usage", 0) / 100)




# Check available credit
credit_url = "https://api.openai.com/v1/dashboard/billing/credit_grants"
credit_response = requests.get(credit_url, headers=headers)
credits = credit_response.json()

print("Total granted: $", credits.get("total_granted", 0))
print("Total used: $", credits.get("total_used", 0))
print("Total available: $", credits.get("total_available", 0))
