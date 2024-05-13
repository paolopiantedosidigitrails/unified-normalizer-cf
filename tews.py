import requests
import os
import google.oauth2.id_token
import google.auth.transport.requests
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = './dt-normalizer-ae036c2f452f.json'
request = google.auth.transport.requests.Request()
audience = 'https://europe-west8-dt-normalizer.cloudfunctions.net/unified_normalizer'
bearer_token = google.oauth2.id_token.fetch_id_token(request, audience)



# Define the headers
headers = {
    'Authorization': f'Bearer {bearer_token}',
    'Content-Type': 'application/json',
}

# Define the data
data = {
    "requestId": "124ab1c",
    "caller": "//bigquery.googleapis.com/projects/myproject/jobs/myproject:US.bquxjob_5b4c112c_17961fafeaf",
    "sessionUser": "test-user@test-company.com",
    "userDefinedContext": {
     "norm_type": "skill"
    },
    "calls": [
     ["Fireproofing"]
    ]
}

# Make the POST request
response = requests.post('https://europe-west8-dt-normalizer.cloudfunctions.net/unified_normalizer', headers=headers, json=data, timeout=3610)

# Print the response
print(response.json())