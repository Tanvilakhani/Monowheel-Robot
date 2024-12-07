import requests

# Define the URL and parameters
base_url = "http://10.136.45.13:5001/data"
params = {"dist": "val"}  # Query parameter

try:
    # Send GET request
    response = requests.get(base_url, params=params)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the response data (assuming it's JSON)
        data = response.json()
        print("Data received:", data)
    else:
        print(f"Request failed with status code: {response.status_code}")
except requests.RequestException as e:
    print(f"An error occurred: {e}")
