import requests

url = "https://www.google.com"  # Replace with your desired website

try:
  response = requests.get(url, timeout=5)  # Set a timeout to avoid waiting forever

  if response.status_code == 200:
    print("Connected to the internet!")
    print(f"Response content: {response.text[:100]}")  # Print the first 100 characters
  else:
    print(f"Error connecting to {url}. Status code: {response.status_code}")

except requests.exceptions.RequestException as e:
  print(f"Connection error: {e}")
