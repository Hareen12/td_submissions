import base64
api_key = ""
encoded_api_key = base64.b64encode(api_key.encode("utf-8")).decode("utf-8")
print("Encoded API Key:", encoded_api_key)
# Decode the API key
decoded_api_key = base64.b64decode(encoded_api_key).decode("utf-8")
print("Decoded API Key:", decoded_api_key)

# Use `decoded_api_key` in your application
