import requests

url = 'http://127.0.0.1:5000/predict_api'
r = requests.post(url,json={'Gender':0, 'Age':20, 'Estimated Salary':40000})

print(r.json())