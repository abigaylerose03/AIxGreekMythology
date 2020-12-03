import requests 

url = "http://localhost:5000/results"
r = requests.post(url, json={'category': 10000, 'status': 1, 'gender': 100, 'defense': 0.23353, 
	'attack': 0.435232})

print(r.json())