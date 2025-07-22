import requests
import json

url = "http://localhost:8080/create_avatar"

payload = json.dumps({
  "json_data": """{
    "avatar_img_url": "string",
    "shape": {
      "additionalProp1": {
        "emoji": "😎",
        "description": "Крутой человек в очках"
      }
    },
    "name": "Nikitosik",
    "sex": "Мужчина",
    "additionalDetails": "Ты очень общительный",
    "interests": [
      "Футбол",
      "Волейбол"
    ],
    "abilities": [
      "Умный"
    ],
    "places": [
      "Парк"
    ]
  }
}"""})

headers = {
  'accept': 'application/json',
  'Content-Type': 'application/json'
}

response = requests.request("POST", url, headers=headers, data=payload)

print(response.text)
