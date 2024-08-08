from ast import keyword
#ule based text classification

def classify_request(text):
    text = text.lower()
    if any(keyword in text for keyword in ["billing","invoice","charge"]):
        return "Billing issue"
    elif any(keyword in text for keyword in["password","access","login","account"]):
        return "Technical support"
    elif any(keyword in text for keyword in["hours","time","location","general"]):
        return "general support"
    else:
        return "other support"

#test sample
requests=[
    "my account is blocked",
    "i need my last billing details",
    "i need to know the timing of my order"
]

for request in requests:
    category = classify_request(request)
    print(f"request: {request}\n category: {category}\n")