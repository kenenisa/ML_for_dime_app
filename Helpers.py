import requests

def get_request(url):
    try:
        response = requests.get(f'https://dime.onrender.com/ml/{url}'.format(url = url))
        # print(response.json())
        return response.json()
    except Exception as e:
        print(e)
    return None

def post_reserver(high_spender_count, low_spender_count):
    try:
        response = requests.post('https://dime.onrender.com/ml/reserve',
                                 json = {"high_spenders" : high_spender_count, "low_spenders" : low_spender_count})
    except Exception as e:
        print(e)

def post_prediction(predicions):
    try:
        response = requests.post('https://dime.onrender.com/ml/prediction', json = predicions)
    except Exception as e:
        print(e)
