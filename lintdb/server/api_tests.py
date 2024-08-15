import requests

def test_search():
    tensor = [0.1] * 128 * 32
    request = {
        "query": {
            "type": "TENSOR",
            "name": "colbert",
            "value": tensor,
            "num_tensors": 1
        },
        "options": {
            "colbert_field": "colbert"
        },
        "k": 10
    }

    resp = requests.post("http://0.0.0.0:8080/v1/Index/search/0", json=request)
    assert resp.status_code == 200
    data = resp.json()

    assert('results' in data), "Results not found in response"

    print("search test passed")

def test_add():
    tensor = [0.1] * 128 * 32
    request = {
        "documents": [
            {
                "id": 50001,
                "fields": [
                    {
                        "name": "colbert",
                        "data_type": "TENSOR",
                        "value": tensor,
                    }
                ]
            }
        ]
    }

    resp = requests.post("http://0.0.0.0:8080/v1/Index/add/0", json=request)
    assert resp.status_code == 200
    data = resp.json()

    assert('ok' in data)

    print("add test passed")

def test_update():
    tensor = [0.2] * 128 * 32
    request = {
        "documents": [
            {
                "id": 50001,
                "fields": [
                    {
                        "name": "colbert",
                        "data_type": "TENSOR",
                        "value": tensor,
                    }
                ]
            }
        ]
    }

    resp = requests.post("http://0.0.0.0:8080/v1/Index/update/0", json=request)
    assert resp.status_code == 200
    data = resp.json()

    assert('ok' in data)

    print("update test passed")

def test_remove():
    request = {
        'ids': [50001]
    }
    resp = requests.post("http://0.0.0.0:8080/v1/Index/remove/0", json=request)
    assert resp.status_code == 200
    data = resp.json()

    assert('ok' in data)

    print("remove test passed")


if __name__ == "__main__":
    test_search()
    test_add()
    test_update()
    test_remove()