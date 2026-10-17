from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

# 测试1：正常输入
def test_normal():
    res = client.post("/predict", json={"input_text": "随便写一句话"})
    assert res.status_code == 200

# 测试2：空输入校验
def test_empty():
    res = client.post("/predict", json={"input_text": ""})
    assert res.status_code == 422

# 测试3：特殊字符
def test_special_char():
    res = client.post("/predict", json={"input_text": "!@#$%^&*()"})
    assert res.status_code == 200
