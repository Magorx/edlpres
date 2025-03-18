from jina import Client
from docarray import DocList
from model import ImageInput, DetectedObjects

image_input = ImageInput(url='http://images.cocodataset.org/val2017/000000001268.jpg')
client = Client(port=12345)
response = client.post('/', inputs=image_input, return_type=DetectedObjects)
print(response[0])
