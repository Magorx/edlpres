from jina import Deployment
from model import ObjectDetector

deployment = Deployment(uses=ObjectDetector, timeout_ready=-1, port=12345)

with deployment:
    deployment.block()
