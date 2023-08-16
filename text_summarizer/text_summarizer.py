from fastapi import FastAPI
from ray import serve

# 1: Define a FastAPI app and wrap it in a deployment with a route handler.
app = FastAPI()


@serve.deployment(route_prefix="/")
@serve.ingress(app)
class SummaryDeployment:
    # FastAPI will automatically parse the HTTP request for us.
    def __init__(self):
        from transformers import pipeline
        self.classifier = pipeline("summarization", model="facebook/bart-large-cnn")

    @app.get("/summarize")
    def summarize(self, text: str) -> str:
        result: list = self.classifier(text, min_length=10, max_length=50)
        print(result)
        return result[0]["summary_text"]

deployment = SummaryDeployment.bind()
