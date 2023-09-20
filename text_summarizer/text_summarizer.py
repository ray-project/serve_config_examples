from fastapi import FastAPI
from ray import serve
import torch

# 1: Define a FastAPI app and wrap it in a deployment with a route handler.
app = FastAPI()

@serve.deployment(
    route_prefix="/",
    ray_actor_options={"num_gpus": 1},
)
@serve.ingress(app)
class SummaryDeployment:
    # FastAPI will automatically parse the HTTP request for us.
    def __init__(self):
        from transformers import BartForConditionalGeneration, BartTokenizer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model_name = "facebook/bart-large-cnn"
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.model = BartForConditionalGeneration.from_pretrained(model_name).to(
            self.device
        )

    # Reference: https://github.com/amaiya/ktrain/blob/master/ktrain/text/summarization/core.py
    @app.get("/summarize")
    def summarize(self, text: str) -> str:
        max_length = 50
        min_length = 10
        no_repeat_ngram_size = 3
        length_penalty = 2.0
        num_beams = 4

        with torch.no_grad():
            answers_input_ids = self.tokenizer.batch_encode_plus(
                [text], return_tensors="pt", truncation=True, max_length=max_length, min_length=min_length
            )["input_ids"].to(self.device)
            summary_ids = self.model.generate(
                answers_input_ids,
                num_beams=num_beams,
                length_penalty=length_penalty,
                max_length=max_length,
                min_length=min_length,
                no_repeat_ngram_size=no_repeat_ngram_size,
            )

            exec_sum = self.tokenizer.decode(
                summary_ids.squeeze(), skip_special_tokens=True
            )
        return exec_sum

deployment = SummaryDeployment.bind()
