"""
This file contains a simple Serve deployment that increments a counter
when it receives a request.
"""

from ray import serve
from starlette.requests import Request


@serve.deployment
class SimpleDeployment:
    """SimpleDeployment directly returns the response."""
    
    def __init__(self):
        self.counter = 0
    
    async def __call__(self, request: Request):
        """Asynchronously process requests and return immediately."""
        self.counter += 1

        return {
            "status": "ok",
            "counter": self.counter,
            "message": "processed"
        }


app = SimpleDeployment.bind()
