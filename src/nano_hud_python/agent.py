from pydantic import BaseModel

class Agent(BaseModel):
    """Litellm agent that wraps a Docker environment."""