import logging

import modal
from config import EmbedderConfig as Config

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Create Modal stub and image
volume = modal.Volume.from_name(
    "model-final-cache-stella-1.5B-2", create_if_missing=True
)
image = modal.Image.debian_slim().pip_install(
    "sentence_transformers",
    "xformers",
    "python-dotenv",
)
app = modal.App("sentence-embedder-1.5B-2", image=image)

MODEL_DIR = "/root/models"


@app.function(
    volumes={MODEL_DIR: volume},  # "mount" the Volume, sharing it with your function
    gpu="l4",  # GPU type
    timeout=1500,
)
def save_model():
    from sentence_transformers import SentenceTransformer

    logger.info("Saving model")
    model = SentenceTransformer(
        "dunzhang/stella_en_1.5B_v5", device="cuda", trust_remote_code=True
    )
    model.save(MODEL_DIR)


@app.cls(gpu="l4", volumes={MODEL_DIR: volume})
class Model:
    @modal.enter()
    def setup(self):
        logger.info("Loading model")

        import os

        from sentence_transformers import SentenceTransformer

        logger.info(os.listdir(MODEL_DIR))
        self.model = SentenceTransformer(
            MODEL_DIR, device="cuda", trust_remote_code=True
        )

    @modal.method()
    async def inference(self, prompt):
        return self.model.encode(prompt)


@app.local_entrypoint()
async def main():

    save_model.remote()  # Comment this line if the model has already been saved on the remote device
    from fastapi import FastAPI
    from pydantic import BaseModel

    config = Config()
    app_api = FastAPI()
    myModal = Model()

    logger.info(myModal.inference.remote("Hello, world!"))

    class Query(BaseModel):
        text: str

    @app_api.get("/health")
    async def health_check():
        return {"status": "healthy"}

    @app_api.post("/embeddings")
    async def get_embedding(query: Query):
        embedding = await myModal.inference.remote.aio(query.text)
        return {"embedding": embedding.tolist()}

    import uvicorn

    config = uvicorn.Config(app_api, host="0.0.0.0", port=config.PORT)
    server = uvicorn.Server(config)
    await server.serve()
