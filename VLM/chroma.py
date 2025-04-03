import chromadb
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch

class ChromaDB:
    def __init__(self, embedding_function=None,collection_name="collection"):
        self.client = chromadb.Client()
        self.collection = self.client.create_collection(name=collection_name)
        self.embedding_function = embedding_function
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def documents_embedding(self, documents):
        if self.embedding_function:
            embeddings = [self.embedding_function(doc) for doc in documents]
        else:
            embeddings = documents
        return embeddings
    def retrieve_documents(self, query, n_results=5):
        if self.embedding_function:
            embedding = self.embedding_function(query)
            results = self.collection.similarity_search_with_score(query=embedding, n_results=n_results)
        else:
            results = self.collection.query(query=query, n_results=n_results)
        return results
    def images_embedding(self, image_path):
        image = Image.open(image_path)
        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            embedding = self.model.get_image_features(**inputs)
        return embedding