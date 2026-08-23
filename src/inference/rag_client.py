"""
RAG Client for PolyMentor knowledge base retrieval.

Provides local, cost-free vector embedding generation via sentence-transformers
and Pinecone vector database querying/upserting capabilities.
"""

import os
import logging
from typing import List, Dict, Any, Union, Optional
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)

# Default model configurations
DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
DEFAULT_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "polymentor-knowledge")
DEFAULT_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
DEFAULT_REGION = os.getenv("PINECONE_REGION", "us-east-1")


class RAGClient:
    """
    Internal RAG API Client wrapper.
    Uses local sentence-transformers for embedding text at zero cost and
    interacts with Pinecone vector database for retrieval and storage.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        index_name: Optional[str] = None,
        model_name: Optional[str] = None,
        cloud: Optional[str] = None,
        region: Optional[str] = None,
        create_index_if_missing: bool = True,
    ):
        """
        Initialize the RAG client.
        
        Args:
            api_key: Pinecone API Key. Defaults to PINECONE_API_KEY environment variable.
            index_name: Pinecone index name. Defaults to PINECONE_INDEX_NAME env var.
            model_name: Local sentence-transformer model name.
            cloud: Pinecone serverless cloud provider (e.g. 'aws').
            region: Pinecone serverless region (e.g. 'us-east-1').
            create_index_if_missing: If True, auto-creates index in Pinecone if absent.
        """
        self.api_key = api_key or os.getenv("PINECONE_API_KEY")
        self.index_name = index_name or DEFAULT_INDEX_NAME
        self.model_name = model_name or DEFAULT_EMBEDDING_MODEL
        self.cloud = cloud or DEFAULT_CLOUD
        self.region = region or DEFAULT_REGION
        self.create_index_if_missing = create_index_if_missing

        self._encoder = None
        self._pinecone_client = None
        self._index = None
        self.is_mock = False

        self._init_encoder()
        self._init_pinecone()

    def _init_encoder(self):
        """Initialize the local sentence-transformer model."""
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading local SentenceTransformer model: {self.model_name}")
            self._encoder = SentenceTransformer(self.model_name)
            if hasattr(self._encoder, "get_embedding_dimension"):
                self.embedding_dimension = self._encoder.get_embedding_dimension()
            else:
                self.embedding_dimension = self._encoder.get_sentence_embedding_dimension()
            logger.info(f"Encoder initialized successfully. Dimension: {self.embedding_dimension}")
        except Exception as e:
            logger.error(f"Failed to load SentenceTransformer model ({self.model_name}): {e}")
            self._encoder = None
            self.embedding_dimension = 384  # Default fallback dimension for MiniLM-L6-v2

    def _init_pinecone(self):
        """Initialize Pinecone client connection."""
        if not self.api_key or self.api_key.startswith("your_"):
            logger.warning("PINECONE_API_KEY is not configured or uses placeholder value. Operating in mock mode.")
            self.is_mock = True
            return

        try:
            import pinecone

            # Modern Pinecone SDK v3+
            if hasattr(pinecone, "Pinecone"):
                self._pinecone_client = pinecone.Pinecone(api_key=self.api_key)
                existing_indexes = [idx.name for idx in self._pinecone_client.list_indexes()]

                if self.index_name not in existing_indexes and self.create_index_if_missing:
                    logger.info(f"Creating serverless Pinecone index '{self.index_name}'...")
                    from pinecone import ServerlessSpec
                    self._pinecone_client.create_index(
                        name=self.index_name,
                        dimension=self.embedding_dimension,
                        metric="cosine",
                        spec=ServerlessSpec(cloud=self.cloud, region=self.region)
                    )
                    logger.info(f"Index '{self.index_name}' created successfully.")

                self._index = self._pinecone_client.Index(self.index_name)
                logger.info(f"Connected to Pinecone index: '{self.index_name}'")

            # Legacy Pinecone SDK v2
            elif hasattr(pinecone, "init"):
                pinecone.init(api_key=self.api_key, environment=self.region)
                if self.index_name not in pinecone.list_indexes() and self.create_index_if_missing:
                    pinecone.create_index(
                        name=self.index_name,
                        dimension=self.embedding_dimension,
                        metric="cosine"
                    )
                self._index = pinecone.Index(self.index_name)
                logger.info(f"Connected to legacy Pinecone index: '{self.index_name}'")
            else:
                raise ImportError("Unrecognized pinecone SDK structure.")

        except Exception as e:
            logger.error(f"Error initializing Pinecone client: {e}. Falling back to mock mode.")
            self.is_mock = True

    def embed_text(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """
        Embed single text string or list of text strings into vector representation(s).
        Uses local sentence-transformers model to save API costs.

        Args:
            text: A single string or a list of strings to encode.

        Returns:
            List of floats (for single string) or List of List of floats (for list of strings).
        """
        if self._encoder is None:
            # Fallback zero vector if encoder failed to initialize
            if isinstance(text, str):
                return [0.0] * self.embedding_dimension
            return [[0.0] * self.embedding_dimension for _ in text]

        is_single = isinstance(text, str)
        inputs = [text] if is_single else text

        embeddings = self._encoder.encode(inputs, convert_to_numpy=True, show_progress_bar=False)
        vector_list = embeddings.tolist()

        return vector_list[0] if is_single else vector_list

    def query_knowledge_base(
        self,
        query: str,
        top_k: int = 5,
        namespace: str = "docs",
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Query the Pinecone knowledge base using dense embeddings from the local model.

        Args:
            query: User search query string.
            top_k: Number of nearest neighbor matches to return.
            namespace: Namespace in Pinecone index.
            filter_dict: Metadata filtering dictionary.

        Returns:
            List of match dictionaries containing 'id', 'score', and 'metadata'.
        """
        query_vector = self.embed_text(query)

        if self.is_mock or self._index is None:
            logger.warning("Query executed in mock mode or index unavailable. Returning mock results.")
            return [
                {
                    "id": f"mock-{i}",
                    "score": 0.95 - (i * 0.05),
                    "metadata": {
                        "text": f"Mock retrieved text answer for query: '{query}' [Match #{i+1}]",
                        "source_path": "docs/COMPLETION_VISION.md",
                        "header": "Overview",
                    },
                }
                for i in range(min(top_k, 3))
            ]

        try:
            kwargs = {
                "vector": query_vector,
                "top_k": top_k,
                "include_metadata": True,
            }
            if namespace:
                kwargs["namespace"] = namespace
            if filter_dict:
                kwargs["filter"] = filter_dict

            results = self._index.query(**kwargs)
            matches = []
            for match in results.get("matches", []):
                matches.append({
                    "id": match.get("id"),
                    "score": float(match.get("score", 0.0)),
                    "metadata": match.get("metadata", {}),
                })
            return matches

        except Exception as e:
            logger.error(f"Error querying Pinecone knowledge base: {e}")
            return []

    def upsert_documents(
        self,
        vectors: List[Dict[str, Any]],
        namespace: str = "docs",
        batch_size: int = 100,
    ) -> Dict[str, Any]:
        """
        Upsert vector records into Pinecone.

        Args:
            vectors: List of items formatted as {"id": str, "values": list[float], "metadata": dict}
            namespace: Pinecone namespace.
            batch_size: Number of vectors per upsert payload.

        Returns:
            Dictionary with execution status and total upserted count.
        """
        if self.is_mock or self._index is None:
            logger.info(f"[Mock Mode] Simulating upsert of {len(vectors)} vectors to namespace '{namespace}'.")
            return {"status": "success", "upserted_count": len(vectors), "mock": True}

        total_upserted = 0
        try:
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i : i + batch_size]
                self._index.upsert(vectors=batch, namespace=namespace)
                total_upserted += len(batch)

            logger.info(f"Successfully upserted {total_upserted} vectors to Pinecone.")
            return {"status": "success", "upserted_count": total_upserted, "mock": False}
        except Exception as e:
            logger.error(f"Error upserting vectors to Pinecone: {e}")
            return {"status": "error", "message": str(e), "upserted_count": total_upserted}

    def get_index_stats(self) -> Dict[str, Any]:
        """Retrieve vector index statistics."""
        if self.is_mock or self._index is None:
            return {"total_vector_count": 0, "mock": True}
        try:
            return self._index.describe_index_stats()
        except Exception as e:
            logger.error(f"Error fetching Pinecone index stats: {e}")
            return {"error": str(e)}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    client = RAGClient()
    print("\n--- RAG Client Quick Verification ---")
    test_vec = client.embed_text("How does PolyMentor analyze code?")
    print(f"Embedding generated! Vector size: {len(test_vec)}")
    query_res = client.query_knowledge_base("PolyMentor code analyzer")
    print(f"Query returned {len(query_res)} results. First result: {query_res[0] if query_res else 'None'}")
