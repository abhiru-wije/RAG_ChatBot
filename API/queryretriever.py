
class QueryRetriever:
    def __init__(self, model, bot_emb):
        self.model = model
        self.bot_emb = bot_emb
    
    def query(self, query_text, n_results=1):

        # Encode the query text into embeddings
        input_em = self.model.encode(query_text).tolist()
        
        # Query the vector database with the embeddings
        results = self.bot_emb.query(
            query_embeddings=[input_em],
            n_results=n_results
        )
        
        return results
    @classmethod
    def as_retriever(cls, search_kwargs=None, search_type="mmr"):
        config = {
            'search_kwargs': search_kwargs,
            'search_type': search_type,
        }
        # Return a configuration, instance, or adjustment based on the provided parameters.
        return config