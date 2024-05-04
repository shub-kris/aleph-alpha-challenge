# Install requiremtents
# pip install gradio==4.29.0

import numpy as np
from numpy.linalg import norm
import pandas as pd
import os
import gradio as gr

from aleph_alpha_client import (Client, 
                                SemanticEmbeddingRequest,
                                SemanticEmbeddingResponse,
                                SemanticRepresentation, 
                                Prompt)

client = Client(token=os.environ['AA_TOKEN'])
old_tickets_df = pd.read_pickle("old_tickets_embeddings.pkl")

model_id = "luminous-base" # id of model to use for Embedding
# data_dir = "" # directory where the data is stored
# new_tickets_df = pd.read_csv(os.path.join(data_dir, "new_tickets.csv")) #load the new ticket

def embed(text: str, model_id: str, embed_dim: int = 128) -> SemanticEmbeddingResponse:
    """Helper function to embed text using the Aleph Alpha Large Language Model.

    Args:
        text (str): text to embed.
        model_id (str): Model to use for embedding.
        embed_dim (int): Dimension of the embeddings.

    Returns:
        SemanticEmbeddingResponse: A response containing the embeddings of the text.
    """
    request = SemanticEmbeddingRequest(prompt=Prompt.from_text(text), 
                                       representation=SemanticRepresentation.Symmetric, 
                                       compress_to_size=embed_dim)
    response = client.semantic_embed(request, model=model_id)
    return response



def get_similar_old_tickets(issue: str, description: str, top_k: int):
    """Helper function to get top-k similar old tickets based on issue and description of new ticket
    
    Args:
        issue (str): Issue of the ticket
        description (str): Description of the ticket.
        top_k (int): Number of similar old tickets to return.    
    """
    
    ## Join issue and description to form a single text and embed
    text = issue + '\n' + description
    response = embed(text=text, model_id="luminous-base")
    
    ## Create a copy of old_tickets dataframe
    results_df = old_tickets_df.copy()
    
    ## Calculate cosine similarity with old tickets embeddings
    results_df['similarity_score'] = old_tickets_df['embeddings'].apply(lambda x: np.dot(x, response.embedding) / (norm(x) * norm(response.embedding)))

    ## Get top k similar old tickets
    similar_old_tickets = results_df.sort_values('similarity_score', ascending=False).head(top_k)
    
    ## Drop some columns
    columns_to_remove = ["embeddings", "description_length", "context"]
    similar_old_tickets = similar_old_tickets.drop(columns_to_remove, axis=1)
    
    # For gradio change dtype of Date to str to render it
    similar_old_tickets['Date'] = similar_old_tickets['Date'].astype('str') 
    return gr.Dataframe(similar_old_tickets, type="pandas")



demo = gr.Interface(
    fn=get_similar_old_tickets,
    inputs=["text", "text", "number"],
    outputs=["dataframe"],
    title = "Semantic Ticket Matching System",
    description = "Get Similar Old Tickets based on Issue and Description of new Ticket",
    examples = [
        ["VPN connection timeout", "VPN connection times out frequently during use", 2],
        ["Emails not syncing on mobile", "User's emails are not syncing properly on their mobile device", 3],
        ["New software installation request", "A request to install new project management software", 3],
        ["Laptop screen flickering", "The laptop screen starts flickering intermittently during use.", 2],
        ["Lost password for multiple accounts", "A user has lost passwords for multiple accounts and needs resets", 2]]
)


if __name__ == "__main__":
    demo.launch()
	