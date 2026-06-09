import gradio as gr
from sentence_transformers import SentenceTransformer
import torch
from huggingface_hub import InferenceClient

with open("chess_basics.txt", "r", encoding = "utf-8") as file:
    chessText = file.read()

model = SentenceTransformer('all-MiniLM-L6-v2')

def preprocessText(text):
    cleanedText = text.strip()
    chunks = cleanedText.strip("\n")
    cleanedChunks = [chunk.strip() for chunk in chunks if chunk is not None]
    return cleanedChunks

def createEmbeddings(textChunks):
    chunkEmbeddings = model.encode(cleanedChunks, convert_to_tensor = True)
    return chunkEmbeddings

def getTopChunks(query, chunkEmbeddings, textChunks):
    queryEmbedding = model.encode(query, convert_to_tensor = True)
    queryEmbeddingNormalized = queryEmbedding / queryEmbedding.norm()
    chunkEmbeddingsNormalized = chunkEmbeddings / chunkEmbeddings.norm(dim = 1, keepdim = True)
    similarities = torch.matmul(chunkEmbeddingsNormalized, queryEmbeddingNormalized)
    topIndices = torch.topk(similarities, k=3).indices
    topChunks = [textChunks[i] for i in topIndices]
    return topChunks

cleanedChunks = preprocessText(chessText)
chunkEmbeddings = createEmbeddings(cleanedChunks)

client = InferenceClient("deepseek-ai/DeepSeek-R1") # spicy challenge

def respond(message, history):
    messages = [{"role": "system", 
                 "content": "You are a knowledgeable, straightforward AI coach specializing in chess tactics for beginners."
                 "Keep responses under 200 words unless asked for more detail about the strategy being used."
                 "Explanations should be clear with examples, following this format:"
                 "User: What is the point of Castling?" 
                 "AI: You use Castling in order to better protect the King during early stages of the game, either by Castling king-side or queen-side. These are denoted with O-O and O-O-O respectively, letting you know how many squares the King crossed over. You must move your corresponding Bishop, Knight, and potentially the Queen in order to achieve this position."}]
    
    if history: 
        messages.extend(history)

    topResults = getTopChunks(message, chunkEmbeddings, cleanedChunks)
    context = "\n\n".join(topResults) # new line, slight deviation from kwk tutorial
   
    messages.append({"role": "system",
                    "content": context})
    
    messages.append({"role": "user", 
                     "content": message})

    response = "" # created new variable, slight deviation from kwk tutorial
    
    responseStream = client.chat_completion(
        messages, stream = True, max_tokens = 1024, temperature = 0.4
    )
    
    for segment in responseStream:
        token = segment.choices[0].delta.content
        if token is not None:
            response += token
            yield response
       
        
chatbot = gr.ChatInterface(respond, title = "Chess Tutor Bot ♟️", 
                           description = "Interested in learning more about basic chess rules and strategy? Start by asking this newly created coach to teach you the best opening moves!")
chatbot.launch()
