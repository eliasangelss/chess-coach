import gradio as gr
from huggingface_hub import InferenceClient
# import random

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

'''
def echo(message, history):
    responses = ["It is certain.", "Reply hazy, try again.", "Don't count on it.", "It is decidedly so.", "Ask again later.", "My reply is no.", "Without a doubt.",
                "Better not tell you now.", "My sources say no.", "Yes, definitely.", "Cannot predict it now.", "Outlook isn't so good.", "You may rely on it.",
                "Concentrate and ask again.", "Very doubtful.", "As I see it, yes.", "Most likely.", "Outlook is good.", "Yes.", "Signs point to yes."]
    return random.choice(responses)
'''
       
        
chatbot = gr.ChatInterface(respond, title = "8-Ball-Turned-Chess-Tutor Bot!", 
                           description = "Interesting in learning more about basic chess tactics and strategy? Start by asking this newly created coach to teach you the best opening moves!")
chatbot.launch()
