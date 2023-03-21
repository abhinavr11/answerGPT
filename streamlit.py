import nltk
nltk.download('punkt')
import streamlit as st
from streamlit_chat import message as st_message
import numpy as np
import sklearn
from sklearn.metrics.pairwise import cosine_similarity
from nltk import sent_tokenize

from sentence_transformers import SentenceTransformer
import os
import openai

file1 = open('WhatsApp Chat with London Wale 💂.txt', 'r',encoding="utf8")
Lines = file1.readlines()

Lines_new=[]
for line in Lines:
  if line != "\n" and len(line)>4:
    Lines_new.append(line.replace("\n",""))

documents = Lines_new

model_emb = SentenceTransformer('bert-base-nli-mean-tokens')

def generate_context(query, vector_embeddings, k):
    # Although it is not explicitly stated in the official document of sentence transformer, the original BERT is meant for a shorter sentence. We will feed the model by sentences instead of the whole documents.
    sentences = sent_tokenize(query)
    base_embeddings_sentences = model_emb.encode(sentences)
    base_embeddings = np.mean(np.array(base_embeddings_sentences), axis=0)
    scores = cosine_similarity([base_embeddings], vector_embeddings).flatten()
    scores_final=scores.tolist()
    sorted_scores = sorted(scores_final, reverse=True)
    top_k_values = sorted_scores[:k]
    top_k_indices = [i for i, value in enumerate(scores_final) if value in top_k_values][:k]
    sorted_scores = sorted(top_k_indices , reverse=False)
    final_doc=""
    entry=[]
    for i in range(0,len(top_k_indices)):
        for y in range(max(top_k_indices[i]-5,0), min(top_k_indices[i], len(documents)-1)):
            entry.append(y)
    entry=list(set(entry))
    for index in entry:
        final_doc=final_doc+documents[index]+"\n"
    print("Lines in the context: ", len(final_doc.split("\n")))
    return final_doc

vector_data= np.load('embeddings.npy')

def GPT_Completion(texts):
    ## Call the API key under your account (in a secure way)
    openai.api_key = os.environ.get('open_ai_key')#"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"
    response = openai.Completion.create(
    engine="text-davinci-003",
    prompt =  texts,
    temperature = 0.6,
    top_p = 1,
    max_tokens = 100,
    frequency_penalty = 0,
    presence_penalty = 0
    )
    return response.choices[0].text




st.title("London Vale")

if "history" not in st.session_state:
    st.session_state.history = []


def generate_answer():
    
    user_message = st.session_state.input_text
    # msg = {'message': user_message }
    # msg  = json.dumps(msg)
    # x = requests.post(url, data = msg)
    
    # message_bot = x.json()['results']
    query =  user_message  #"Where can I get good Pizza in London?"
    print(query,"\n \n \n")
    context= generate_context(query, vector_data, 20)
    print(context, "\n\n\n")
    recipe = f'Given a Whatsapp group chat data as "Chat History" and a query as "Query", output the answer to the query \n Chat History: {context} \n Query: {query} \n Output:'
    message_bot = GPT_Completion(recipe)
    st.session_state.history.append({"message": user_message, "is_user": True})
    st.session_state.history.append({"message": message_bot, "is_user": False})


st.text_input("Talk to the bot", key="input_text", on_change=generate_answer)

for chat in st.session_state.history:
    st_message(**chat)  # unpacking


