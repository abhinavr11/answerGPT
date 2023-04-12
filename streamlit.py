import nltk
nltk.download('punkt')
import streamlit as st
from streamlit_chat import message as st_message
import numpy as np
import sklearn
from sklearn.metrics.pairwise import cosine_similarity
from nltk import sent_tokenize

from sentence_transformers import SentenceTransformer
import os,time,glob
import openai


# @st.cache_data




@st.cache_resource
def return_model():
    return SentenceTransformer('bert-base-nli-mean-tokens')

model_emb = return_model()


def generate_context(query, vector_embeddings, k):
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

def getDocs(filename):
    file1 = open(filename, 'r',encoding="utf-8")
    Lines = file1.readlines()
    Lines_new=[]
    for line in Lines:
        if line != "\n" and len(line)>4:
            Lines_new.append(line.replace("\n",""))
    del Lines
    return Lines_new

def getEmbs(document,name):
    
    if [f for f in glob.glob(f"embs_{name}.npy")] :
        vector_data= np.load(f'embs_{name}.npy')
        return vector_data
    
    vectors = []
    with st.spinner(text="Processing the chats hang in there ..."):
        my_bar = st.progress(0)
        for i, document in enumerate(documents):

            sentences = sent_tokenize(document)
            embeddings_sentences = model_emb.encode(sentences)
            embeddings = np.mean(np.array(embeddings_sentences), axis=0)

            vectors.append(embeddings)
            my_bar.progress(i/len(documents))
    
    with open(f'embs_{name}.npy', 'wb') as f:
        np.save(f, np.array(vectors))
    return vectors



def GPT_Completion(texts):
    ## Call the API key under your account (in a secure way)
    openai.api_key = os.environ.get('open_ai_key')
    response = openai.Completion.create(
    engine="text-davinci-003",
    prompt =  texts,
    temperature = 0,
    top_p = 1,
    max_tokens = 100,
    frequency_penalty = 0,
    presence_penalty = 0
    )
    return response.choices[0].text

def generate_answer(name_of_person,vector_data):
    
    user_message = st.session_state.input_text
    query =  user_message  
    #print(query,"\n \n \n")
    context= generate_context(query, vector_data, 15)
    #print(context, "\n\n\n")
    recipe = f'Given a Whatsapp chat data as "Chat History" reply in the style of the person named {name_of_person} in the chat \n Chat History: {context} \n Query: {query} \n Output:'
    message_bot = GPT_Completion(recipe)
    st.session_state.history.append({"message": user_message, "is_user": True})
    st.session_state.history.append({"message": message_bot, "is_user": False})

def getUploadedFile():
    st.session_state.uploaded_file = st.file_uploader("Choose a file",key="file_upload") 
    if uploaded_file:
        return uploaded_file


if "history" not in st.session_state:
    st.session_state.history = []

if 'ctr' not in st.session_state:
    st.session_state['ctr'] = 0



def uploader_callback():
    if st.session_state['file_uploader'] is not None:
        st.session_state['ctr'] += 1
        print('Uploaded file #%d' % st.session_state['ctr'])
    
uploaded_file = st.file_uploader(label="File uploader", on_change=uploader_callback, key="file_uploader")

#uploaded_file = st.file_uploader("Choose a file",key="file_upload")#getUploadedFile()   
if uploaded_file:
    with open(uploaded_file.name,"wb") as f: 
        f.write(uploaded_file.getbuffer()) 
    documents = getDocs(uploaded_file.name)
    vector_data = getEmbs(document=documents,name=uploaded_file.name) 
       
    print('done')
    name_of_person = st.text_input(
    "Enter the name of the person you want the bot to talk like 👇",
    )  
    name_of_person = 'Dad'
  
    
    st.text_input("Talk to the bot", key="input_text", on_change=generate_answer,args=(name_of_person,vector_data))
    for chat in st.session_state.history:
        st_message(**chat) 








