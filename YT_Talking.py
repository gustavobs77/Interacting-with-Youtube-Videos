#!/usr/bin/env python
# coding: utf-8

# # Método 1 - Aprendizado ( Method 1- learning)

# ## installing tools and libraries

# In[ ]:


#!pip install langchain-community langchain-huggingface langchain_ollama
#!pip install youtube-transcript-api
#!pip install pytube
#!pip install --upgrade pytube


# In[114]:


import os
import io
import getpass
from langchain_community.document_loaders import YoutubeLoader
from langchain_community.llms import HuggingFaceHub
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate


# ## Getting the video informations

# In[240]:


video_loader = YoutubeLoader.from_youtube_url("https://www.youtube.com/watch?v=PeMlggyqz0Y&t=6s", language = ["pt", "pt-BR","en"] )


# In[242]:


infos = video_loader.load()
#infos


# In[243]:


transcription = infos[0].page_content
#transcricao


# In[244]:


video_info = f"""
Informações do vídeo:
Título: {infos[0].metadata.get('title', 'unailable')}
Autor: {infos[0].metadata.get('author', 'unailable')}
Data: {infos[0].metadata.get('publish_date', 'unailable')[:10]}
URL: https://www.youtube.com/watch?v={infos[0].metadata.get('source', 'unailable')}

Transcrição: {infos[0].page_content}
"""


# In[245]:


#creating a txt file with the video infos
with io.open("transcription.txt", "w", encoding = "utf-8") as f:
    for doc in infos:
        f.write(video_info)


# ## Load models - LLM - HuggingFaceHub or Ollama (phi3)

# In[247]:


# If you choose a Huggingface template, the access token will be required
# If you chose the Ollama phi3 model, you must download ollama on : https://ollama.com/download. After that, open the command prompt and type :ollama pull phi3
os.environ["HUGGINGFACEHUB_API_TOKEN"] = getpass.getpass()


# ## Defining the model functions

# In[254]:


def model_hf_hub(model = "meta-llama/Meta-Llama-3-8B-Instruct", temperature = 0.1):
    llm = HuggingFaceHub(
        repo_id = model, 
        model_kwargs={
        "temperature":temperature, 
        "return_full_text":False, 
        "max_new_tokens":1024
                     }
                        )
    return llm


# In[256]:


def model_ollama(model = "phi3", temperature = 0.1):
    llm = ChatOllama(model = model,temperature = temperature)
    return llm


# In[258]:


model_class = "ollama" #[hf_hub, "ollama"]
if model_class == "hf_hub":
    llm = model_hf_hub()
else:
    llm = model_ollama()


# ## Creating the Prompt and the Chain

# In[261]:


system_prompt = "você é um assistente prestativo e deve responder a uma consulta baseada em uma transcrição de um vídeo"
inputs = "Query: {query} \n Transcription: {transcription}"
if model_class.startswith("hf"):
    user_prompt = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
else:
    user_prompt = "{}".format(inputs)


# In[263]:


prompt_template = ChatPromptTemplate.from_messages([("system",system_prompt), ("user", user_prompt)])


# In[265]:


chain = prompt_template | llm | StrOutputParser()


# ## Testing

# In[268]:


res = chain.invoke({"transcription":transcricao, "query":"qual a principal informação do vídeo ?"})
print(res)


# # Method 2- Unified Pipeline

# In[ ]:


import os
import io
import getpass
from langchain_community.document_loaders import YoutubeLoader
from langchain_community.llms import HuggingFaceHub
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate


# In[213]:


def llm_chain(model_class):
  system_prompt = "Você é um assistente virtual prestativo e deve responder a uma consulta com base na transcrição de um vídeo, que será fornecida abaixo."

  inputs = "Consulta: {consulta} \n Transcrição: {transcricao}"

  if model_class.startswith("hf"):
      user_prompt = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>".format(inputs)
  else:
      user_prompt = "{}".format(inputs)

  prompt_template = ChatPromptTemplate.from_messages([
      ("system", system_prompt),
      ("user", user_prompt)
  ])

  ### Carregamento da LLM
  if model_class == "hf_hub":
      llm = model_hf_hub()
  
  else:
      llm = model_ollama()

  chain = prompt_template | llm | StrOutputParser()

  return chain


# In[217]:


def get_video_info(url_video, language="pt", translation=None):

  video_loader = YoutubeLoader.from_youtube_url(
      url_video,
      language=language,
      translation=translation,
  )

  infos = video_loader.load()[0]
  metadata = infos.metadata
  transcript = infos.page_content

  return transcript, metadata


# In[219]:


transcript, metadata = get_video_info("https://www.youtube.com/watch?v=II28i__Tf3M")


# In[223]:


#metadata, transcript


# In[236]:


def interpret_video(url, query="resuma", model_class="hf_hub", language="pt", translation=None):

  try:
    transcript, metadata = get_video_info(url, language, translation)

    chain = llm_chain(model_class)

    res = chain.invoke({"transcricao": transcript, "consulta": query})
    print(res)

  except Exception as e:
    print("Erro ao carregar transcrição")
    print(e)


# ## Inserting the Parameters

# In[270]:


url_video = "https://www.youtube.com/watch?v=PeMlggyqz0Y&t=6s" # @param {type:"string"}
query_user = "sumarize de forma clara de entender" # @param {type:"string"}
model_class = "ollama" # @param ["hf_hub", "ollama"]
language = ["pt", "pt-BR", "en"] # @param {type:"string"}


# In[272]:


interpret_video(url_video, query_user, model_class, language)


# In[ ]:




