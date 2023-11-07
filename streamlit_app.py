import os
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA, RetrievalQAWithSourcesChain, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI
from langchain.vectorstores import Qdrant
from qdrant_client import QdrantClient
from PIL import Image

##openai_api_key = os.getenv("OPENAI_API_KEY")

QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
model_name_embeddings = "text-embedding-ada-002"
model_name_llm = "gpt-4" #"gpt-3.5-turbo"

st.set_page_config(page_title = "✈🧳 Chatbot de turismo - Perú", page_icon = "🇵🇪")

listplaces = {}
listplaces = {
    "cusco" : "Machu Picchu",
    "arequipa" : "Plaza de Armas",
    "cajamarca" : "Sitios turísticos de Cajamarca"
}

# Clear the Chat Messages
def clear_chat_history():
    st.session_state.messages = [{"role" : "assistant", "content": msg_chatbot.format(place = place)}]

#Create a Side bar
with st.sidebar:
    
    st.title("✈🧳 Chatbot de turismo - Perú")

    place = st.selectbox('Eliga el lugar turístico donde es tu próximo viaje',('Cusco', 'Arequipa', 'Cajamarca'), key = "place", on_change = clear_chat_history)
    
    st.markdown('Colaboradores del proyecto:')
    st.markdown('[Martin Grados](https://www.linkedin.com/in/luisgrados)')
    st.markdown('[James Espichan](https://www.linkedin.com/in/james-muller-espichan-vilca)')

    openai_api_key = st.text_input("Ingrese tu API Key de OpenAI y dale Enter para habilitar el chatbot", key = "chatbot_api_key", type = "password")
    "[Genera tu API Key (OpenAI)](https://platform.openai.com/account/api-keys)"

    place_lower = place.lower()
    

    st.markdown(
        """
        En un mundo en constante evolución tecnológica, la industria turística busca 
        aprovechar las herramientas digitales para mejorar la experiencia del visitante. 
        Cusco, uno de los destinos turísticos más emblemáticos de Perú y del mundo, 
        requiere soluciones innovadoras para atender a la creciente demanda de información y 
        servicios de calidad por parte de sus visitantes. En este contexto, 
        un chatbot se presenta como una solución tecnológica ideal para satisfacer nuestras necesidades.
        
        ### Propósito

        El propósito principal del chatbot para el sector turismo en Cusco es facilitar la planificación
        y ejecución de la experiencia turística de los visitantes a través de 
        respuestas inmediatas, precisas y personalizadas sobre lugares de interés, 
        desplazamientos y costos asociados.

        ### Objetivos

        - Información actualizada y accesible : Ofrecer a los turistas información actualizada y relevante sobre puntos de interés en Cusco.
        - Orientación de desplazamientos : Proveer recomendaciones sobre cómo desplazarse en la ciudad y sus alrededores, ofreciendo opciones desde transporte público hasta tours organizados.
        - Transparencia en costos : Brindar detalles sobre precios actuales de entradas, pasajes, tours y otros servicios, permitiendo al turista presupuestar y planificar mejor su viaje.
        - Atención 24/7 : Dado que es un servicio digital, el chatbot puede atender consultas en cualquier momento, beneficiando a aquellos turistas que operan en diferentes zonas horarias o que necesitan información fuera de horas laborables.
        - Adaptabilidad lingüística : Posibilidad de interactuar con turistas de diversas nacionalidades, ofreciendo respuestas en varios idiomas.
        
        ### Fuentes 

        - Perú Rail - (https://www.perurail.com)
        - Perú Travel - (https://www.peru.travel)
        - Patrimonio Mundial - (https://patrimoniomundial.cultura.pe)
        - Cusco Perú - (https://www.cuscoperu.com)
        - Boleto Machu Picchu - (https://www.boletomachupicchu.com)
    """
    )

msg_chatbot = """
        Soy un chatbot que te ayudará a conocer {place}: 
        
        ### Puedo ayudarte con lo relacionado a:

        - 🚶🏻 Como desplazarte por la ciudad.
        - 🥘 Restaurantes típicos imperdibles
        - 💰 Precios del tren de Perú Rail
        - 🎟 Boleto para entrar a Machu Picchu

        ### Preguntas frecuentes

        - ¿qué comida típica puedo comer?
        - ¿qué puedo hacer en esta hermosa ciudad?
        - ¿qué restaurantes me recomiendas visitar? ponlo en una lista
        - ¿cómo puedo comprar los boletos para entrar a Machu Picchu?
        - ¿cómo puedo desplazarme en la ciudad?
        - ¿qué tipo de servicios de tren tiene Perú Rail? ponlo en una lista
        
"""

#Store the LLM Generated Reponese
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content" : msg_chatbot.format(place = place)}]
    
# Diplay the chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Create a Function to generate
def generate_openai_pinecone_response(prompt_input, place):
    
    llm = ChatOpenAI(
        openai_api_key = openai_api_key,
        model_name = model_name_llm,
        temperature = 0
    )

    template = """Responda a la pregunta basada en el siguiente contexto, eres un asistente virtual experto en el turismo de Cusco.
    Si no puedes responder a la pregunta, usa la siguiente respuesta "No lo sé disculpa, puedes contactar a  los colaboradores del proyecto [Martin Grados] (https://www.linkedin.com/in/luisgrados) y  [James Espichan] (https://www.linkedin.com/in/james-muller-espichan-vilca), para que agregar esta información a la base de conocimiento del chatbot"

    Contexto: 
    {context}.
    Pregunta: {question}
    Respuesta utilizando también emoticones: 
    """
    
    prompt = PromptTemplate(
        input_variables = ["context", "question"],
        template = template
    )

    chain_type_kwargs = {"prompt": prompt}

    qa = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type = 'stuff',
        retriever = vectorstore.as_retriever(search_kwargs={'filter': {'place': place.lower()}}), #as_retriever(), #add filter dependiendo del lugar que se seleccione en la lista
        verbose = True,
        chain_type_kwargs = chain_type_kwargs
    )

    output = qa.run(prompt_input)

    return output

st.sidebar.button('Limpiar historial de chat', on_click = clear_chat_history)

#Si ya se ingresó el API Key, habilitar el chatbot
if openai_api_key:

    embed = OpenAIEmbeddings(
        model = model_name_embeddings,
        openai_api_key = openai_api_key
    )
    
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    vectorstore = Qdrant(client, QDRANT_COLLECTION_NAME, embeddings=embed)

    prompt = st.chat_input("Ingresa tu pregunta", max_chars = 100)
    if prompt:
        st.session_state.messages.append({"role": "user", "content" : prompt})
        with st.chat_message("user"):
            st.write(prompt)
    
    # Generar una nueva respuesta si el último mensaje no es de un assistant, sino un user
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Esperando respuesta, dame unos segundos."):
                
                response = generate_openai_pinecone_response(prompt, place)
                placeholder = st.empty()
                full_response = ''
                
                for item in response:
                    full_response += item
                    placeholder.markdown(full_response)
                placeholder.markdown(full_response)
    
        message = {"role" : "assistant", "content" : full_response}
        st.session_state.messages.append(message) #Agrega elemento a la caché de mensajes de chat.
