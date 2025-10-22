import os
import gradio as gr
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from dotenv import load_dotenv

# Ortam değişkenlerinin yüklenmesi
load_dotenv()

# Modellerin ve veritabanının yüklenmesi
try:
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7, convert_system_message_to_human=True)
    print("Gemini modeli başarıyla yüklendi.")
except Exception as e:
    print(f"HATA: Gemini modeli yüklenemedi. GOOGLE_API_KEY'inizi kontrol edin: {e}")
    exit()

# Embedding modelinin tanımlanması
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

# FAISS veritabanının yolu
FAISS_INDEX_PATH = "faiss_recipe_index"

# FAISS veritabanının yüklenmesi
print("Embedding modeli ve FAISS veritabanı yükleniyor...")
try:
    embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    
    # Kaydedilmiş FAISS veritabanının yüklenmesi
    vector_store = FAISS.load_local(
        FAISS_INDEX_PATH, 
        embeddings,
        allow_dangerous_deserialization=True
    )
    print("FAISS veritabanı başarıyla yüklendi.")
except Exception as e:
    print(f"HATA: Veritabanı yüklenemedi. Lütfen önce index_data.py'yi çalıştırdığınızdan emin olun. Hata: {e}")
    exit()

retriever = MultiQueryRetriever.from_llm(
    retriever=vector_store.as_retriever(search_kwargs={"k": 15}),
    llm=llm
)

# RAG Zinciri Kurulumu

# Prompt Şablonu
template = """
Sen, sadece sağlanan tarif bağlamını kullanarak soruları yanıtlayan bir yemek tarifi asistanısın.
Kullanıcının sorusuna en uygun tarifi bulup, tarifi açık ve anlaşılır bir şekilde sun.
Eğer bağlamda tam olarak uygun bir tarif yoksa ancak malzemelerle ilgili benzer tarifler varsa, bu tarifleri sunmaya çalış. Eğer hiçbir ilgili tarif bulunamazsa, "Üzgünüm, bu konuda uygun bir tarif bulamadım. Başka bir şey denemek ister misiniz?" de.
Tarifi sunarken, tüm içeriği (Başlık, Malzemeler ve Talimatlar) Türkçe'ye çevirerek ve bu bölümleri net bir şekilde ayırarak sun.

Bağlam:
{context}

Soru:
{question}

Cevap:
"""

prompt = ChatPromptTemplate.from_template(template)

# Çıktı formatlama fonksiyonu
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
# Fonksiyonun örnek girdi ve çıktısı:
# Input: [Document(page_content="Başlık: Spagetti Carbonara\nMalzemeler: Spagetti, Yumurta, Parmesan, Pancetta\nTalimatlar: 1. Spagettiyi haşlayın...\n"), Document(page_content="Başlık: Tavuklu Makarna\nMalzemeler: Makarna, Tavuk, Krem\nTalimatlar: 1. Makarnayı haşlayın...\n")]
# Output: "Başlık: Spagetti Carbonara\nMalzemeler: Spagetti, Yumurta, Parmesan, Pancetta\nTalimatlar: 1. Spagettiyi haşlayın...\n\nBaşlık: Tavuklu Makarna\nMalzemeler: Makarna, Tavuk, Krem\nTalimatlar: 1. Makarnayı haşlayın...\n" 

# RAG zinciri tanımlaması
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Gradio Arayüzü

# Gradio fonksiyonu
def get_bot_response(message, history):
    print(f"Gelen Soru: {message}")
    response = rag_chain.invoke(message)
    print(f"Oluşturulan Cevap: {response}")
    return response

# Gradio arayüzünün başlatılması
print("Gradio arayüzü başlatılıyor...")

chat = gr.ChatInterface(
    type="messages",
    fn=get_bot_response,
    title="Yemek Tarifi Asistanı",
    description="Malzemelerinize veya canınızın çektiği bir yemeğe göre tarifler sorun. Ben sizin için bulurum!",
    theme="soft",
    examples=[        
        "Spagetti karbonara nasıl yapılır?", 
        "Tavuk ve mantar ile yapabileceğim bir yemek önerir misin?",
        "Çikolatalı bir tatlı tarifi önerir misin?",
        "Elimde patates, soğan ve kıyma var. Ne yapabilirim?"
    ],
    )   

chat.launch()