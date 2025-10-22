import os
import ast
import shutil
from datasets import load_dataset
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from tqdm import tqdm

# FAISS veritabanı yolu ve embedding modeli adının tanımlanması
FAISS_INDEX_PATH = "faiss_recipe_index"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

# İşlenecek maksimum tarif sayısı
MAX_RECIPES = 10000

# Tarif verilerinin yüklenmesi ve Document formatına dönüştürülmesi
def create_recipe_documents():
    file_path = "manual_data/full_dataset.csv"
    print(f"'{file_path}' dosyası yükleniyor...")

    column_names = ['index', 'title', 'ingredients', 'directions', 'link', 'source', 'NER']
    dataset = load_dataset(
        "csv", 
        data_files=file_path, 
        column_names=column_names, 
        skiprows=1 
    )['train']

    print(f"{len(dataset)} tarif bulundu. {MAX_RECIPES} tanesi işlenecek.")

    documents = []

    # Tariflerin Başlık, Malzemeler ve Talimatlar bölümlerinin ayrıştırılması ve Document formatına dönüştürülüp listeye eklenmesi
    for recipe in tqdm(dataset.select(range(MAX_RECIPES)), desc="Tarifler işleniyor"):
        title = recipe.get('title', 'Başlık Yok')

        try:
            ingredients_list = ast.literal_eval(recipe.get('ingredients', '[]'))
            directions_list = ast.literal_eval(recipe.get('directions', '[]'))
        except (ValueError, SyntaxError):
            ingredients_list = []
            directions_list = []

        ingredients_text = ", ".join(ingredients_list) if ingredients_list else "Malzeme Yok"
        directions_text = " ".join(directions_list) if directions_list else "Talimat Yok"

        content = f"Başlık: {title}\nMalzemeler: {ingredients_text}\nTalimatlar: {directions_text}"

        doc = Document(page_content=content, metadata={"source": "recipe_nlg", "title": title})
        documents.append(doc)

    print(f"Toplam {len(documents)} tarif başarıyla Document formatına dönüştürüldü.")
    return documents

# FAISS vektör veritabanının oluşturulması ve kaydedilmesi
def create_and_save_faiss_index(documents):
    if not documents:
        print("İşlenecek belge bulunamadı. İndeks oluşturma işlemi iptal edildi.")
        return

    print(f"Embedding modeli yükleniyor: {MODEL_NAME}")
    embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)

    print("FAISS vektör veritabanı oluşturuluyor...")
    vector_store = FAISS.from_documents(documents, embeddings)

    print(f"Vektör veritabanı '{FAISS_INDEX_PATH}' klasörüne kaydediliyor...")
    vector_store.save_local(FAISS_INDEX_PATH)
    print("Vektör veritabanı başarıyla kaydedildi.")

# Ana program akışı
if __name__ == "__main__":
    if os.path.exists(FAISS_INDEX_PATH):
        print(f"'{FAISS_INDEX_PATH}' klasörü zaten mevcut.")
        user_input = input("Mevcut indeksi silip yeniden oluşturmak istiyor musunuz? (evet/hayır): ").lower()
        if user_input == 'evet':
            print(f"Mevcut indeks '{FAISS_INDEX_PATH}' siliniyor...")
            shutil.rmtree(FAISS_INDEX_PATH)
            print("İndeks silindi. Yeniden oluşturulacak.")
        else:
            print("İşlem iptal edildi. Mevcut indeks korunacak.")
            exit()

    recipe_documents = create_recipe_documents()
    create_and_save_faiss_index(recipe_documents)