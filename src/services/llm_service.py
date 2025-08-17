import os
import pickle
import requests
from bs4 import BeautifulSoup
import numpy as np
import faiss
from openai import OpenAI


urls = [
    'https://eora.ru/cases/promyshlennaya-bezopasnost',
    'https://eora.ru/cases/lamoda-systema-segmentacii-i-poiska-po-pohozhey-odezhde',
    'https://eora.ru/cases/navyki-dlya-golosovyh-assistentov/karas-golosovoy-assistent',
    'https://eora.ru/cases/assistenty-dlya-gorodov',
    'https://eora.ru/cases/avtomatizaciya-v-promyshlennosti/chemrar-raspoznovanie-molekul',
    'https://eora.ru/cases/zeptolab-skazki-pro-amnyama-dlya-sberbox',
    'https://eora.ru/cases/goosegaming-algoritm-dlya-ocenki-igrokov',
    'https://eora.ru/cases/dodo-pizza-robot-analitik-otzyvov',
    'https://eora.ru/cases/ifarm-nejroset-dlya-ferm',
    'https://eora.ru/cases/zhivibezstraha-navyk-dlya-proverki-rodinok',
    'https://eora.ru/cases/sportrecs-nejroset-operator-sportivnyh-translyacij',
    'https://eora.ru/cases/avon-chat-bot-dlya-zhenshchin',
    'https://eora.ru/cases/navyki-dlya-golosovyh-assistentov/navyk-dlya-proverki-loterejnyh-biletov',
    'https://eora.ru/cases/computer-vision/iss-analiz-foto-avtomobilej',
    'https://eora.ru/cases/purina-master-bot',
    'https://eora.ru/cases/skinclub-algoritm-dlya-ocenki-veroyatnostej',
    'https://eora.ru/cases/skolkovo-chat-bot-dlya-startapov-i-investorov',
    'https://eora.ru/cases/purina-podbor-korma-dlya-sobaki',
    'https://eora.ru/cases/purina-navyk-viktorina',
    'https://eora.ru/cases/dodo-pizza-pilot-po-avtomatizacii-kontakt-centra',
    'https://eora.ru/cases/dodo-pizza-avtomatizaciya-kontakt-centra',
    'https://eora.ru/cases/icl-bot-sufler-dlya-kontakt-centra',
    'https://eora.ru/cases/s7-navyk-dlya-podbora-aviabiletov',
    'https://eora.ru/cases/workeat-whatsapp-bot',
    'https://eora.ru/cases/absolyut-strahovanie-navyk-dlya-raschyota-strahovki',
    'https://eora.ru/cases/kazanexpress-poisk-tovarov-po-foto',
    'https://eora.ru/cases/kazanexpress-sistema-rekomendacij-na-sajte',
    'https://eora.ru/cases/intels-proverka-logotipa-na-plagiat',
    'https://eora.ru/cases/karcher-viktorina-s-voprosami-pro-uborku',
    'https://eora.ru/cases/chat-boty/purina-friskies-chat-bot-na-sajte',
    'https://eora.ru/cases/nejroset-segmentaciya-video',
    'https://eora.ru/cases/chat-boty/essa-nejroset-dlya-generacii-rolikov',
    'https://eora.ru/cases/qiwi-poisk-anomalij',
    'https://eora.ru/cases/frisbi-nejroset-dlya-raspoznavaniya-pokazanij-schetchikov',
    'https://eora.ru/cases/skazki-dlya-gugl-assistenta',
    'https://eora.ru/cases/chat-boty/hr-bot-dlya-magnit-kotoriy-priglashaet-na-sobesedovanie'
    ]


# Глобальный список URL


class LlmService:
    def __init__(self, urls, openai_api_key,
                 cache_folder="cache",
                 chunk_size=500,
                 overlap=100):
        self.urls = urls
        self.openai = OpenAI(api_key=openai_api_key)

        self.chunk_size = chunk_size
        self.overlap = overlap

        self.cache_folder = cache_folder
        os.makedirs(self.cache_folder, exist_ok=True)

        # Пути для сохранения эмбеддингов и текстов чанков
        self.embeddings_path = os.path.join(self.cache_folder, "embeddings.npy")
        self.chunk_texts_path = os.path.join(self.cache_folder, "chunk_texts.pkl")
        self.index_path = os.path.join(self.cache_folder, "faiss_index.bin")

        self.embeddings = None
        self.chunk_texts = []
        self.index = None

        # Загружаем или создаём все ресурсы
        self._load_or_prepare_embeddings()

    def pars_link(self, url):
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 \
                (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            body = soup.find('body')
            text = body.get_text(separator=' ', strip=True) if body else ''
            return text
        except Exception as e:
            print(f"Ошибка при парсинге URL {url}: {str(e)}")
            return ''

    def split_text_into_chunks(self, text):
        chunks = []
        start = 0
        length = len(text)
        while start < length:
            end = min(start + self.chunk_size, length)
            chunk = text[start:end]
            chunks.append(chunk)
            start += self.chunk_size - self.overlap
        return chunks

    def _save_pickle(self, data, path):
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def _load_pickle(self, path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    def _save_index(self):
        faiss.write_index(self.index, self.index_path)

    def _load_index(self):
        return faiss.read_index(self.index_path)

    def _load_or_prepare_embeddings(self):
        # Проверяем, есть ли на диске уже эмбеддинги и чанки текстов
        if os.path.isfile(self.embeddings_path) and os.path.isfile(self.chunk_texts_path) and os.path.isfile(self.index_path):
            print("Загружаем готовые эмбеддинги и тексты чанков из кэша...")
            self.embeddings = np.load(self.embeddings_path)
            self.chunk_texts = self._load_pickle(self.chunk_texts_path)
            self.index = self._load_index()
        else:
            print("Кэш эмбеддингов не найден, начинаем парсинг и генерацию...")
            self._full_prepare()

    def _full_prepare(self):
        print("Парсим и собираем все тексты...")
        all_chunks = []
        for url in self.urls:
            text = self.pars_link(url)
            if text:
                chunks = self.split_text_into_chunks(text)
                all_chunks.extend(chunks)

        print(f"Общее число чанков: {len(all_chunks)}")
        self.chunk_texts = all_chunks

        print("Генерируем эмбеддинги для чанков...")
        self.embeddings = []
        for chunk in self.chunk_texts:
            response = self.openai.embeddings.create(input=chunk, model="text-embedding-3-large")
            embedding = response['data'][0]['embedding']
            self.embeddings.append(np.array(embedding, dtype=np.float32))

        if not self.embeddings:
            raise ValueError("Не удалось сгенерировать ни одного эмбеддинга.")

        embeddings_matrix = np.vstack(self.embeddings)
        dim = embeddings_matrix.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        faiss.normalize_L2(embeddings_matrix)
        self.index.add(embeddings_matrix)

        print(f"Создан FAISS индекс с {self.index.ntotal} векторами.")

        # Сохраняем на диск для последующих запусков
        np.save(self.embeddings_path, embeddings_matrix)
        self._save_pickle(self.chunk_texts, self.chunk_texts_path)
        self._save_index()
        print("Запись данных кэша завершена.")

    def _search_similar_chunks(self, query_embedding, top_k=5):
        faiss.normalize_L2(query_embedding)
        distances, indices = self.index.search(query_embedding, top_k)
        results = [self.chunk_texts[idx] for idx in indices[0] if idx != -1]
        return results

    def answer_question(self, question, top_k=5):
        response = self.openai.embeddings.create(
            input=question,
            model="text-embedding-3-large"
        )
        question_embedding = np.array(response['data'][0]['embedding'], dtype=np.float32).reshape(1, -1)

        relevant_chunks = self._search_similar_chunks(question_embedding, top_k=top_k)
        context = "\n\n".join(relevant_chunks)

        prompt = f"""Ты помощник компании EORA. Используй следующий контекст из наших проектов и дай профессиональный,
        полный и понятный ответ на вопрос пользователя.

Контекст:
{context}

Вопрос: {question}
Ответ:"""

        chat_resp = self.openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user", "content": prompt}],
            max_tokens=600,
            temperature=0.3,
        )
        answer = chat_resp.choices[0].message.content
        return answer
