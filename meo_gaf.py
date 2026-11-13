from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import dump
from sklearn.metrics.pairwise import cosine_similarity
import json
import os
import requests

STORE_DIR = Path("store")
DATA_DIR = Path("data")
STORE_DIR.mkdir(exist_ok=True)
UZHE_USAL = STORE_DIR / "used_ids.json"

marker = '#'

OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2"

def chunk_text(text):
    chunks = []
    positions = []
    idx = 0
    while True:
        idx = text.find(marker, idx)
        if idx == -1:
            break
        if idx == 0 or text[idx - 1] == "\n":
            positions.append(idx)
        idx += len(marker)

    if not positions:
        if text.strip():
            return [text.strip()]
        return []

    for i, start in enumerate(positions):
        end = positions[i + 1] if i + 1 < len(positions) else len(text)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

    return chunks


records = []
for path in sorted(DATA_DIR.glob("*.txt")):
    text = path.read_text(encoding="utf-8")
    for number, chunk in enumerate(chunk_text(text), start=1):
        records.append({
            "id": f"{path.stem} - {number}",
            "source": path.name,
            "chunk": number,
            "text": chunk,
        })

print(f"Найдено чанков: {len(records)}")
texts = [record["text"] for record in records]

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=1,
    sublinear_tf=True
)
matrix = vectorizer.fit_transform(texts)

dump(vectorizer, STORE_DIR / "vectorizer.joblib")
dump(matrix, STORE_DIR / "vectors.joblib")
(STORE_DIR / "metadata.json").write_text(
    json.dumps(records, ensure_ascii=False, indent=2),
    encoding="utf-8"
)

if UZHE_USAL.exists():
    content = UZHE_USAL.read_text(encoding="utf-8").strip()
    if content:
        try:
            used_ids = set(json.loads(content))
        except json.JSONDecodeError:
            used_ids = set()
    else:
        used_ids = set()
else:
    used_ids = set()

def ask_ollama(question: str, context: str, model: str = OLLAMA_MODEL) -> str:
    """Отправляет вопрос и найденный контекст в Ollama, возвращает ответ."""
    prompt = (
        "Ты — ассистент, который отвечает на вопросы, используя только предоставленный контекст.\n"
        f"Контекст:\n{context}\n\n"
        f"Вопрос: {question}\n\n"
        "Ответ:"
    )
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_predict": 512,
        },
    }
    try:
        resp = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "").strip()
    except requests.ConnectionError:
        return ("[Ошибка] Не удалось подключиться к Ollama. "
                "Убедитесь, что сервер запущен: ollama serve")
    except requests.Timeout:
        return "[Ошибка] Ollama не ответила за 120 секунд. Попробуйте ещё раз."
    except requests.HTTPError as e:
        return f"[Ошибка HTTP] {e.response.status_code}: {e.response.text}"
    except Exception as e:
        return f"[Неизвестная ошибка] {e}"

while True:
    if len(used_ids) >= len(records):
        print("\nВсе задания из базы уже были выданы!")
        break

    query = input("\nВведите запрос (или 'выход' для выхода): ")
    if query.strip().lower() == "выход":
        os.remove('C:/Users/lsasi/OneDrive/Desktop/probnikraga/store/used_ids.json')
        break

    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, matrix).ravel()

    
    for i, record in enumerate(records):
        if record["id"] in used_ids:
            scores[i] = -1.0

    order = scores.argmax()
    best_s = scores[order]

    
    if best_s <= 0:
        print("\nДля этого задания нет подходящего ответа ")
        continue

    record = records[order]

    kakoy = used_ids
    used_ids.add(record["id"])
    UZHE_USAL.write_text(
        json.dumps(list(used_ids), ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print("\n=== НАЙДЕНО ЗАДАНИЕ ===")
    print(f"Источник: {record['source']}")
    print(f"Chunk: {record['chunk']}")
    print(f"ID: {record['id']}")
    print(f"Сходство: {best_s:.4f}")
    print("-" * 25)
    print(record["text"])
    print(f"\n(Осталось доступных заданий: {len(records) - len(used_ids)})")
    print('Уже использовали: ', kakoy)
    
    neironka = input('хотите получить ответ от олламы? (напишите "да", если хотите): ')
    if neironka.strip().lower() == "да":
        neiro_query = input("Введите запрос к модели по заданию: ")
        print("=== ОТВЕТ OLLAMA ===") 
        answer = ask_ollama(neiro_query, record["text"])
        print(answer)
    else: 
        continue