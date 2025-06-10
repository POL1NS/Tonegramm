import sys
import subprocess
import pkg_resources
import importlib.util
import re
import json
import asyncio
from collections import defaultdict
import signal
import time

# Обработка прерывания
class GracefulExiter:
    def __init__(self):
        self.state = False
        signal.signal(signal.SIGINT, self.change_state)

    def change_state(self, signum, frame):
        print("\nПолучен сигнал прерывания, завершаю работу...")
        self.state = True

    def exit(self):
        return self.state

# Проверка прерывания
flag = GracefulExiter()

# Установка отсутствующих пакетов
required = {'telethon', 'nest_asyncio', 'pymorphy3', 'transformers', 'torch', 'pandas', 'plotly', 'nltk'}
installed = {pkg.key for pkg in pkg_resources.working_set}
missing = required - installed

if missing:
    print(f"Установка отсутствующих пакетов: {missing}")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', *missing], stdout=subprocess.DEVNULL)
        print("Пакеты успешно установлены!")
    except subprocess.CalledProcessError as e:
        print(f"Ошибка при установке: {e}. Попытка установки с --user")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--user', *missing], stdout=subprocess.DEVNULL)

# Проверка наличия telethon
if importlib.util.find_spec("telethon") is None:
    raise ModuleNotFoundError("Не удалось установить telethon. Установите вручную: pip install telethon")

# Импорт остальных модулей после установки
from telethon.sync import TelegramClient
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
import pymorphy3
import plotly.express as px
import plotly.io as pio
import nest_asyncio
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Инициализация
nest_asyncio.apply()
nltk.download('vader_lexicon')

# Настройки API Telegram
api_id = 'айпи айди'
api_hash = "айпи хаш"
phone = "номер телефона"
channels = ['ссылка на канал']

# Загрузка модели для русского языка
ru_model_name = "blanchefort/rubert-base-cased-sentiment"
ru_tokenizer = AutoTokenizer.from_pretrained(ru_model_name, model_max_length=512)
ru_model = AutoModelForSequenceClassification.from_pretrained(ru_model_name)

# Морфологический анализатор
morph = pymorphy3.MorphAnalyzer()

def detect_language(text):
    if re.search(r'[а-яА-Я]', text):
        return 'ru'
    elif re.search(r'[a-zA-Z]', text):
        return 'en'
    return 'unknown'

def get_word_forms(word):
    try:
        parsed = morph.parse(word)[0]
        return set([w.word for w in parsed.lexeme])
    except:
        return {word}

def preprocess_text(text):
    emoji_map = {
        '🔥': ' отлично ', '❤️': ' любовь ', '⭐️': ' звезда ', '✨': ' прекрасно ',
        '👍': ' одобрение ', '🎉': ' праздник ', '😊': ' улыбка ', '😂': ' смех ',
        '😍': ' обожание ', '🙏': ' спасибо ', '🤗': ' объятия ', '😢': ' грусть ',
        '😡': ' злость ', '🤔': ' задумчивость ', '👏': ' аплодисменты '
    }
    for emoji, replacement in emoji_map.items():
        text = text.replace(emoji, replacement)
    return text

def get_sentiment_category(overall_score, text=None, threshold=30):  # Увеличил порог
    if not text:
        if overall_score >= threshold:
            return "Позитивно"
        elif overall_score <= -threshold:
            return "Негативно"
        return "Нейтрально"
    
    # Расширенные списки маркеров
    positive_markers = ['❤️', '🔥', 'молодцы', 'поздравля', 'отличн', 'умни', 
                       'супер', 'спасибо', 'благодарю', 'класс', 'круто', 
                       'замечательно', 'прекрасно', 'восхитительно', 'рад', 'счастье']
    
    negative_markers = ['плохо', 'ужасно', 'кошмар', 'разочарован', 'злюсь', 
                       'ненавижу', 'бесит', 'отвратительно', 'грустно', 'обидно']
    
    neutral_markers = ['не ', 'нет', 'но ', 'без ', 'никто', 'никогда', 'проблема',
                      'ошибка', 'исправьте', 'неправильно']
    
    # Подсчет маркеров
    text_lower = text.lower()
    pos_count = sum(1 for m in positive_markers if m in text_lower)
    neg_count = sum(1 for m in negative_markers if m in text_lower)
    neu_count = sum(1 for m in neutral_markers if m in text_lower)
    
    # Корректировка оценки
    adjusted_score = overall_score + (pos_count * 8) - (neg_count * 10) - (neu_count * 3)
    
    if adjusted_score >= threshold or pos_count >= 3:
        return "Позитивно"
    elif adjusted_score <= -threshold or neg_count >= 2:
        return "Негативно"
    return "Нейтрально"

async def analyze_channel_async(client, channel_url):
    results = []
    conflict_cases = [] 
    user_stats = defaultdict(lambda: {'count': 0, 'total_score': 0, 'positive': 0, 'negative': 0})
    
    try:
        target = await client.get_entity(channel_url)
        channel_name = getattr(target, 'title', 'Unknown Channel')
        
        messages = await client.get_messages(target, limit=1000)  # Уменьшил лимит
        
        for i, msg in enumerate(messages):
            if flag.exit():
                print(f"Прерывание обработки канала {channel_name}")
                break
                
            if not msg.text:
                continue
                
            try:
                lang = detect_language(msg.text)
                raw_text = msg.text[:10000]  # Уменьшил длину текста
                text = preprocess_text(raw_text)
                
                if lang == 'ru':
                    inputs = ru_tokenizer(
                        text, 
                        return_tensors='pt', 
                        truncation=True, 
                        padding=True,
                        max_length=256  # Уменьшил длину
                    )
                    with torch.no_grad():
                        outputs = ru_model(**inputs)
                        probs = torch.nn.functional.softmax(outputs.logits, dim=-1).squeeze().tolist()
                    
                    if ru_model.config.id2label[0] == "negative":
                        neg, neu, pos = probs
                    else:
                        label_to_index = {'neg': None, 'neu': None, 'pos': None}
                        for i in range(3):
                            label = ru_model.config.id2label[i].lower()
                            if 'neg' in label:
                                label_to_index['neg'] = i
                            elif 'pos' in label:
                                label_to_index['pos'] = i
                            else:
                                label_to_index['neu'] = i
                        neg = probs[label_to_index['neg']]
                        neu = probs[label_to_index['neu']]
                        pos = probs[label_to_index['pos']]
                    
                    neg_percent = neg * 100
                    neu_percent = neu * 100
                    pos_percent = pos * 100
                    overall_score = pos_percent - neg_percent
                    sentiment_category = get_sentiment_category(overall_score, raw_text)
                    
                    # Сбор статистики по пользователям
                    username = 'Неизвестный'
                    if hasattr(msg, 'sender') and msg.sender:
                        username = getattr(msg.sender, 'username', 
                                         getattr(msg.sender, 'first_name', 
                                                str(getattr(msg.sender, 'id', 'Неизвестный'))))

                    user_stats[username]['count'] += 1
                    user_stats[username]['total_score'] += overall_score
                    if sentiment_category == "Позитивно":
                        user_stats[username]['positive'] += 1
                    elif sentiment_category == "Негативно":
                        user_stats[username]['negative'] += 1
                
                    results.append({
                        'Дата': msg.date.strftime('%Y-%m-%d %H:%M:%S'),
                        'Текст': raw_text[:300] + "..." if len(raw_text) > 300 else raw_text,
                        'Полный текст': raw_text,
                        'Обработанный текст': text,
                        'Негативно': round(neg_percent, 1),
                        'Нейтрально': round(neu_percent, 1),
                        'Позитивно': round(pos_percent, 1),
                        'Общий балл': round(overall_score, 3),
                        'Категория': sentiment_category,
                        'Язык': lang,
                        'Пользователь': username,
                        'Канал': channel_name
                    })
                    
                    # Периодический вывод прогресса
                    if i % 50 == 0:
                        print(f"Обработано {i} сообщений из канала {channel_name}")
                        
            except Exception as e:
                print(f"Ошибка при обработке сообщения: {str(e)}")
                continue
                
    except Exception as e:
        print(f"Ошибка при анализе канала {channel_url}: {str(e)}")
    
    return results, conflict_cases, user_stats

async def analyze_messages_async():
    all_results = []
    all_conflicts = []
    all_user_stats = defaultdict(lambda: {'count': 0, 'total_score': 0, 'positive': 0, 'negative': 0})
    
    try:
        async with TelegramClient('session', api_id, api_hash) as client:
            await client.start(phone)
            
            for channel in channels:
                if flag.exit():
                    break
                    
                print(f"\nАнализирую канал: {channel}")
                start_time = time.time()
                
                results, conflicts, user_stats = await analyze_channel_async(client, channel)
                all_results.extend(results)
                all_conflicts.extend(conflicts)
                
                for user, stats in user_stats.items():
                    for key in ['count', 'total_score', 'positive', 'negative']:
                        all_user_stats[user][key] += stats[key]
                
                print(f"Канал {channel} обработан за {time.time()-start_time:.1f} сек")
                
    except Exception as e:
        print(f"Критическая ошибка: {str(e)}")
    
    return all_results, all_user_stats

def analyze_messages():
    try:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(analyze_messages_async())
    except Exception as e:
        print(f"Ошибка в главной функции: {str(e)}")
        return [], defaultdict(lambda: {'count': 0, 'total_score': 0, 'positive': 0, 'negative': 0})

# Остальные функции (create_interactive_plot, print_top_messages и т.д.) остаются без изменений

def main():
    try:
        query = input("Поисковый запрос (оставьте пустым для всех сообщений): ").strip().lower()

        print("Анализ сообщений, пожалуйста подождите...")
        start_time = time.time()
        
        messages, user_stats = analyze_messages()
        df = pd.DataFrame(messages)

        if query:
            word_forms = get_word_forms(query)
            pattern = '|'.join(re.escape(wf) for wf in word_forms)
            df = df[df['Полный текст'].str.contains(pattern, case=False, na=False)]

        if len(df) > 0:
            print(f"\nНайдено сообщений: {len(df)}")
            print(f"Время анализа: {time.time()-start_time:.1f} сек")
            
            # Вывод распределения
            print_distribution(df)
            
            # Примеры сообщений
            print("\nПримеры сообщений:")
            for category in ["Негативно", "Нейтрально", "Позитивно"]:
                examples = df[df['Категория'] == category].head(1)
                if not examples.empty:
                    print(f"\n{category}:")
                    print(examples[['Дата', 'Канал', 'Текст', 'Общий балл']].to_string(index=False))
            
            # Топ сообщений
            print_top_messages(df, "Позитивно")
            print_top_messages(df, "Негативно")
            
            # Топ пользователей
            print_top_users(user_stats, "Позитивно")
            print_top_users(user_stats, "Негативно")
            
            # Визуализация
            create_interactive_plot(df, query)

            # Сохранение результатов
            with open('results.json', 'w', encoding='utf-8') as f:
                json.dump(df.to_dict('records'), f, ensure_ascii=False, indent=2)
            
            df[['Дата', 'Канал', 'Текст', 'Категория', 'Общий балл', 'Язык', 'Пользователь']].to_csv(
                'sentiment_report.csv', index=False, encoding='utf-8')
            print("\nРезультаты сохранены в файлы: results.json и sentiment_report.csv")
        else:
            print("Сообщения не найдены по заданному запросу.")

    except Exception as e:
        print(f"Ошибка в main: {str(e)}")
    finally:
        print("Анализ завершен.")

if __name__ == "__main__":
    main()