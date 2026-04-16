# Truman Show Fraud Detector

Мультиагентная система на LangChain для поиска мошеннических транзакций в датасете `The Truman Show - train`.

## Что делает

- загружает `transactions.csv`, `users.json`, `locations.json`, `sms.json`, `mails.json`;
- строит поведенческие признаки по каждому пользователю;
- выделяет подозрительные транзакции эвристиками;
- при наличии ключа OpenRouter прогоняет кандидатов через 3 агентов:
  - `Signals Agent` анализирует фишинг и давление через сообщения;
  - `Network Agent` анализирует новизну получателя и платежный паттерн;
  - `Decision Agent` принимает финальное решение;
- при наличии Langfuse пишет трейсы вызовов.

## Быстрый старт

1. Установить зависимости:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Подготовить `.env`:

```bash
cp .env.example .env
```

3. Запустить:

```bash
python main.py
```

Вернется JSON-массив с `transaction_id`, которые система посчитала fraud.

Полный отчет по кандидатам:

```bash
python main.py --report
```

## Переменные окружения

- `OPENROUTER_API_KEY` — ключ OpenRouter.
- `OPENROUTER_MODEL` — модель через OpenRouter, по умолчанию `openai/gpt-4o-mini`.
- `LANGFUSE_PUBLIC_KEY`
- `LANGFUSE_SECRET_KEY`
- `LANGFUSE_HOST` — по умолчанию `https://challenges.reply.com/langfuse`
- `TEAM_NAME` — используется в trace session id.

## Режимы работы

- Без `OPENROUTER_API_KEY`: детерминированный baseline по эвристикам.
- С `OPENROUTER_API_KEY`: baseline + LangChain multi-agent reasoning.

## Замечание

Так как в train-наборе нет явных меток fraud/non-fraud, текущая система строит shortlist на основе аномалий, фишинговых сигналов и отклонений от обычного поведения. Это хороший каркас для challenge, но пороги и промпты стоит дотюнить после проверки на валидации.
