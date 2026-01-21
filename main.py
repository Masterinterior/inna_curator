import os
import re
import time
import json
import base64
import asyncio
import math
from typing import Dict, List, Any, Optional, Tuple

import requests
from fastapi import FastAPI, Request

# ================= ENV =================
TG_BOT_TOKEN = os.getenv("TG_BOT_TOKEN", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

HTTP_TIMEOUT = 25
MAX_IMAGE_BYTES = 6_000_000

# Knowledge base files
KB_PATH = "knowledge/knowledge.txt"
KB_EMB_PATH = "knowledge/embeddings.json"  # prebuilt semantic index

# ================= APP =================
app = FastAPI()

# ================= DAILY MESSAGE LIMIT =================
DAILY_LIMIT = 70  # ответов Инны в сутки на один чат
DAILY_COUNTER: Dict[int, Dict[str, int]] = {}

def _today_key() -> str:
    return time.strftime("%Y-%m-%d", time.localtime())

def can_reply_today(chat_id: int) -> Tuple[bool, int]:
    day = _today_key()
    rec = DAILY_COUNTER.get(chat_id)
    if not rec or rec.get("day") != day:
        DAILY_COUNTER[chat_id] = {"day": day, "count": 0}
        rec = DAILY_COUNTER[chat_id]
    remaining = max(0, DAILY_LIMIT - int(rec.get("count", 0)))
    return (remaining > 0, remaining)

def inc_today(chat_id: int):
    day = _today_key()
    rec = DAILY_COUNTER.get(chat_id)
    if not rec or rec.get("day") != day:
        DAILY_COUNTER[chat_id] = {"day": day, "count": 0}
        rec = DAILY_COUNTER[chat_id]
    rec["count"] = int(rec.get("count", 0)) + 1

# ================= MEMORY =================
CONTEXT_LIMIT = 14  # keep last 12-14 messages
CHAT_CONTEXT: Dict[int, List[Dict[str, Any]]] = {}

# recent assistant outputs (to reduce repetition)
RECENT_ASSISTANT: Dict[int, List[str]] = {}
RECENT_LIMIT = 6

# ======== IMAGE HISTORY ========
IMAGE_KEEP = 12  # store more now because albums
IMAGE_HISTORY: Dict[int, List[Dict[str, Any]]] = {}
IMAGE_SEQ: Dict[int, int] = {}  # photo counter: #1, #2, ...

# ======== ALBUM BUFFER (Variant 2) ========
ALBUM_DEBOUNCE_SEC = 2.0
ALBUM_BUFFER: Dict[Tuple[int, str], Dict[str, Any]] = {}  # (chat_id, album_id) -> data

# legacy last image store (optional)
LAST_IMAGE: Dict[int, bytes] = {}
LAST_IMAGE_AT: Dict[int, float] = {}
IMAGE_TTL = 60 * 30  # 30 minutes

def has_fresh_image(chat_id: int) -> bool:
    if chat_id not in LAST_IMAGE:
        return False
    return (time.time() - LAST_IMAGE_AT.get(chat_id, 0)) <= IMAGE_TTL

# ================= SYSTEM ROLE =================
SYSTEM_ROLE = (
    "Ты — Инна, профессиональный дизайнер интерьера и куратор обучения конкретной школы. "
    "У тебя есть база знаний по урокам и ты всегда опираешься на неё, если вопрос про обучение. "
    "Ты уверенная, эмпатичная женщина. "
    "Ты говоришь спокойно, по делу, без заумных терминов, но профессионально. "
    "ВАЖНО: Пиши без Markdown (**звёздочек**). Если нужно выделение — используй HTML-теги <b> и <i>. "
    "Если пользователь присылает изображение без вопроса — "
    "ты даёшь короткий тёплый комментарий (2–3 предложения), "
    "отмечаешь настроение и сильные стороны "
    "и мягко предлагаешь, чем можешь помочь дальше. "
    "Если пользователь задаёт вопрос — отвечай развёрнуто и уверенно. "
    "Если работаешь по изображению — всегда опирайся на визуальный контекст. "
    "Если вопрос про обучение и уроки — НЕ пиши общие фразы «зависит от курса» и т.п., "
    "а сразу давай точные места из нашей программы. "
    "Часто заканчивай ответ приглашением продолжить: «Если хочешь — могу…». "
    "Используй эмоджи уместно."
)

# ================= TOPIC GUARD (FORBIDDEN TOPICS) =================
ALLOWED_TOPIC_RE = re.compile(
    r"(дизайн|интерьер|ремонт|отделк|планировк|перепланировк|зонирован|эргономик|"
    r"мебел|кухн|ванн|сануз|спальн|гостин|детск|прихож|"
    r"свет|освещен|электрик|сантех|прораб|подрядчик|строител|"
    r"материал|фактур|плитк|краск|обои|паркет|ламинат|"
    r"стил|мид|mid|мемфис|memphis|лофт|сканди|джапанди|"
    r"обучен|курс|урок|модул|ступен|дз|домашн|"
    r"homestyler|remplanner|archicad|3ds|max|photoshop|ps|canva|figma|"
    r"нейросет|ai|ии|midjourney|stable|prompt|промт|рендер|визуализац|"
    r"соцсет|контент|сторис|рилс|ютуб|вк|телеграм|продвижен|личн(ый|ого)\s+бренд|"
    r"цен(а|ы)|прайс|стоимост|коммерческ|кп|продаж|клиент|"
    r"договор|оферт|счет|акт|предоплат|оплат|аванс|прав(о|а)|юридическ|ип|ооо|усн|ндс)",
    re.IGNORECASE,
)

FORBIDDEN_TOPIC_RE = re.compile(
    r"(физик|квант|относител|формул|интеграл|дифференц|математ|"
    r"медицин|болезн|симптом|диагноз|таблетк|лекарств|анализ(ы)?|"
    r"политик|выбор|парт(ия|ии)|санкц|"
    r"инвестиц|акци|портфел|облигац|крипт|биткоин|курс\s+валют|"
    r"эзотерик|астрол|таро|"
    r"подар(ок|ки)|что\s+подарить|идеи\s+подарков?|"
    r"дет(ям|ей)|ребен(ок|ка|ку)|школ|садик|игрушк|"
    r"праздник|день\s+рождения|н(овый|года)|рождеств|8\s+марта|23\s+февраля|"
    r"быт|лайфстайл|рецепт|еда|готов(ить|ка)|"
    r"отношен(ия|ий)|муж|жена|девушк|парень|любовь|"
    r"расстат|развод|измен(а|ы)|ревност|"
    r"психолог|психотерап|чувств|обид|токсичн|"
    r"18\+|порно|эротик)",
    re.IGNORECASE,
)

OFFTOP_REPLY = (
    "Я могу помочь по дизайну интерьера и ремонту, обучению и программам дизайнера, "
    "дизайнерскому софту, AI для дизайна, личному бренду/контенту, ценам и договорам. "
    "Сформулируй вопрос в этих рамках — и я помогу 🙂"
)

def is_forbidden_topic(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    if FORBIDDEN_TOPIC_RE.search(t) and not ALLOWED_TOPIC_RE.search(t):
        return True
    return False

# ================= TELEGRAM TEXT SANITIZE =================
MD_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
MD_ITALIC_RE = re.compile(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)")

def to_tg_html(text: str) -> str:
    """
    Telegram uses parse_mode=HTML.
    Convert basic Markdown (**bold**, *italic*) to HTML to avoid showing stars.
    Keeps existing HTML tags (KB already uses <b>).
    """
    t = (text or "").strip()
    if not t:
        return t
    t = MD_BOLD_RE.sub(r"<b>\1</b>", t)
    t = MD_ITALIC_RE.sub(r"<i>\1</i>", t)
    t = t.replace("**", "")
    return t

# ================= TELEGRAM =================
def tg_send(chat_id: int, text: str):
    try:
        safe = to_tg_html(text)
        r = requests.post(
            f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage",
            json={
                "chat_id": chat_id,
                "text": safe,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            },
            timeout=HTTP_TIMEOUT,
        )
        if r.status_code != 200:
            print("TG sendMessage error:", r.status_code, r.text)
    except Exception as e:
        print("TG sendMessage exception:", repr(e))

def tg_typing(chat_id: int):
    try:
        requests.post(
            f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendChatAction",
            json={"chat_id": chat_id, "action": "typing"},
            timeout=HTTP_TIMEOUT,
        )
    except Exception as e:
        print("TG typing exception:", repr(e))

def tg_get_photo(file_id: str) -> Optional[bytes]:
    try:
        meta = requests.get(
            f"https://api.telegram.org/bot{TG_BOT_TOKEN}/getFile",
            params={"file_id": file_id},
            timeout=HTTP_TIMEOUT,
        ).json()

        if not meta.get("ok"):
            return None

        path = meta["result"]["file_path"]
        img = requests.get(
            f"https://api.telegram.org/file/bot{TG_BOT_TOKEN}/{path}",
            timeout=HTTP_TIMEOUT,
        ).content

        if img and len(img) <= MAX_IMAGE_BYTES:
            return img
        return None
    except Exception as e:
        print("TG getPhoto exception:", repr(e))
        return None

# ================= OPENAI =================
def openai_chat(messages: List[Dict[str, Any]], max_tokens: int = 900, temperature: float = 0.45) -> str:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)

        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return (r.choices[0].message.content or "").strip()
    except Exception as e:
        print("OPENAI chat exception:", repr(e))
        return "Сейчас у меня небольшая техническая пауза 🙏 Попробуй ещё раз через минуту."

def openai_with_image(
    prompt: str,
    image: bytes,
    context: List[Dict[str, Any]],
    max_tokens: int = 900,
    temperature: float = 0.55,
) -> str:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)

        b64 = base64.b64encode(image).decode()

        messages = (
            [{"role": "system", "content": SYSTEM_ROLE}]
            + context
            + [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ],
            }]
        )

        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return (r.choices[0].message.content or "").strip()
    except Exception as e:
        print("OPENAI image exception:", repr(e))
        return "Я вижу, что пришло фото, но сейчас не могу его разобрать из-за тех. паузы 🙏 Попробуй ещё раз."

def openai_embed(text: str) -> Optional[List[float]]:
    """
    Creates embedding for query (fast & cheap).
    """
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        r = client.embeddings.create(
            model="text-embedding-3-small",
            input=[text],
        )
        return r.data[0].embedding
    except Exception as e:
        print("OPENAI embed exception:", repr(e))
        return None

# ================= CONTEXT =================
def add_context(chat_id: int, role: str, content: str):
    CHAT_CONTEXT.setdefault(chat_id, [])
    CHAT_CONTEXT[chat_id].append({"role": role, "content": content})
    CHAT_CONTEXT[chat_id] = CHAT_CONTEXT[chat_id][-CONTEXT_LIMIT:]

def remember_assistant(chat_id: int, text: str):
    RECENT_ASSISTANT.setdefault(chat_id, [])
    RECENT_ASSISTANT[chat_id].append((text or "")[:900])
    RECENT_ASSISTANT[chat_id] = RECENT_ASSISTANT[chat_id][-RECENT_LIMIT:]

def avoid_repetition_hint(chat_id: int) -> str:
    recent = RECENT_ASSISTANT.get(chat_id) or []
    if not recent:
        return ""
    last = "\n---\n".join(recent[-3:])
    return (
        "ВАЖНО: не повторяй дословно предыдущие ответы и избегай штампов "
        "(«уютно и стильно», «теплая палитра», «индивидуальность», «если хочешь — могу…»). "
        "Меняй лексику и фокус: композиция/свет/цвет/функция/материалы.\n"
        f"НЕ ПОВТОРЯЙ формулировки из последних ответов:\n{last}"
    )

# ======== IMAGE HISTORY HELPERS ========
IMAGE_REF_RE = re.compile(
    r"(на\s+фото|на\s+картинке|по\s+фото|по\s+картинке|посмотри|оцен(и|ка)|"
    r"что\s+не\s+так|что\s+исправить|переделай|вариант|планировк|"
    r"в\s+этом\s+интерьере|здесь|тут|это\s+фото|это\s+изображение|"
    r"предыдущ(ее|ем)\s+фото|прошл(ое|ом)\s+фото|перв(ое|ом)\s+фото|втор(ое|ом)\s+фото|треть(е|ем)\s+фото)",
    re.IGNORECASE,
)
ORDINAL_RE = re.compile(r"\b(перв|втор|трет|четвер|пят)\w*\b", re.IGNORECASE)

VISUAL_TOPIC_RE = re.compile(
    r"(интерьер|комнат|помещен|планировк|стил|цвет|палитр|свет|освещен|"
    r"диван|кресл|кухн|ванн|сануз|спальн|гостин|детск|прихож|"
    r"слишком|мужск|женск|девчач|уютн|холодн|тепл|дешев|дорог|"
    r"что\s+добавить|что\s+убрать|как\s+улучшить|как\s+исправить)",
    re.IGNORECASE,
)

COMPARE_RE = re.compile(
    r"(какой\s+вариант|что\s+лучше|сравни|левый|правый|1\s+или\s+2|первый\s+или\s+второй|выбери\s+лучший|какой\s+нравится)",
    re.IGNORECASE,
)
PHOTO_NUM_RE = re.compile(r"#\s*(\d+)")

def push_image(chat_id: int, img: bytes, desc: str, album_id: Optional[str] = None) -> int:
    IMAGE_SEQ[chat_id] = IMAGE_SEQ.get(chat_id, 0) + 1
    num = IMAGE_SEQ[chat_id]

    IMAGE_HISTORY.setdefault(chat_id, [])
    IMAGE_HISTORY[chat_id].append({
        "num": num,
        "ts": time.time(),
        "image": img,
        "desc": (desc or "").strip(),
        "album_id": album_id,
    })
    IMAGE_HISTORY[chat_id] = IMAGE_HISTORY[chat_id][-IMAGE_KEEP:]
    return num

def pick_image_from_history(chat_id: int, user_text: str) -> Optional[Dict[str, Any]]:
    hist = IMAGE_HISTORY.get(chat_id) or []
    if not hist:
        return None
    t = (user_text or "").lower()

    if "предыдущ" in t or "прошл" in t:
        return hist[-2] if len(hist) >= 2 else hist[-1]

    nums = [int(x) for x in PHOTO_NUM_RE.findall(user_text)]
    if nums:
        target = nums[0]
        for it in hist:
            if it.get("num") == target:
                return it

    if ORDINAL_RE.search(t):
        if "перв" in t:
            target = 1
        elif "втор" in t:
            target = 2
        elif "трет" in t:
            target = 3
        elif "четвер" in t:
            target = 4
        elif "пят" in t:
            target = 5
        else:
            target = None
        if target is not None:
            for it in hist:
                if it.get("num") == target:
                    return it
            return hist[-1]

    return hist[-1]

def build_visual_context_messages(chat_id: int, limit: int = 4) -> List[Dict[str, Any]]:
    hist = IMAGE_HISTORY.get(chat_id) or []
    if not hist:
        return []
    tail = hist[-limit:]
    parts = []
    for it in tail:
        num = it.get("num", "?")
        desc = (it.get("desc") or "").strip()
        if desc:
            parts.append(f"Фото #{num}: {desc}")
    if not parts:
        return []
    joined = "\n\n".join(parts).strip()
    return [{"role": "assistant", "content": f"Визуальные заметки по последним фото:\n{joined}"}]

def get_context(chat_id: int) -> List[Dict[str, Any]]:
    ctx = CHAT_CONTEXT.get(chat_id, []).copy()
    return build_visual_context_messages(chat_id, limit=4) + ctx

# ================= INTENT: KB LINKS VS HOW-TO =================
COURSE_LOCATOR_RE = re.compile(
    r"(в\s+каком\s+уроке|какой\s+урок|где\s+в\s+курсе|где\s+это\s+в\s+обучении|"
    r"где\s+посмотрет|в\s+каком\s+модул|в\s+какой\s+ступен|"
    r"лежит\s+в\s+программе|открыть\s+урок|найти\s+урок)",
    re.IGNORECASE,
)

HOWTO_RE = re.compile(
    r"(как\s+сделать|как\s+собрать|как\s+создать|как\s+настроить|"
    r"объясни|покажи|инструкц|пошагово|мне\s+нужно|хочу\s+сделать|"
    r"не\s+понял|не\s+поняла|не\s+получается|ошибка|почему)",
    re.IGNORECASE,
)

LIST_LESSONS_RE = re.compile(
    r"(перечисли|список|все)\s+(уроки|уроков)\b",
    re.IGNORECASE,
)
MODULE_NUM_RE = re.compile(r"\bмодул[ьяею]\s*(\d+)\b", re.IGNORECASE)

def normalize(s: str) -> str:
    s = (s or "").lower().strip().replace("ё", "е")
    s = re.sub(r"[\"'“”«»]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s

def should_show_kb_links(text: str) -> bool:
    if not text:
        return False

    t = normalize(text)

    has_lesson_word = ("урок" in t)
    has_locator_words = any(w in t for w in [
        "где", "в каком", "какой", "лежит", "найти", "открыть", "посмотреть",
        "в курсе", "в обучении", "в программе", "в модуле", "в ступени",
        "модуль", "ступень", "раздел"
    ])

    if has_lesson_word and has_locator_words:
        return True

    return bool(COURSE_LOCATOR_RE.search(text))

def is_howto(text: str) -> bool:
    return bool(text and HOWTO_RE.search(text))

def wants_list_lessons(text: str) -> bool:
    return bool(text and LIST_LESSONS_RE.search(text))

def extract_module_num(text: str) -> Optional[int]:
    if not text:
        return None
    m = MODULE_NUM_RE.search(text)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None

# ================= KB (parse + semantic retrieval + LLM rerank) =================
KB_INDEX: List[Dict[str, Any]] = []
KB_EMB_VECS: List[List[float]] = []

COURSE_RE = re.compile(r"^\s*СТРУКТУРА\s+КУРСА", re.IGNORECASE)
STEP_RE = re.compile(r"^\s*\d+\s*ступен", re.IGNORECASE)
MODULE_RE = re.compile(r"^\s*\d+\s*модул", re.IGNORECASE)
LESSON_RE = re.compile(r"^\s*\d+\s*урок\b", re.IGNORECASE)

URL_LINE_RE = re.compile(r"(https?://\S+)", re.IGNORECASE)
LESSON_URL_LINE_RE = re.compile(r"(?:Ссылка\s+на\s+урок|Ссылка)\s*:\s*(https?://\S+)", re.IGNORECASE)
DZ_RE = re.compile(r"^\s*ДЗ(?:\s*\([^)]+\))?\s*:\s*(.+)$", re.IGNORECASE)

SECTION_FULL_RE = re.compile(
    r"^Раздел\s*[«\"\']?([^»\"\'\:]+)[»\"\']?\s*:\s*(.+)$",
    re.IGNORECASE
)

def split_materials(s: str) -> List[str]:
    parts = [p.strip() for p in (s or "").split(",")]
    parts = [p.strip(" .;") for p in parts if p.strip()]
    return parts

def load_kb() -> Tuple[int, str]:
    global KB_INDEX
    if not os.path.exists(KB_PATH):
        KB_INDEX = []
        return 0, f"KB not found: {KB_PATH}"

    lines = [ln.rstrip() for ln in open(KB_PATH, "r", encoding="utf-8", errors="ignore").read().splitlines()]
    if not lines:
        KB_INDEX = []
        return 0, "KB empty"

    course_title, course_url = "", ""
    step_title, step_url = "", ""
    module_title, module_url = "", ""

    KB_INDEX = []
    i = 0
    while i < len(lines):
        ln = (lines[i] or "").strip()
        if not ln:
            i += 1
            continue

        if COURSE_RE.search(ln):
            course_title = ln
            course_url = ""
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines):
                m = URL_LINE_RE.search(lines[j].strip())
                if m:
                    course_url = m.group(1).rstrip(").,;")
                    i = j
            i += 1
            continue

        if STEP_RE.search(ln):
            step_title = ln
            step_url = ""
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines):
                m = URL_LINE_RE.search(lines[j].strip())
                if m:
                    step_url = m.group(1).rstrip(").,;")
                    i = j
            module_title, module_url = "", ""
            i += 1
            continue

        if MODULE_RE.search(ln):
            module_title = ln
            module_url = ""
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines):
                m = URL_LINE_RE.search(lines[j].strip())
                if m:
                    module_url = m.group(1).rstrip(").,;")
                    i = j
            i += 1
            continue

        if LESSON_RE.search(ln):
            lesson_title = ln
            lesson_url = ""
            sections: List[str] = []
            homework = ""

            j = i + 1
            while j < len(lines):
                cur = (lines[j] or "").strip()
                if not cur:
                    j += 1
                    continue
                if COURSE_RE.search(cur) or STEP_RE.search(cur) or MODULE_RE.search(cur) or LESSON_RE.search(cur):
                    break

                mlu = LESSON_URL_LINE_RE.search(cur)
                if mlu:
                    lesson_url = mlu.group(1).rstrip(").,;")
                else:
                    mdz = DZ_RE.search(cur)
                    if mdz:
                        homework = mdz.group(1).strip()
                    else:
                        sections.append(cur)

                j += 1

            lesson_blob = "\n".join(sections).strip()

            section_items: List[Tuple[str, List[str]]] = []
            for s in sections:
                msec = SECTION_FULL_RE.match(s.strip())
                if not msec:
                    continue
                sec_title = msec.group(1).strip()
                materials_raw = msec.group(2).strip()
                mats = split_materials(materials_raw) if materials_raw else []
                section_items.append((sec_title, mats))

            if not section_items:
                section_items = [("", ["(материал не указан)"])]

            for sec_title, mats in section_items:
                if not mats:
                    mats = ["(материал не указан)"]

                for mat in mats:
                    KB_INDEX.append({
                        "type": "micro",
                        "kind": "Видеоурок",
                        "course_title": course_title,
                        "course_url": course_url,
                        "step_title": step_title,
                        "step_url": step_url,
                        "module_title": module_title,
                        "module_url": module_url,
                        "lesson_title": lesson_title,
                        "lesson_url": lesson_url,
                        "section_title": sec_title,
                        "material_title": mat,
                        "homework": homework,
                        "lesson_blob": lesson_blob,
                        "text": normalize(" ".join([
                            course_title, step_title, module_title, lesson_title,
                            sec_title, mat, lesson_blob, homework
                        ])),
                    })

            i = j
            continue

        i += 1

    return len(KB_INDEX), "OK"

def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return -1.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for i in range(len(a)):
        x = float(a[i])
        y = float(b[i])
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0 or nb <= 0:
        return -1.0
    return dot / (math.sqrt(na) * math.sqrt(nb))

def load_embeddings() -> Tuple[int, str]:
    global KB_EMB_VECS
    KB_EMB_VECS = []

    if not os.path.exists(KB_EMB_PATH):
        return 0, f"Embeddings not found: {KB_EMB_PATH}"

    try:
        data = json.loads(open(KB_EMB_PATH, "r", encoding="utf-8", errors="ignore").read())
    except Exception as e:
        return 0, f"Embeddings JSON read error: {repr(e)}"

    items = None
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        if isinstance(data.get("items"), list):
            items = data["items"]
        elif isinstance(data.get("data"), list):
            items = data["data"]
        elif isinstance(data.get("embeddings"), list):
            items = data["embeddings"]

    if not isinstance(items, list):
        return 0, "Embeddings JSON format not recognized"

    vecs: List[List[float]] = []
    for it in items:
        if isinstance(it, dict):
            emb = it.get("embedding")
            if isinstance(emb, list) and emb and isinstance(emb[0], (int, float)):
                vecs.append([float(x) for x in emb])

    if not vecs:
        return 0, "No vectors found in embeddings.json"

    if KB_INDEX and len(vecs) >= len(KB_INDEX):
        KB_EMB_VECS = vecs[:len(KB_INDEX)]
        return len(KB_EMB_VECS), "OK (aligned)"
    else:
        KB_EMB_VECS = vecs
        return len(KB_EMB_VECS), "OK (unaligned)"

def _expand_query_for_semantic(q: str) -> str:
    t = normalize(q)
    add: List[str] = []

    if "мид" in t or "mid" in t:
        add += ["мид-сенчури", "mid-century", "mid century", "midcentury", "мидсенчури"]
    if "эко" in t or "eco" in t:
        add += ["эко-стиль", "eco style", "экостиль"]
    if "средизем" in t or "mediterr" in t:
        add += ["средиземноморский", "mediterranean"]
    if "мемфис" in t or "memphis" in t:
        add += ["мемфис", "memphis"]

    if "ванн" in t or "bath" in t:
        add += ["санузел", "санузлы", "туалет", "bathroom", "wc"]
    if "сануз" in t or "туалет" in t or "wc" in t:
        add += ["ванная", "ванна", "bathroom"]

    if "фотошоп" in t or "photoshop" in t or "ps" in t:
        add += ["adobe photoshop", "фш", "psd"]
    if "3д" in t or "3d" in t:
        add += ["3d", "3д", "3д коллаж", "3d collage", "коллаж", "moodboard", "мудборд"]

    if add:
        return (q + "\n\nСинонимы/переводы: " + ", ".join(add)).strip()
    return q

def kb_candidates_semantic(query: str, k: int = 20) -> List[Dict[str, Any]]:
    if not query or not KB_INDEX or not KB_EMB_VECS:
        return []

    q2 = _expand_query_for_semantic(query)
    qvec = openai_embed(q2)
    if not qvec:
        return []

    n = min(len(KB_INDEX), len(KB_EMB_VECS))
    if n <= 0:
        return []

    scored: List[Tuple[float, int]] = []
    for i in range(n):
        sim = _cosine(qvec, KB_EMB_VECS[i])
        if sim > -0.5:
            scored.append((sim, i))

    scored.sort(key=lambda x: x[0], reverse=True)

    out: List[Dict[str, Any]] = []
    seen = set()
    for sim, idx in scored[: max(k * 4, 40)]:
        it = KB_INDEX[idx]
        key = (it.get("lesson_url"), it.get("section_title"), it.get("material_title"))
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
        if len(out) >= k:
            break
    return out

def kb_candidates_keyword(query: str, k: int = 20) -> List[Dict[str, Any]]:
    if not query or not KB_INDEX:
        return []

    q = normalize(query)

    expansions: List[str] = []
    if "ванн" in q:
        expansions += ["санузел", "санузлы", "туалет", "bathroom", "wc"]
    if "сануз" in q or "туалет" in q or "wc" in q:
        expansions += ["ванная", "ванна", "bathroom"]

    if "3d" in q or "3д" in q:
        expansions += ["3д", "3d", "коллаж", "3д коллаж", "3d collage", "мудборд", "moodboard"]
    if "коллаж" in q or "moodboard" in q:
        expansions += ["3д", "3d", "3д коллаж", "3d collage"]

    if "photoshop" in q or "фотошоп" in q:
        expansions += ["ps", "adobe photoshop"]

    if "мид" in q or "mid" in q:
        expansions += ["мид-сенчури", "mid-century", "mid century", "мидсенчури"]
    if "эко" in q or "eco" in q:
        expansions += ["эко-стиль", "eco style", "экостиль"]

    terms = [w for w in re.findall(r"[a-zа-я0-9]+", q) if len(w) >= 3]
    for e in expansions:
        terms += [w for w in re.findall(r"[a-zа-я0-9]+", normalize(e)) if len(w) >= 3]
    terms = list(dict.fromkeys(terms))

    need_bath = ("ванн" in q) or ("сануз" in q) or ("туалет" in q) or ("bath" in q) or ("wc" in q)
    bath_terms = ["сануз", "ванн", "туалет", "bath", "wc"]

    scored: List[Tuple[float, Dict[str, Any]]] = []
    for it in KB_INDEX:
        t = it.get("text", "")
        if not t:
            continue

        score = 0.0
        mt = normalize(it.get("material_title", ""))
        lb = normalize(it.get("lesson_blob", ""))
        hw = normalize(it.get("homework", ""))

        for w in terms:
            if w in t:
                score += 1.0
            if w in mt:
                score += 1.5
            if w in lb:
                score += 1.2
            if w in hw:
                score += 2.6

        if need_bath and any(bt in t for bt in bath_terms):
            score += 3.0

        if score > 0:
            scored.append((score, it))

    scored.sort(key=lambda x: x[0], reverse=True)

    uniq: List[Dict[str, Any]] = []
    seen = set()
    for _, it in scored:
        key = (it.get("lesson_url"), it.get("section_title"), it.get("material_title"))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(it)
        if len(uniq) >= k:
            break

    return uniq

def kb_candidates(query: str, k: int = 20) -> List[Dict[str, Any]]:
    sem = kb_candidates_semantic(query, k=k)
    if sem:
        return sem
    return kb_candidates_keyword(query, k=k)

def kb_select_with_llm(user_query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not candidates:
        return []

    packed = []
    for idx, it in enumerate(candidates[:20], 1):
        packed.append({
            "id": idx,
            "step": it.get("step_title", ""),
            "module": it.get("module_title", ""),
            "lesson": it.get("lesson_title", ""),
            "section": it.get("section_title", ""),
            "material": it.get("material_title", ""),
            "url": it.get("lesson_url", ""),
            "homework": it.get("homework", ""),
            "blob": (it.get("lesson_blob", "")[:550] if it.get("lesson_blob") else ""),
        })

    selector_system = (
        "Ты — методист и куратор курса дизайна интерьера.\n"
        "Тебе дали вопрос студента и список кандидатов из базы знаний.\n"
        "Задача: выбрать до 3 самых релевантных кандидатов ПО СМЫСЛУ.\n"
        "Учитывай синонимы и переводы RU<->EN (mid-century = мид-сенчури), формулировки в ДЗ, цель урока.\n"
        "Если в базе НЕТ ничего подходящего — верни NONE.\n"
        "Верни ТОЛЬКО JSON без текста вокруг:\n"
        "{\"pick\":[1,5],\"reason\":\"...\"} или {\"pick\":[],\"reason\":\"NONE\"}\n"
    )

    raw = openai_chat(
        [
            {"role": "system", "content": selector_system},
            {"role": "user", "content": f"Вопрос студента:\n{user_query}\n\nКандидаты:\n{json.dumps(packed, ensure_ascii=False)}"},
        ],
        max_tokens=350,
        temperature=0.2,
    )

    try:
        data = json.loads(raw)
        picks = data.get("pick", [])
        if not picks:
            return []
        chosen = []
        for p in picks[:3]:
            if isinstance(p, int) and 1 <= p <= len(candidates[:20]):
                chosen.append(candidates[p - 1])
        return chosen
    except Exception:
        return candidates[:1]

def best_material_name(it: Dict[str, Any], user_query: str) -> str:
    q = normalize(user_query)
    hw = (it.get("homework") or "").strip()
    mat = (it.get("material_title") or "").strip()

    if ("коллаж" in q or "3д" in q or "3d" in q) and hw:
        return hw if len(hw) <= 240 else (hw[:240].rstrip() + "…")

    return mat or "(материал)"

def dedupe_hits_by_lesson(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not hits:
        return []
    out: List[Dict[str, Any]] = []
    seen = set()
    for it in hits:
        key = (it.get("lesson_url") or "").strip() or (it.get("lesson_title") or "").strip()
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out

def format_kb_hits(hits: List[Dict[str, Any]], user_query: str) -> str:
    if not hits:
        return ""

    hits = dedupe_hits_by_lesson(hits)

    out = ["\n\n📚 <b>Нашла в обучении:</b>"]
    for it in hits[:3]:
        out.append("\n📌 <b>Материал:</b> Видеоурок")

        out.append("\n<b>Где лежит в программе:</b>")
        if it.get("step_title"):
            out.append(f"— <b>Ступень:</b> {it.get('step_title')}")
        if it.get("module_title"):
            out.append(f"— <b>Модуль:</b> {it.get('module_title')}")
        if it.get("lesson_title"):
            out.append(f"— <b>Урок:</b> {it.get('lesson_title')}")
        if it.get("section_title"):
            out.append(f"— <b>Раздел:</b> {it.get('section_title')}")

        out.append(f"\n<b>Название материала:</b> {best_material_name(it, user_query)}")

        url = it.get("lesson_url", "")
        if url:
            out.append(f"\n<b>Ссылка:</b>\n{url}")

        hw = (it.get("homework") or "").strip()
        if hw:
            out.append("\n<b>Домашнее задание:</b>")
            out.append(f"{hw}")

    return "\n".join(out).strip()

def format_module_lessons(module_num: int) -> str:
    if not KB_INDEX:
        return "База знаний пока не загружена 🙏"

    target = str(module_num)
    module_hits: List[Dict[str, Any]] = []
    for it in KB_INDEX:
        mt = (it.get("module_title") or "").strip().lower().replace("ё", "е")
        if re.search(rf"(^|\s){re.escape(target)}\s*модул", mt):
            module_hits.append(it)

    if not module_hits:
        return f"Не нашла модуль {module_num} в базе 🙏 Если скажешь точное название модуля — найду точнее."

    lessons: Dict[str, Dict[str, str]] = {}
    for it in module_hits:
        lt = (it.get("lesson_title") or "").strip()
        url = (it.get("lesson_url") or "").strip()
        if not lt:
            continue
        if lt not in lessons:
            lessons[lt] = {"url": url}
        if not lessons[lt]["url"] and url:
            lessons[lt]["url"] = url

    if not lessons:
        return f"В модуле {module_num} нашлись записи, но без названий уроков. Проверь формат knowledge.txt 🙏"

    def lesson_sort_key(title: str) -> Tuple[int, str]:
        m = re.search(r"(\d+)\s*урок", title.lower())
        if m:
            return (int(m.group(1)), title)
        return (10**9, title)

    ordered = sorted(lessons.items(), key=lambda kv: lesson_sort_key(kv[0]))

    out: List[str] = []
    out.append(f"📚 <b>Модуль {module_num}: уроки</b>\n")
    for title, meta in ordered:
        url = (meta.get("url") or "").strip()
        if url:
            out.append(f"— <b>{title}</b>\n{url}")
        else:
            out.append(f"— <b>{title}</b>\n(ссылка не указана в базе)")

    out.append("\nЕсли хочешь — уточни тему (например: «3D коллаж ванной»), и я дам точный урок и раздел.")
    return "\n".join(out).strip()

@app.on_event("startup")
def _startup():
    n, msg = load_kb()
    print(f"KB loaded: {n} items, status: {msg}")

    en, emsg = load_embeddings()
    print(f"Embeddings loaded: {en} vectors, status: {emsg}")

# ================= ALBUM PROCESSOR =================
async def _process_album(chat_id: int, album_id: str):
    key = (chat_id, album_id)
    data = ALBUM_BUFFER.get(key)
    if not data:
        return

    await asyncio.sleep(ALBUM_DEBOUNCE_SEC)

    data2 = ALBUM_BUFFER.get(key)
    if not data2 or data2.get("task_id") != data.get("task_id"):
        return

    images: List[bytes] = data2.get("images", [])
    caption: str = (data2.get("caption") or "").strip()

    ALBUM_BUFFER.pop(key, None)

    if not images:
        return

    tg_typing(chat_id)

    nums: List[int] = []
    descs: List[Tuple[int, str]] = []
    for idx, img in enumerate(images, 1):
        describe_prompt = (
            "Опиши, что на изображении, чтобы я могла анализировать это в следующих сообщениях. "
            "Если это интерьер — укажи тип помещения, план, мебель, цвета, свет, стиль. "
            "Если это отдельный предмет — укажи материал, цвет, форму, стиль, фактуру. "
            "Сделай 5–7 предложений, конкретно и без воды."
        )
        desc = openai_with_image(describe_prompt, img, [], max_tokens=320, temperature=0.35)
        num = push_image(chat_id, img, desc, album_id=album_id)
        nums.append(num)
        descs.append((num, desc))

    add_context(chat_id, "user", f"[Пользователь прислал альбом фото: {', '.join([f'#{n}' for n in nums])}]")

    if caption:
        add_context(chat_id, "user", caption)

        if COMPARE_RE.search(caption) and len(descs) >= 2:
            packed = "\n\n".join([f"Фото #{n}:\n{d}" for n, d in descs])
            messages = [
                {"role": "system", "content": SYSTEM_ROLE + "\n" + avoid_repetition_hint(chat_id)},
                {
                    "role": "user",
                    "content": (
                        f"{caption}\n\n"
                        "Ниже описания нескольких вариантов (это один альбом). "
                        "Выбери лучший вариант и объясни:\n"
                        "1) Победитель: фото #...\n"
                        "2) 3 причины\n"
                        "3) Для каждого проигравшего — по 1 точечной правке (коротко)\n"
                        "Без общих фраз.\n\n"
                        f"{packed}"
                    ),
                },
            ]
            answer = openai_chat(messages, max_tokens=650, temperature=0.55)
            remember_assistant(chat_id, answer)
            add_context(chat_id, "assistant", answer)
            tg_send(chat_id, answer)
            return

        packed = "\n\n".join([f"Фото #{n}:\n{d}" for n, d in descs])
        messages = [
            {"role": "system", "content": SYSTEM_ROLE + "\n" + avoid_repetition_hint(chat_id)},
            {
                "role": "user",
                "content": (
                    f"{caption}\n\n"
                    "У тебя есть описания всех фото из альбома ниже — используй их как визуальный контекст. "
                    "Отвечай конкретно по запросу, без фразы «я не вижу фото». "
                    "Если вопрос предполагает выбор — выбери и обоснуй.\n\n"
                    f"{packed}"
                ),
            },
        ]
        answer = openai_chat(messages, max_tokens=750, temperature=0.5)
        remember_assistant(chat_id, answer)
        add_context(chat_id, "assistant", answer)
        tg_send(chat_id, answer)
        return

    focus = ["композицию", "свет", "цвет", "функциональность", "материалы и фактуры"]
    f = focus[nums[-1] % len(focus)]
    packed_short = "\n\n".join([f"Фото #{n}: {d[:220].strip()}…" for n, d in descs])
    messages = [
        {"role": "system", "content": SYSTEM_ROLE + "\n" + avoid_repetition_hint(chat_id)},
        {
            "role": "user",
            "content": (
                f"Пользователь прислал альбом из {len(descs)} фото без подписи.\n"
                f"Дай один короткий комментарий (3–5 предложений) с фокусом на {f}. "
                "Затем задай один вопрос, что именно человеку нужно: выбрать лучший вариант, найти ошибки, "
                "или предложить правки. Не используй штампы.\n\n"
                f"Описания фото:\n{packed_short}"
            ),
        },
    ]
    answer = openai_chat(messages, max_tokens=320, temperature=0.6)
    remember_assistant(chat_id, answer)
    add_context(chat_id, "assistant", answer)
    tg_send(chat_id, answer)

def _schedule_album(chat_id: int, album_id: str):
    key = (chat_id, album_id)
    data = ALBUM_BUFFER.get(key)
    if not data:
        return
    data["task_id"] = str(time.time())
    ALBUM_BUFFER[key] = data
    asyncio.create_task(_process_album(chat_id, album_id))

# ================= WEBHOOK =================
@app.post("/webhook")
async def webhook(req: Request):
    update = await req.json()
    msg = update.get("message")
    if not msg:
        return {"ok": True}

    chat_id = msg["chat"]["id"]
    text = (msg.get("text") or "").strip()
    caption = (msg.get("caption") or "").strip()
    photos = msg.get("photo") or []
    album_id = msg.get("media_group_id")

    # ===== PHOTO RECEIVED =====
    if photos:
        img = tg_get_photo(photos[-1]["file_id"])
        if not img:
            return {"ok": True}

        LAST_IMAGE[chat_id] = img
        LAST_IMAGE_AT[chat_id] = time.time()

        # ---- ALBUM MODE (Variant 2): buffer and answer once ----
        if album_id:
            key = (chat_id, str(album_id))
            buf = ALBUM_BUFFER.get(key) or {"images": [], "caption": "", "task_id": ""}
            buf["images"].append(img)
            if caption and not buf.get("caption"):
                buf["caption"] = caption
            ALBUM_BUFFER[key] = buf
            _schedule_album(chat_id, str(album_id))
            return {"ok": True}

        # ---- SINGLE PHOTO (existing behavior) ----
        tg_typing(chat_id)

        describe_prompt = (
            "Опиши, что на изображении, чтобы я могла анализировать это в следующих сообщениях. "
            "Если это интерьер — укажи тип помещения, план, мебель, цвета, свет, стиль. "
            "Если это отдельный предмет — укажи материал, цвет, форму, стиль, фактуру. "
            "Сделай 6–8 предложений, конкретно и без воды."
        )
        visual_description = openai_with_image(describe_prompt, img, [], max_tokens=350, temperature=0.35)
        num = push_image(chat_id, img, visual_description)

        # If caption exists -> answer it with vision
        if caption:
            add_context(chat_id, "user", caption)
            ctx = get_context(chat_id)
            answer = openai_with_image(caption, img, ctx, max_tokens=900, temperature=0.55)
            remember_assistant(chat_id, answer)
            add_context(chat_id, "assistant", answer)
            tg_send(chat_id, answer)
            return {"ok": True}

        # No caption -> auto comment
        add_context(chat_id, "user", f"[Пользователь прислал фото без текста. Фото #{num}]")
        auto_prompt = (
            "Дай короткий тёплый комментарий по тому, что на фото: "
            "2–3 предложения, без штампов; одно сильное наблюдение + один вопрос в конце."
        )
        auto_answer = openai_with_image(auto_prompt, img, [], max_tokens=220, temperature=0.6)
        remember_assistant(chat_id, auto_answer)
        add_context(chat_id, "assistant", auto_answer)
        tg_send(chat_id, auto_answer)
        return {"ok": True}

    # ===== TEXT MESSAGE =====
    if text:
        tg_typing(chat_id)

        # ===== DAILY LIMIT CHECK =====
        ok, remaining = can_reply_today(chat_id)
        if not ok:
            tg_send(
                chat_id,
                "Мы сегодня уже очень много разобрали 💛\n"
                "Я отвечаю подробно, поэтому есть дневной лимит.\n\n"
                "Завтра продолжим — если вопрос срочный, попробуй сформулировать его одним сообщением."
            )
            inc_today(chat_id)
            return {"ok": True}

        # ===== TOPIC GUARD (forbidden topics) =====
        if is_forbidden_topic(text):
            tg_send(chat_id, OFFTOP_REPLY)
            inc_today(chat_id)
            return {"ok": True}

        add_context(chat_id, "user", text)

        # ====== LIST LESSONS FOR MODULE (only if user asked list) ======
        if wants_list_lessons(text):
            mn = extract_module_num(text)
            if mn is not None:
                answer = format_module_lessons(mn)
                remember_assistant(chat_id, answer)
                add_context(chat_id, "assistant", answer)
                tg_send(chat_id, answer)
                inc_today(chat_id)
                return {"ok": True}
            else:
                answer = "Ок 🙂 Напиши, пожалуйста: «все уроки модуля 2» (с номером модуля)."
                remember_assistant(chat_id, answer)
                add_context(chat_id, "assistant", answer)
                tg_send(chat_id, answer)
                inc_today(chat_id)
                return {"ok": True}

        # ====== 🔥 COMPARISON REQUEST (by #numbers or last 2) ======
        if COMPARE_RE.search(text) and len(IMAGE_HISTORY.get(chat_id, [])) >= 2:
            hist = IMAGE_HISTORY.get(chat_id, [])
            nums = [int(x) for x in PHOTO_NUM_RE.findall(text)]

            def get_by_num(n: int) -> Optional[Dict[str, Any]]:
                for it in hist:
                    if it.get("num") == n:
                        return it
                return None

            if len(nums) >= 2:
                a = get_by_num(nums[0])
                b = get_by_num(nums[1])
                if a and b and a.get("image") and b.get("image"):
                    img_a, img_b = a["image"], b["image"]
                    label_a, label_b = f"Фото #{nums[0]}", f"Фото #{nums[1]}"
                else:
                    img_a, img_b = hist[-2]["image"], hist[-1]["image"]
                    label_a, label_b = "Фото A (предпоследнее)", "Фото B (последнее)"
            else:
                img_a, img_b = hist[-2]["image"], hist[-1]["image"]
                label_a, label_b = "Фото A (предпоследнее)", "Фото B (последнее)"

            desc_a = openai_with_image(
                f"Коротко опиши {label_a} для сравнения (3–5 пунктов: композиция, цвет, свет, стиль, функция).",
                img_a, [], max_tokens=220, temperature=0.35
            )
            desc_b = openai_with_image(
                f"Коротко опиши {label_b} для сравнения (3–5 пунктов: композиция, цвет, свет, стиль, функция).",
                img_b, [], max_tokens=220, temperature=0.35
            )

            messages = [
                {"role": "system", "content": SYSTEM_ROLE + "\n" + avoid_repetition_hint(chat_id)},
                {
                    "role": "user",
                    "content": (
                        f"{text}\n\n"
                        f"{label_a} — описание:\n{desc_a}\n\n"
                        f"{label_b} — описание:\n{desc_b}\n\n"
                        "Ответ оформи так:\n"
                        "1) Выбор: ...\n"
                        "2) Почему (3 пункта)\n"
                        "3) Что улучшить в проигравшем (2 пункта)\n"
                        "Без штампов и общих фраз."
                    ),
                },
            ]
            answer = openai_chat(messages, max_tokens=520, temperature=0.55)
            remember_assistant(chat_id, answer)
            add_context(chat_id, "assistant", answer)
            tg_send(chat_id, answer)
            inc_today(chat_id)
            return {"ok": True}

        # ====== KB LINKS ONLY WHEN ASKED "где в курсе / в каком уроке / ссылка на урок" ======
        kb_block = ""
        if should_show_kb_links(text):
            cand = kb_candidates(text, k=20)
            picked = kb_select_with_llm(text, cand)
            kb_block = format_kb_hits(picked, text)

        if kb_block:
            guide = openai_chat(
                [
                    {"role": "system", "content": SYSTEM_ROLE},
                    {
                        "role": "user",
                        "content": (
                            f"Пользователь спросил: {text}\n"
                            f"Я нашла в нашей базе такие материалы:\n{kb_block}\n\n"
                            "Напиши короткую подводку (2–3 предложения) как куратор НАШЕЙ программы: "
                            "без общих фраз типа «зависит от курса/программы». "
                            "Сразу направь человека, что открыть и с чего начать. "
                            "Не повторяй ссылки, они ниже."
                        ),
                    },
                ],
                max_tokens=140,
                temperature=0.25,
            )
            answer = (guide.strip() + kb_block).strip()
            remember_assistant(chat_id, answer)
            add_context(chat_id, "assistant", answer)
            tg_send(chat_id, answer)
            inc_today(chat_id)
            return {"ok": True}

        # ====== OTHERWISE: NORMAL ANSWER (WITH VISUAL CONTEXT IF RELEVANT) ======
        ctx = get_context(chat_id)
        messages = [{"role": "system", "content": SYSTEM_ROLE + "\n" + avoid_repetition_hint(chat_id)}] + ctx

        has_any_photo = bool(IMAGE_HISTORY.get(chat_id))
        looks_like_visual_question = bool(IMAGE_REF_RE.search(text) or VISUAL_TOPIC_RE.search(text))

        if is_howto(text):
            messages = [{"role": "system", "content": SYSTEM_ROLE + "\n" + avoid_repetition_hint(chat_id)}] + ctx + [{
                "role": "user",
                "content": (
                    f"{text}\n\n"
                    "Ответь как практикующий дизайнер: дай пошаговый план. "
                    "Не отправляй человека 'смотреть уроки' и не давай ссылки, если он не спрашивал 'где в курсе'. "
                    "Не используй Markdown (**звёздочки**). "
                    "В конце добавь: «Если хочешь — скажи “где это в курсе”, и я дам точный урок и ссылку.»"
                )
            }]

        if has_any_photo and looks_like_visual_question:
            picked_img = pick_image_from_history(chat_id, text)
            if picked_img and picked_img.get("image"):
                answer = openai_with_image(text, picked_img["image"], ctx, max_tokens=900, temperature=0.55)
            else:
                answer = openai_chat(messages, max_tokens=900, temperature=0.45)
        else:
            answer = openai_chat(messages, max_tokens=900, temperature=0.45)

        remember_assistant(chat_id, answer)
        add_context(chat_id, "assistant", answer)
        tg_send(chat_id, answer)
        inc_today(chat_id)
        return {"ok": True}

    return {"ok": True}
