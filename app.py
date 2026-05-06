import os
import io
import re
import base64
import json
import anthropic
from flask import Flask, request, jsonify, session, send_from_directory
from flask_cors import CORS
from supabase import create_client
from dotenv import load_dotenv
from geopy.distance import geodesic

# ─────────────────────────────────
# LOAD ENV
# ─────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, "new.env"))

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET", "alcoshield_secret_2025")

app.config.update(
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=False,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_PERMANENT=False
)

CORS(app, supports_credentials=True, origins=[
    "http://127.0.0.1:5500",
    "http://localhost:5500",
    "http://127.0.0.1:5000",
    "http://localhost:5000",
    "null"
])

# ─────────────────────────────────
# SUPABASE
# ─────────────────────────────────
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

print("SUPABASE_URL =", SUPABASE_URL)
print("SUPABASE_KEY exists =", bool(SUPABASE_KEY))

if not SUPABASE_URL or not SUPABASE_KEY:
    raise Exception("Missing SUPABASE_URL or SUPABASE_KEY in new.env")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ─────────────────────────────────
# ANTHROPIC CLIENT (for Vision OCR)
# ─────────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
anthropic_client  = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None
print("Anthropic Vision OCR =", "enabled" if anthropic_client else "disabled (no API key)")

# ─────────────────────────────────
# EASYOCR — import ocr_scanner.py
# ocr_scanner.py must be in the SAME directory as app.py
# ─────────────────────────────────
_easyocr_available = False
try:
    from ocr_scanner import extract_text_from_bytes as _easyocr_extract
    _easyocr_available = True
    print("EasyOCR (ocr_scanner) = enabled")
except ImportError:
    print("EasyOCR (ocr_scanner) = NOT FOUND — place ocr_scanner.py next to app.py")
except Exception as _ocr_init_err:
    print(f"EasyOCR init error: {_ocr_init_err}")


# ─────────────────────────────────
# HELPERS
# ─────────────────────────────────
def current_user():
    return session.get("user")


# ─────────────────────────────────────────────────────────
# ALCOHOL KEYWORD LISTS
# Used by both the simple boolean helper AND the rich detector
# ─────────────────────────────────────────────────────────
ALCOHOL_KEYWORDS = [
    "vodka", "whisky", "whiskey", "rum", "gin", "tequila", "brandy",
    "bourbon", "scotch", "cognac", "absinthe", "schnapps", "mezcal",
    "beer", "wine", "lager", "ale", "stout", "porter", "cider", "mead",
    "sake", "champagne", "prosecco", "sangria",
    "alcohol", "liquor", "spirits", "booze", "brew", "distillery",
    "winery", "brewery", "pub", "bar ", " bar", "tavern", "cocktail",
    "nightclub", "lounge bar", "liqueur",
    "wine shop", "liquor store", "off licence", "bottle shop",
    "the spirits", "spirit lounge", "liquor barn", "beer garden",
    "bar & grill", "bar and grill",
]

# ─────────────────────────────────────────────────────────
# RICH ALCOHOL DETECTOR
# Returns structured result consumed by all OCR layers,
# SMS scanner, and the boolean helper below.
# ─────────────────────────────────────────────────────────

# Grouped rules — add patterns per category as the project grows
ALCOHOL_RULES = {
    "spirits": [
        "vodka", "whisky", "whiskey", "rum", "gin", "tequila", "brandy",
        "bourbon", "scotch", "cognac", "absinthe", "schnapps", "mezcal",
        "kahlua", "baileys", "triple sec", "cointreau", "sambuca",
    ],
    "beer_wine": [
        "beer", "wine", "lager", "ale", "stout", "porter", "cider", "mead",
        "sake", "champagne", "prosecco", "sangria", "rose wine",
        "white wine", "red wine", "sparkling wine", "kingfisher",
        "heineken", "corona", "budweiser", "carlsberg",
    ],
    "venue_indicators": [
        "cocktail", "pub", "bar ", " bar", "tavern", "nightclub",
        "lounge bar", "beer garden", "bar & grill", "bar and grill",
        "distillery", "winery", "brewery",
    ],
    "retail_indicators": [
        "liquor store", "wine shop", "off licence", "bottle shop",
        "liquor barn", "spirit lounge", "spirits", "liqueur",
        "booze", "alcohol", "liquor",
    ],
    "recipe_ingredients": [
        # Cooking-with-alcohol patterns common in receipts / menu items
        "splash of rum", "dash of whiskey", "cup of wine", "tbsp of vodka",
        "marinated in beer", "beer batter", "wine sauce", "rum cake",
        "tiramisu", "coq au vin", "beef bourguignon", "beer bread",
        "flambe", "flambé",
    ],
    "abbreviations": [
        # Common receipt/bill shorthand
        "wsky", "bbn", "hsk",       # whisky variants
        "bdwy lager", "bkb",        # beer abbreviations
        "ipa", "xo ", " xo",        # IPA beer, XO cognac
        "vdk", "gnss",              # vodka, guinness
    ],
}

# Words that look alcoholic but aren't — skip flagging when found
NON_ALCOHOL_WHITELIST = [
    "apple cider vinegar", "cider vinegar", "vanilla extract",
    "grape juice", "wine gum", "ginger beer extract", "mocktail",
]


def detect_alcohol_recipe(text: str) -> dict:
    """
    Rich, rule-based alcohol detector.

    Returns:
        {
            "detected":        bool,
            "items_found":     list[str],   — matched keyword hits (deduped)
            "rules_triggered": list[str],   — which rule categories fired
            "confidence":      str,         — "high" | "medium" | "low"
            "verdict":         str          — "ALCOHOL" | "CLEAN" | "AMBIGUOUS" | "NO_TEXT"
        }
    """
    if not text or not text.strip():
        return {
            "detected": False, "items_found": [], "rules_triggered": [],
            "confidence": "low", "verdict": "NO_TEXT",
        }

    text_lower = text.lower()

    # Whitelist check — build set of whitelisted substrings present in text
    whitelist_active = {w for w in NON_ALCOHOL_WHITELIST if w in text_lower}

    items_found:     list[str] = []
    rules_triggered: list[str] = []

    for rule_name, keywords in ALCOHOL_RULES.items():
        for kw in keywords:
            if kw in text_lower:
                # Skip if this keyword is entirely contained in a whitelisted phrase
                if any(kw in wl for wl in whitelist_active):
                    continue
                items_found.append(kw.strip())
                if rule_name not in rules_triggered:
                    rules_triggered.append(rule_name)

    detected = len(items_found) > 0

    # Confidence: strong signals → high; venue/abbreviation alone → medium
    high_rules = {"spirits", "beer_wine", "retail_indicators", "recipe_ingredients"}
    if rules_triggered and (set(rules_triggered) & high_rules):
        confidence = "high"
    elif rules_triggered:
        confidence = "medium"
    else:
        confidence = "low"

    if not detected:
        verdict = "CLEAN"
    elif confidence == "medium":
        verdict = "AMBIGUOUS"
    else:
        verdict = "ALCOHOL"

    return {
        "detected":        detected,
        "items_found":     list(dict.fromkeys(items_found))[:10],  # deduped, order-preserving
        "rules_triggered": rules_triggered,
        "confidence":      confidence,
        "verdict":         verdict,
    }


def alcohol_detect(text: str) -> bool:
    """Simple boolean wrapper — keeps all existing callers working."""
    return detect_alcohol_recipe(text)["detected"]


def get_limit(user: dict) -> float:
    return 14.0 if user.get("gender", "male") == "male" else 10.0


def calc_risk(pct: float) -> str:
    if pct >= 100:   return "critical"
    elif pct >= 75:  return "high"
    elif pct >= 50:  return "medium"
    else:            return "low"


def estimate_units_from_amount(amount: float) -> float:
    if not amount or amount <= 0:
        return 0.0
    return min(round(amount / 200.0, 1), 10.0)


def safe_insert(table: str, data: dict):
    try:
        res = supabase.table(table).insert(data).execute()
        return res, None
    except Exception as e:
        err_str = str(e)
        safe_payloads = {
            "transactions": ["user_id", "units", "status"],
            "sms_logs":     ["user_id", "text", "sender", "amount", "alcohol"],
            "ocr_logs":     ["user_id", "filename", "detected", "ocr_text"],
        }
        if table in safe_payloads:
            minimal = {k: data[k] for k in safe_payloads[table] if k in data}
            try:
                res = supabase.table(table).insert(minimal).execute()
                return res, None
            except Exception as e2:
                return None, str(e2)
        return None, err_str


def get_current_consumption(uid: str):
    approved_txns = supabase.table("transactions").select("units") \
        .eq("user_id", uid).eq("status", "approved").execute().data or []
    return sum(t.get("units", 0) for t in approved_txns)


# ─────────────────────────────────
# BEHAVIORAL INTERVENTION MODULE
# ─────────────────────────────────
def behavioral_intervention(new_pct: float, trigger_source: str, status: str = "approved",
                             vendor: str = None, proximity_risk: str = None) -> dict:
    risk = calc_risk(new_pct)

    if status == "blocked":
        msg  = "🚫 Transaction RESTRICTED. Your weekly alcohol limit has been reached. This purchase was denied to protect your health."
        cls  = "blocked"
    elif new_pct >= 100:
        msg  = "🚨 WEEKLY THRESHOLD EXCEEDED. Consumption tracking shows you have surpassed your safe limit. No further alcohol is recommended."
        cls  = "danger"
    elif new_pct >= 75:
        msg  = "⚠️ BEHAVIORAL ALERT: You have consumed 75%+ of your weekly allowance. Context-aware system recommends stopping here."
        cls  = "danger"
    elif new_pct >= 50:
        msg  = "⚡ MID-WEEK ALERT: Over 50% of your weekly limit consumed. Pace yourself — daily allowance tracking is active."
        cls  = "caution"
    elif new_pct >= 25:
        msg  = "ℹ️ Consumption update logged. Stay mindful — you have remaining units this week. Drink responsibly."
        cls  = "safe"
    else:
        msg  = "✅ Consumption update logged. Well within your weekly limit. Remember to track all purchases."
        cls  = "safe"

    if proximity_risk in ("critical", "high") and vendor:
        if proximity_risk == "critical":
            msg = f"🚨 GPS ALERT: You are critically close to '{vendor}'. " + msg
        else:
            msg = f"⚠️ GPS PROXIMITY: Near '{vendor}'. " + msg

    return {
        "jitai_msg":    msg,
        "jitai_class":  cls,
        "risk":         risk,
        "percentage":   new_pct,
        "trigger":      trigger_source,
    }


# ─────────────────────────────────────────────────────────
# OCR LAYER 1 — CLAUDE VISION AI
# ─────────────────────────────────────────────────────────
def analyze_bill_with_claude(image_bytes: bytes, media_type: str, filename: str) -> dict:
    """
    Primary OCR layer: Claude Vision reads the receipt image and returns
    structured JSON covering all fields the frontend expects.
    """
    if not anthropic_client:
        return _try_easyocr(image_bytes, filename)

    try:
        b64 = base64.standard_b64encode(image_bytes).decode("utf-8")

        prompt = """You are an AI assistant for AlcoShield, an alcohol consumption monitoring system.

Analyze this receipt/bill image carefully and extract the following information in JSON format ONLY (no markdown, no extra text):

{
  "extracted_text": "<full raw text you can read from the receipt>",
  "all_items": ["<list of all line items / product names found>"],
  "alcohol_items": ["<list ONLY items that are alcoholic beverages — beer, wine, spirits, cocktails, liquor, etc.>"],
  "total_amount": <numeric total amount in INR (rupees), or 0 if not found>,
  "item_amounts": {"<item name>": <price>},
  "detected": <true if ANY alcohol items found, false otherwise>,
  "confidence": "<high|medium|low — your confidence in the analysis>",
  "vendor_name": "<restaurant/store name if visible>",
  "bill_date": "<date if visible, else null>",
  "analysis_summary": "<1-2 sentence plain English summary of what this bill contains and whether alcohol was found>"
}

Rules:
- Be thorough — check item names, codes, abbreviations that might indicate alcohol (e.g., 'BDWY LAGER', 'HSK' for whisky, 'BKB' for beer, etc.)
- Extract the EXACT rupee total from the bill (look for 'Total', 'Grand Total', 'Amount Due', 'Net Amount')
- If you cannot read the image clearly, set confidence to 'low' and detected to false
- Return ONLY valid JSON, no preamble or explanation"""

        response = anthropic_client.messages.create(
            model="claude-opus-4-6",
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": b64,
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                }
            ],
        )

        raw = response.content[0].text.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        data = json.loads(raw)

        return {
            "detected":          bool(data.get("detected", False)),
            "items_found":       data.get("alcohol_items", []),
            "all_items":         data.get("all_items", []),
            "amount_found":      float(data.get("total_amount", 0) or 0),
            "item_amounts":      data.get("item_amounts", {}),
            "ocr_text":          data.get("extracted_text", ""),
            "ocr_text_preview":  (data.get("extracted_text", "") or "")[:600],
            "confidence":        data.get("confidence", "medium"),
            "vendor_name":       data.get("vendor_name", ""),
            "bill_date":         data.get("bill_date"),
            "analysis_summary":  data.get("analysis_summary", ""),
            "method_used":       "claude-vision",
            "error":             None,
        }

    except json.JSONDecodeError as e:
        print(f"Claude Vision JSON parse error: {e}\nRaw: {raw[:500]}")
        return _try_easyocr(image_bytes, filename, str(e))
    except Exception as e:
        print(f"Claude Vision error: {e}")
        return _try_easyocr(image_bytes, filename, str(e))


# ─────────────────────────────────────────────────────────
# OCR LAYER 2 — EASYOCR  (ocr_scanner.py)
# Slots between Claude Vision and pytesseract.
# Falls through to pytesseract automatically on any failure.
# ─────────────────────────────────────────────────────────
def _try_easyocr(image_bytes: bytes, filename: str, upstream_error: str = None) -> dict:
    """
    EasyOCR layer via ocr_scanner.extract_text_from_bytes().
    Runs the rich alcohol detector on the extracted text and returns
    the same dict shape as all other layers so the /api/ocr route
    never needs to know which engine ran.
    """
    if not _easyocr_available:
        return _try_pytesseract(image_bytes, "image/jpeg", filename, upstream_error)

    try:
        ocr_result = _easyocr_extract(image_bytes)

        if not ocr_result.get("success"):
            print(f"EasyOCR returned failure: {ocr_result.get('error')}")
            return _try_pytesseract(image_bytes, "image/jpeg", filename,
                                    ocr_result.get("error", upstream_error))

        ocr_text = (ocr_result.get("text") or "").strip()

        if not ocr_text:
            # No text found — cascade down
            return _try_pytesseract(image_bytes, "image/jpeg", filename, upstream_error)

        # ── Run the rich alcohol detector on EasyOCR output ──
        detection  = detect_alcohol_recipe(ocr_text)
        detected   = detection["detected"]
        items      = detection["items_found"]
        rules      = detection["rules_triggered"]
        confidence = detection["confidence"]

        # Extract rupee amount from the raw text
        amounts = re.findall(
            r"(?:rs\.?|₹)\s*(\d+(?:,\d+)?(?:\.\d{1,2})?)",
            ocr_text.lower()
        )
        amount = float(amounts[-1].replace(",", "")) if amounts else 0.0

        ocr_segments   = ocr_result.get("segments", 0)
        ocr_confidence = ocr_result.get("confidence", 0.0)

        summary = (
            f"EasyOCR extracted {ocr_segments} text segments "
            f"(avg confidence {ocr_confidence:.2f}). "
            + (
                f"Alcohol keywords found: {', '.join(items[:5])}."
                if detected else "No alcohol keywords detected."
            )
        )

        return {
            "detected":          detected,
            "items_found":       items,
            "all_items":         [t for t in ocr_text.split("\n") if t.strip()],
            "amount_found":      amount,
            "item_amounts":      {},
            "ocr_text":          ocr_text,
            "ocr_text_preview":  ocr_text[:600],
            "confidence":        confidence,
            "vendor_name":       "",
            "bill_date":         None,
            "analysis_summary":  summary,
            "method_used":       "easyocr",
            "rules_triggered":   rules,
            "error":             upstream_error,
        }

    except Exception as e:
        print(f"EasyOCR wrapper exception: {e}")
        return _try_pytesseract(image_bytes, "image/jpeg", filename, str(e))


# ─────────────────────────────────────────────────────────
# OCR LAYER 3 — PYTESSERACT
# Last image-based fallback before filename keyword check.
# ─────────────────────────────────────────────────────────
def _try_pytesseract(file_bytes: bytes, media_type: str, filename: str,
                     upstream_error: str = None) -> dict:
    """pytesseract fallback with rich alcohol detection."""
    try:
        from PIL import Image
        import pytesseract

        if media_type.startswith("image/"):
            img      = Image.open(io.BytesIO(file_bytes))
            ocr_text = pytesseract.image_to_string(img)

            detection  = detect_alcohol_recipe(ocr_text)
            detected   = detection["detected"]
            items      = detection["items_found"]
            confidence = detection["confidence"]

            # Extract rupee amount
            amounts = re.findall(
                r"(?:rs\.?|₹)\s*(\d+(?:,\d+)?(?:\.\d{1,2})?)",
                ocr_text.lower()
            )
            amount = float(amounts[-1].replace(",", "")) if amounts else 0.0

            return {
                "detected":          detected,
                "items_found":       items,
                "all_items":         [],
                "amount_found":      amount,
                "item_amounts":      {},
                "ocr_text":          ocr_text,
                "ocr_text_preview":  ocr_text[:600],
                "confidence":        confidence if detected else "low",
                "vendor_name":       "",
                "bill_date":         None,
                "analysis_summary":  (
                    f"pytesseract extracted text. "
                    f"{'Alcohol keywords found.' if detected else 'No alcohol keywords.'}"
                ),
                "method_used":       "pytesseract",
                "error":             upstream_error,
            }
    except ImportError:
        pass
    except Exception as e:
        print(f"pytesseract error: {e}")

    # ── Final fallback: filename keyword check ──
    detected = alcohol_detect(filename)
    items    = [kw.strip() for kw in ALCOHOL_KEYWORDS if kw.strip() in filename.lower()][:5]
    return {
        "detected":          detected,
        "items_found":       items,
        "all_items":         [],
        "amount_found":      0.0,
        "item_amounts":      {},
        "ocr_text":          "",
        "ocr_text_preview":  "",
        "confidence":        "low",
        "vendor_name":       "",
        "bill_date":         None,
        "analysis_summary":  (
            "No OCR engine available. Used filename keyword detection only."
        ),
        "method_used":       "filename-fallback",
        "error":             upstream_error,
    }


# ─────────────────────────────────────────────────────────
# FALLBACK dispatcher (kept for backward-compat callers)
# ─────────────────────────────────────────────────────────
def _fallback_ocr(filename: str, error: str = None) -> dict:
    """
    Called when Claude Vision is unavailable AND no image bytes are
    in scope (e.g. JSON-parse error path). Delegates to EasyOCR/pytesseract
    can't run without bytes — so falls straight to filename detection.
    """
    detected = alcohol_detect(filename)
    items    = [kw.strip() for kw in ALCOHOL_KEYWORDS if kw.strip() in filename.lower()][:5]
    return {
        "detected":          detected,
        "items_found":       items,
        "all_items":         [],
        "amount_found":      0.0,
        "item_amounts":      {},
        "ocr_text":          "",
        "ocr_text_preview":  "",
        "confidence":        "low",
        "vendor_name":       "",
        "bill_date":         None,
        "analysis_summary":  (
            f"AI Vision unavailable — used filename detection only. "
            f"Error: {error or 'no API key'}"
        ),
        "method_used":       "filename-fallback",
        "error":             error,
    }


# ─────────────────────────────────
# SERVE FRONTEND
# ─────────────────────────────────
@app.route("/")
def home():
    return send_from_directory(".", "main.html")


# ─────────────────────────────────
# AUTH
# ─────────────────────────────────
@app.route("/api/register", methods=["POST"])
def register():
    data = request.json
    required = ["name", "email", "password", "gender", "age"]
    if not data or not all(k in data for k in required):
        return jsonify({"error": "All fields required"}), 400

    age = int(data.get("age", 0))

    if age <= 21:
        return jsonify({
            "error": "Access Denied — you must be over 21 to register.",
            "denied": True,
        }), 403

    existing = supabase.table("users").select("id").eq("email", data["email"]).execute()
    if existing.data:
        return jsonify({"error": "Email already registered"}), 409

    try:
        res = supabase.table("users").insert({
            "name":     data["name"],
            "email":    data["email"],
            "password": data["password"],
            "gender":   data["gender"],
            "age":      age,
        }).execute()
    except Exception as e:
        return jsonify({"error": f"Registration failed: {str(e)}"}), 500

    if not res.data:
        return jsonify({"error": "Registration failed"}), 500

    user = res.data[0]
    session["user"] = user
    limit = get_limit(user)
    return jsonify({
        "user":         user,
        "initialized":  True,
        "weekly_limit": limit,
        "message":      f"Account created. Consumption tracking initialized. Weekly limit set to {limit} units.",
    })


@app.route("/api/login", methods=["POST"])
def login():
    data = request.json
    if not data or not data.get("email") or not data.get("password"):
        return jsonify({"error": "Email and password required"}), 400

    try:
        res = supabase.table("users").select("*") \
            .eq("email", data["email"]) \
            .eq("password", data["password"]) \
            .execute()
    except Exception as e:
        return jsonify({"error": f"Login error: {str(e)}"}), 500

    if not res.data:
        return jsonify({"error": "Invalid email or password"}), 401

    session["user"] = res.data[0]
    return jsonify({"user": res.data[0]})


@app.route("/api/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"ok": True})


# ─────────────────────────────────
# DASHBOARD
# ─────────────────────────────────
@app.route("/api/dashboard")
def dashboard():
    user = current_user()
    if not user:
        return jsonify({"error": "Not logged in"}), 401
    return jsonify(user)


# ─────────────────────────────────
# INSIGHTS
# ─────────────────────────────────
@app.route("/api/insights")
def insights():
    user = current_user()
    if not user:
        return jsonify({"error": "Not logged in"}), 401

    uid   = user["id"]
    limit = get_limit(user)

    try:
        txns     = supabase.table("transactions").select("*").eq("user_id", uid).execute().data or []
        sms_rows = supabase.table("sms_logs").select("*").eq("user_id", uid).execute().data or []
        ocr_rows = supabase.table("ocr_logs").select("*").eq("user_id", uid).execute().data or []
    except Exception as e:
        return jsonify({"error": f"DB read error: {str(e)}"}), 500

    approved    = [t for t in txns if t.get("status") == "approved"]
    blocked     = [t for t in txns if t.get("status") == "blocked"]
    consumption = sum(t.get("units", 0) for t in approved)
    sms_flagged = len([s for s in sms_rows if s.get("alcohol")])
    ocr_flagged = len([o for o in ocr_rows if o.get("detected")])

    pct  = round((consumption / limit) * 100, 1) if limit else 0
    risk = calc_risk(pct)

    total_txns   = len(approved) + len(blocked)
    blocked_pct  = round(len(blocked)  / total_txns * 100, 1) if total_txns else 0
    approved_pct = round(len(approved) / total_txns * 100, 1) if total_txns else 0

    daily_data = [0.0] * 7
    for i, t in enumerate(reversed(approved[-35:])):
        day_idx = min(i // max(len(approved[-35:]) // 7, 1), 6)
        daily_data[day_idx] = round(daily_data[day_idx] + t.get("units", 0), 1)

    intervention = behavioral_intervention(pct, trigger_source="insights")

    return jsonify({
        "limit":           limit,
        "consumption":     round(consumption, 2),
        "remaining":       round(max(limit - consumption, 0), 2),
        "transactions":    len(approved),
        "blocked":         len(blocked),
        "sms_flagged":     sms_flagged,
        "ocr_flagged":     ocr_flagged,
        "risk":            risk,
        "daily_allowance": round(limit / 7, 2),
        "percentage":      pct,
        "daily_data":      daily_data,
        "blocked_pct":     blocked_pct,
        "approved_pct":    approved_pct,
        "total_txns":      total_txns,
        "intervention":    intervention,
    })


# ─────────────────────────────────
# STATS
# ─────────────────────────────────
@app.route("/api/stats")
def stats():
    user = current_user()
    if not user:
        return jsonify({"error": "Not logged in"}), 401

    uid   = user["id"]
    limit = get_limit(user)

    try:
        txns = supabase.table("transactions").select("*").eq("user_id", uid).execute().data or []
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    approved = [t for t in txns if t.get("status") == "approved"]
    blocked  = [t for t in txns if t.get("status") == "blocked"]

    daily_units = [0.0] * 7
    bucket_size = max(len(approved) // 7, 1)
    for i, t in enumerate(reversed(approved[-35:])):
        day_idx = min(i // bucket_size, 6)
        daily_units[day_idx] = round(daily_units[day_idx] + t.get("units", 0), 1)

    total = len(approved) + len(blocked)

    return jsonify({
        "days":           ["Day 1", "Day 2", "Day 3", "Day 4", "Day 5", "Day 6", "Day 7"],
        "daily_units":    daily_units,
        "daily_limit":    round(limit / 7, 2),
        "weekly_limit":   limit,
        "total_consumed": round(sum(t.get("units", 0) for t in approved), 2),
        "total_approved": len(approved),
        "total_blocked":  len(blocked),
        "blocked_pct":    round(len(blocked)  / total * 100, 1) if total else 0,
        "approved_pct":   round(len(approved) / total * 100, 1) if total else 0,
    })


# ─────────────────────────────────
# PREDICT
# ─────────────────────────────────
@app.route("/api/predict", methods=["POST"])
def predict():
    user = current_user()
    if not user:
        return jsonify({"error": "Not logged in"}), 401

    data  = request.json or {}
    units = float(data.get("units", 0))
    if units <= 0:
        return jsonify({"error": "Units must be positive"}), 400

    uid   = user["id"]
    limit = get_limit(user)

    try:
        current_total = get_current_consumption(uid)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    predicted_total = current_total + units
    exceeds         = predicted_total > limit
    predicted_pct   = round((predicted_total / limit) * 100, 1) if limit else 0
    current_pct     = round((current_total   / limit) * 100, 1) if limit else 0

    intervention = behavioral_intervention(
        predicted_pct,
        trigger_source="transaction_predict",
        status="blocked" if exceeds else "approved",
    )

    return jsonify({
        "will_exceed":      exceeds,
        "current_total":    round(current_total, 2),
        "predicted_total":  round(predicted_total, 2),
        "remaining_after":  round(max(limit - predicted_total, 0), 2),
        "current_pct":      current_pct,
        "predicted_pct":    predicted_pct,
        "limit":            limit,
        "decision":         "BLOCK" if exceeds else "APPROVE",
        "intervention":     intervention,
    })


# ─────────────────────────────────
# TRANSACTIONS
# ─────────────────────────────────
@app.route("/api/transaction", methods=["POST"])
def transaction():
    user = current_user()
    if not user:
        return jsonify({"error": "Not logged in"}), 401

    data = request.json
    if not data or "units" not in data:
        return jsonify({"error": "Units required"}), 400

    uid        = user["id"]
    units      = float(data["units"])
    vendor     = data.get("vendor", "Unknown Vendor")
    amount_inr = float(data.get("amount_inr", 0) or 0)

    if units <= 0:
        return jsonify({"error": "Units must be positive"}), 400

    limit = get_limit(user)

    try:
        current_total = get_current_consumption(uid)
    except Exception as e:
        return jsonify({"error": f"DB read error: {str(e)}"}), 500

    predicted_total = current_total + units
    pre_pct         = round((current_total / limit) * 100, 1) if limit else 0

    if predicted_total > limit:
        status    = "blocked"
        new_total = current_total
        new_pct   = pre_pct
    else:
        status    = "approved"
        new_total = predicted_total
        new_pct   = round((new_total / limit) * 100, 1)

    row, err = safe_insert("transactions", {
        "user_id":     uid,
        "units":       units,
        "status":      status,
        "vendor_name": vendor,
        "amount_inr":  amount_inr,
    })
    if err:
        return jsonify({"error": f"Insert failed: {err}"}), 500

    intervention = behavioral_intervention(
        new_pct,
        trigger_source="transaction",
        status=status,
        vendor=vendor,
    )

    return jsonify({
        "status":       status,
        "units":        units,
        "total":        round(new_total, 2),
        "remaining":    round(max(limit - new_total, 0), 2),
        "percentage":   new_pct,
        "pre_pct":      pre_pct,
        "risk":         calc_risk(new_pct),
        "vendor_name":  vendor,
        "amount_inr":   amount_inr,
        "intervention": intervention,
        "jitai_msg":    intervention["jitai_msg"],
    })


@app.route("/api/history")
def history():
    user = current_user()
    if not user:
        return jsonify({"error": "Not logged in"}), 401

    try:
        rows = supabase.table("transactions").select("*") \
            .eq("user_id", user["id"]) \
            .order("id", desc=True) \
            .limit(50) \
            .execute().data or []
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify(rows)


# ─────────────────────────────────
# SMS MONITOR
# ─────────────────────────────────
@app.route("/api/sms-scan", methods=["POST"])
def sms_scan():
    user = current_user()
    if not user:
        return jsonify({"error": "Not logged in"}), 401

    data   = request.json or {}
    text   = data.get("text", "")
    sender = data.get("sender", "Unknown")
    amount = float(data.get("amount", 0) or 0)

    detected        = alcohol_detect(text)
    estimated_units = 0.0
    payment_status  = "none"
    intervention    = {}

    if detected and amount > 0:
        uid             = user["id"]
        limit           = get_limit(user)
        estimated_units = estimate_units_from_amount(amount)

        try:
            current_total  = get_current_consumption(uid)
            new_total      = current_total + estimated_units
            new_pct        = round((new_total / limit) * 100, 1)
            payment_status = "over_limit" if new_total > limit else "within_limit"

            safe_insert("transactions", {
                "user_id":     uid,
                "units":       estimated_units,
                "status":      "approved",
                "vendor_name": f"SMS-External: {sender}",
                "amount_inr":  amount,
            })

            intervention = behavioral_intervention(
                new_pct,
                trigger_source="sms_monitoring",
                status="approved",
                vendor=sender,
            )

        except Exception as e:
            print(f"SMS unit estimation error: {e}")

    elif detected:
        intervention = {
            "jitai_msg":   "⚠️ Alcohol-related external SMS detected but no amount provided. Please check your consumption manually.",
            "jitai_class": "caution",
            "risk":        "medium",
            "trigger":     "sms_monitoring_no_amount",
        }

    row, err = safe_insert("sms_logs", {
        "user_id":         user["id"],
        "text":            text,
        "sender":          sender,
        "amount":          amount,
        "alcohol":         detected,
        "estimated_units": estimated_units,
    })
    if err:
        return jsonify({"error": f"SMS log failed: {err}"}), 500

    return jsonify({
        "alcohol":          detected,
        "estimated_units":  estimated_units,
        "payment_status":   payment_status,
        "intervention":     intervention,
        "jitai_msg":        intervention.get("jitai_msg", ""),
        "message": "Alcohol-related external transaction detected and tracked!" if detected else "No alcohol detected.",
    })


@app.route("/api/sms-history")
def sms_history():
    user = current_user()
    if not user:
        return jsonify({"error": "Not logged in"}), 401

    try:
        rows = supabase.table("sms_logs").select("*") \
            .eq("user_id", user["id"]) \
            .order("id", desc=True) \
            .limit(50) \
            .execute().data or []
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify(rows)


# ─────────────────────────────────────────────────────────
# OCR BILL VERIFICATION — 3-LAYER PIPELINE
#
# Layer 1: Claude Vision AI  (needs ANTHROPIC_API_KEY)
# Layer 2: EasyOCR           (needs ocr_scanner.py in same dir)
# Layer 3: pytesseract       (needs pytesseract + Tesseract binary)
# Layer 4: filename fallback (always available)
#
# All layers return the same dict shape so the route logic below
# never needs to know which engine ran.
# ─────────────────────────────────────────────────────────
@app.route("/api/ocr", methods=["POST"])
def ocr():
    user = current_user()
    if not user:
        return jsonify({"error": "Not logged in"}), 401

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    f          = request.files["file"]
    filename   = f.filename or "unknown"
    file_bytes = f.read()
    media_type = f.content_type or "image/jpeg"

    # ── Choose OCR engine ──
    if anthropic_client and media_type.startswith("image/"):
        # Layer 1: Claude Vision (cascades to EasyOCR internally on failure)
        result = analyze_bill_with_claude(file_bytes, media_type, filename)
    elif _easyocr_available and media_type.startswith("image/"):
        # Layer 2: EasyOCR (cascades to pytesseract internally on failure)
        result = _try_easyocr(file_bytes, filename)
    else:
        # Layer 3 / 4: pytesseract → filename fallback
        result = _try_pytesseract(file_bytes, media_type, filename)

    # ── Unpack result (identical keys from all layers) ──
    detected         = result["detected"]
    items_found      = result["items_found"]
    all_items        = result.get("all_items", [])
    amount_found     = result.get("amount_found", 0.0)
    ocr_text         = result.get("ocr_text", "")
    ocr_text_preview = result.get("ocr_text_preview", "")
    confidence       = result.get("confidence", "medium")
    vendor_name      = result.get("vendor_name", "")
    bill_date        = result.get("bill_date")
    analysis_summary = result.get("analysis_summary", "")
    method_used      = result.get("method_used", "unknown")
    ai_error         = result.get("error")

    # ── Consumption tracking ──
    estimated_units = 0.0
    auto_logged     = False
    intervention    = {}

    uid   = user["id"]
    limit = get_limit(user)

    try:
        current_total = get_current_consumption(uid)
        current_pct   = round((current_total / limit) * 100, 1)
    except Exception:
        current_total = 0
        current_pct   = 0

    if detected:
        if amount_found > 0:
            estimated_units = estimate_units_from_amount(amount_found)

        # Auto-log units when we have a confident amount
        if estimated_units > 0 and confidence in ("high", "medium"):
            try:
                new_total = current_total + estimated_units
                new_pct   = round((new_total / limit) * 100, 1)
                safe_insert("transactions", {
                    "user_id":     uid,
                    "units":       estimated_units,
                    "status":      "approved",
                    "vendor_name": f"OCR-{method_used.upper()}: {vendor_name or filename}",
                    "amount_inr":  amount_found,
                })
                auto_logged = True
                current_pct = new_pct
            except Exception as e:
                print(f"OCR auto-log error: {e}")

        intervention = behavioral_intervention(
            current_pct,
            trigger_source="ocr_verification",
            status="approved",
            vendor=vendor_name or filename,
        )

        items_str = ", ".join(items_found) if items_found else "alcohol items detected"
        auto_msg  = (
            f" · {estimated_units} units auto-logged to tracking." if auto_logged
            else " · Log manually in Payment Gate if this was a real purchase."
        )
        intervention["jitai_msg"] = (
            f"🧾 OCR AI ALERT [{method_used.upper()}]: Alcohol detected — {items_str}."
            f"{' ₹' + str(int(amount_found)) + ' extracted.' if amount_found > 0 else ''}"
            f"{auto_msg}"
        )
    else:
        intervention = {
            "jitai_msg":   (
                f"✅ [{method_used.upper()}] Receipt '{filename}' verified — "
                "no alcohol items detected. This is a clean, non-alcoholic purchase."
            ),
            "jitai_class": "safe",
            "risk":        "low",
            "trigger":     "ocr_verification",
        }

    # ── Persist to DB ──
    row, err = safe_insert("ocr_logs", {
        "user_id":  uid,
        "filename": filename,
        "detected": detected,
        "ocr_text": (ocr_text or "")[:500],
    })
    if err:
        return jsonify({"error": f"OCR log failed: {err}"}), 500

    # ── Return full response (all fields expected by main.html) ──
    return jsonify({
        "filename":          filename,
        "detected":          detected,
        "items_found":       items_found,
        "all_items":         all_items,
        "amount_found":      amount_found,
        "estimated_units":   estimated_units,
        "auto_logged":       auto_logged,
        "confidence":        confidence,
        "vendor_name":       vendor_name,
        "bill_date":         bill_date,
        "analysis_summary":  analysis_summary,
        "method_used":       method_used,
        "ocr_text_preview":  ocr_text_preview,
        "ai_error":          ai_error,
        "intervention":      intervention,
        "jitai_msg":         intervention.get("jitai_msg", ""),
        "message":           "Alcohol item found in bill!" if detected else "No alcohol items detected.",
    })


@app.route("/api/ocr-history")
def ocr_history():
    user = current_user()
    if not user:
        return jsonify({"error": "Not logged in"}), 401

    try:
        rows = supabase.table("ocr_logs").select("*") \
            .eq("user_id", user["id"]) \
            .order("id", desc=True) \
            .limit(50) \
            .execute().data or []
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify(rows)


# ─────────────────────────────────
# GPS PROXIMITY
# ─────────────────────────────────
VENDORS = [
    {"name": "Wine Store A",          "lat": 12.9716, "lng": 77.5946},
    {"name": "The Tavern Bar",        "lat": 12.9720, "lng": 77.6000},
    {"name": "Liquor Shop C",         "lat": 12.9750, "lng": 77.6050},
    {"name": "Spirits Den",           "lat": 12.9680, "lng": 77.5900},
    {"name": "Pub & Grub",            "lat": 12.9800, "lng": 77.6100},
    {"name": "Koramangala Bar",       "lat": 12.9352, "lng": 77.6244},
    {"name": "Indiranagar Wines",     "lat": 12.9784, "lng": 77.6408},
    {"name": "MG Road Liquor Mart",   "lat": 12.9756, "lng": 77.6067},
    {"name": "Whitefield Pub",        "lat": 12.9698, "lng": 77.7500},
    {"name": "JP Nagar Bottle Shop",  "lat": 12.9060, "lng": 77.5900},
]


@app.route("/api/gps", methods=["POST"])
def gps():
    user = current_user()
    if not user:
        return jsonify({"error": "Not logged in"}), 401

    data = request.json or {}
    if "lat" not in data or "lng" not in data:
        return jsonify({"error": "lat and lng required"}), 400

    user_loc = (float(data["lat"]), float(data["lng"]))

    nearby       = []
    risk_summary = {"critical": 0, "high": 0, "medium": 0, "low": 0}

    for v in VENDORS:
        dist = geodesic(user_loc, (v["lat"], v["lng"])).meters
        if dist < 200:    proximity_risk = "critical"
        elif dist < 500:  proximity_risk = "high"
        elif dist < 1000: proximity_risk = "medium"
        else:             proximity_risk = "low"

        risk_summary[proximity_risk] += 1
        nearby.append({
            "name":           v["name"],
            "lat":            v["lat"],
            "lng":            v["lng"],
            "distance":       int(dist),
            "proximity_risk": proximity_risk,
        })

    nearby.sort(key=lambda x: x["distance"])

    intervention = {}
    gps_alert    = False
    current_pct  = 0

    if nearby:
        closest = nearby[0]
        uid     = user["id"]
        limit   = get_limit(user)
        try:
            current_total = get_current_consumption(uid)
            current_pct   = round((current_total / limit) * 100, 1)
        except Exception:
            current_pct = 0

        if closest["proximity_risk"] in ("critical", "high"):
            gps_alert    = True
            intervention = behavioral_intervention(
                current_pct,
                trigger_source="gps_proximity",
                status="approved",
                vendor=closest["name"],
                proximity_risk=closest["proximity_risk"],
            )
        else:
            intervention = {
                "jitai_msg":   (
                    f"📍 Location detected. Nearest alcohol vendor is {closest['name']} "
                    f"({closest['distance']}m away). You are at safe distance."
                ),
                "jitai_class": "safe",
                "risk":        calc_risk(current_pct),
                "trigger":     "gps_proximity",
            }

        try:
            supabase.table("gps_logs").insert({
                "user_id":        uid,
                "user_lat":       user_loc[0],
                "user_lng":       user_loc[1],
                "closest_vendor": closest["name"],
                "closest_dist":   closest["distance"],
                "proximity_risk": closest["proximity_risk"],
            }).execute()
        except Exception:
            pass

    return jsonify({
        "nearby":       nearby,
        "risk_summary": risk_summary,
        "current_pct":  current_pct,
        "gps_alert":    gps_alert,
        "intervention": intervention,
        "jitai_msg":    intervention.get("jitai_msg", ""),
    })


# ─────────────────────────────────
# CONTINUOUS CONSUMPTION CHECK
# ─────────────────────────────────
@app.route("/api/consumption-check")
def consumption_check():
    user = current_user()
    if not user:
        return jsonify({"error": "Not logged in"}), 401

    uid   = user["id"]
    limit = get_limit(user)

    try:
        current_total = get_current_consumption(uid)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    current_pct        = round((current_total / limit) * 100, 1)
    threshold_exceeded = current_total >= limit
    intervention       = behavioral_intervention(current_pct, trigger_source="continuous_tracking")

    return jsonify({
        "consumption":        round(current_total, 2),
        "limit":              limit,
        "remaining":          round(max(limit - current_total, 0), 2),
        "percentage":         current_pct,
        "risk":               calc_risk(current_pct),
        "threshold_exceeded": threshold_exceeded,
        "intervention":       intervention,
    })


# ─────────────────────────────────
# RESET
# ─────────────────────────────────
@app.route("/api/reset", methods=["POST"])
def reset():
    user = current_user()
    if not user:
        return jsonify({"error": "Not logged in"}), 401

    uid = user["id"]
    try:
        supabase.table("transactions").delete().eq("user_id", uid).execute()
        supabase.table("sms_logs").delete().eq("user_id", uid).execute()
        supabase.table("ocr_logs").delete().eq("user_id", uid).execute()
    except Exception as e:
        return jsonify({"error": f"Reset error: {str(e)}"}), 500

    try:
        supabase.table("gps_logs").delete().eq("user_id", uid).execute()
    except Exception:
        pass

    return jsonify({"ok": True, "message": "All data reset successfully"})


# ─────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, port=5000)