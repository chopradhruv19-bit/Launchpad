import os
import requests
from flask import Flask, request, jsonify

app = Flask(__name__)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent"

SYSTEM_PROMPT = """You are LaunchPad, an AI co-founder for first-time entrepreneurs. A founder will describe their business idea. Respond with a concise, concrete plan in this exact structure, using plain language and no fluff:

BUSINESS PLAN
- Who it's for (be specific, not "everyone")
- The core problem you're solving
- How the product/service works, in one paragraph
- Why this beats what people do today (be specific about the alternative)

PRICING MODEL
- Suggested price point and pricing structure
- One sentence on why this price makes sense for this customer

FIRST 10 CUSTOMERS
- A short, ready-to-send outreach message (2-3 sentences, no corporate tone)
- 3 specific places to find these first customers

ONE THING TO DO THIS WEEK
- The single highest-leverage action to take right now

Keep the whole response under 350 words. Be specific to their idea, never generic."""

LEGAL_KEYWORDS = [
    "register my company", "incorporate", "llc", "tax structure",
    "contract", "trademark", "liability", "gst registration", "legal entity",
]


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


@app.route("/generate-plan", methods=["POST", "OPTIONS"])
def generate_plan():
    if request.method == "OPTIONS":
        return ("", 204)

    if not GEMINI_API_KEY:
        return jsonify({"error": "server_not_configured",
                         "detail": "GEMINI_API_KEY is not set on this Cloud Run service."}), 500

    data = request.get_json(force=True, silent=True) or {}
    idea = (data.get("idea") or "").strip()

    if not idea:
        return jsonify({"error": "idea is required"}), 400

    # Human oversight trigger: legal/tax questions get a fixed disclaimer,
    # not a model-generated answer.
    lowered = idea.lower()
    if any(word in lowered for word in LEGAL_KEYWORDS):
        return jsonify({
            "plan": (
                "This touches on legal registration, tax structure, or contracts. "
                "LaunchPad doesn't generate advice on this — please talk to a "
                "licensed advisor or your local business registration office "
                "before acting on anything legal or tax-related. Once that's "
                "sorted, come back and describe the product/customer side of "
                "your idea and I'll help with the plan."
            ),
            "flagged": True,
        })

    payload = {
        "contents": [
            {"role": "user", "parts": [{"text": SYSTEM_PROMPT + "\n\nFounder idea: " + idea}]}
        ]
    }
    headers = {"Content-Type": "application/json", "x-goog-api-key": GEMINI_API_KEY}

    try:
        resp = requests.post(GEMINI_URL, json=payload, headers=headers, timeout=30)
    except requests.RequestException as exc:
        return jsonify({"error": "gemini_unreachable", "detail": str(exc)}), 502

    if resp.status_code != 200:
        return jsonify({"error": "gemini_error", "detail": resp.text[:300]}), 502

    body = resp.json()
    try:
        text = body["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError):
        text = "No response generated — try rephrasing your idea."

    return jsonify({"plan": text, "flagged": False})


@app.route("/healthz", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
