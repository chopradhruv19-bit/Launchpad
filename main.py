import os
import time
import requests
from flask import Flask, request, jsonify

app = Flask(__name__)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# Try the main model first, then a lighter fallback if Gemini is overloaded.
GEMINI_MODELS = [
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
]

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
    "register my company",
    "incorporate",
    "llc",
    "tax structure",
    "contract",
    "trademark",
    "liability",
    "gst registration",
    "legal entity",
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
        return jsonify({
            "error": "server_not_configured",
            "detail": "GEMINI_API_KEY is not set on the server."
        }), 500

    data = request.get_json(force=True, silent=True) or {}
    idea = (data.get("idea") or "").strip()

    if not idea:
        return jsonify({
            "error": "idea_required",
            "detail": "Please enter a business idea."
        }), 400

    # Human-oversight safety trigger
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
            {
                "role": "user",
                "parts": [
                    {
                        "text": SYSTEM_PROMPT
                        + "\n\nFounder idea: "
                        + idea
                    }
                ]
            }
        ]
    }

    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": GEMINI_API_KEY,
    }

    # Temporary Gemini errors can happen during high demand.
    retryable_statuses = {429, 500, 502, 503, 504}

    last_error = None

    # Try the primary model and then the fallback model.
    for model in GEMINI_MODELS:

        gemini_url = (
            "https://generativelanguage.googleapis.com/"
            "v1beta/models/"
            + model
            + ":generateContent"
        )

        for attempt in range(3):

            try:
                resp = requests.post(
                    gemini_url,
                    json=payload,
                    headers=headers,
                    timeout=45,
                )

            except requests.RequestException as exc:
                last_error = str(exc)

                if attempt < 2:
                    time.sleep(2 ** attempt)
                    continue

                break

            # Successful Gemini response
            if resp.status_code == 200:

                try:
                    body = resp.json()
                    text = body["candidates"][0]["content"]["parts"][0]["text"]

                    return jsonify({
                        "plan": text,
                        "flagged": False
                    })

                except (KeyError, IndexError, TypeError, ValueError):
                    return jsonify({
                        "error": "invalid_gemini_response",
                        "detail": "Gemini returned an unexpected response."
                    }), 502

            # Gemini is temporarily overloaded or rate limited.
            if resp.status_code in retryable_statuses:

                last_error = resp.text[:500]

                if attempt < 2:
                    # 2 seconds, then 4 seconds, then 8 seconds
                    time.sleep(2 ** attempt)
                    continue

                # Try the fallback model
                break

            # Non-retryable Gemini error
            return jsonify({
                "error": "gemini_error",
                "detail": resp.text[:500]
            }), 502

    # Both models were unavailable after retries.
    return jsonify({
        "error": "gemini_temporarily_unavailable",
        "detail": (
            "Gemini is temporarily experiencing high demand. "
            "Please try again in a few seconds."
        )
    }), 503


@app.route("/healthz", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8080))
    )
