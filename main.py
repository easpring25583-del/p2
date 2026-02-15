import json
import os
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import vertexai
from vertexai.generative_models import (
    Content,
    FunctionDeclaration,
    GenerativeModel,
    Part,
    Tool,
)

# Optional (EC3): Secret Manager
def read_secret_payload(secret_resource: str) -> Optional[str]:
    """
    secret_resource format:
      projects/PROJECT_ID/secrets/SECRET_NAME/versions/latest
    """
    try:
        from google.cloud import secretmanager
        client = secretmanager.SecretManagerServiceClient()
        resp = client.access_secret_version(request={"name": secret_resource})
        return resp.payload.data.decode("utf-8")
    except Exception:
        return None


# -----------------------------
# Config
# -----------------------------
# Local debug switch:
# - DEBUG_STUB=1 (default): do NOT require GCP env vars; return stub output for local UI/API debugging.
# - DEBUG_STUB=0: enable Vertex AI Gemini calls (requires PROJECT_ID + Vertex permissions in Cloud Run).
DEBUG_STUB = os.getenv("DEBUG_STUB", "1") == "1"

PROJECT_ID = os.getenv("PROJECT_ID")
LOCATION = os.getenv("LOCATION", "us-central1")
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.0-flash-001")

# Optional: pull extra config from Secret Manager (EC3 demo)
# e.g. SECRET_CONFIG = projects/.../secrets/MEAL_PLANNER_CONFIG/versions/latest
SECRET_CONFIG = os.getenv("SECRET_CONFIG")
SECRET_CONFIG_TEXT = read_secret_payload(SECRET_CONFIG) if SECRET_CONFIG else None

# Vertex AI init (fail loudly if misconfigured; do NOT silently fall back to stub)
model = None
if not DEBUG_STUB:
    if not PROJECT_ID:
        raise RuntimeError("Missing env var PROJECT_ID (set it in Cloud Run > Variables & Secrets).")
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    model = GenerativeModel(MODEL_NAME)


# -----------------------------
# Tool (EC2)
# -----------------------------
normalize_request_func = FunctionDeclaration(
    name="normalize_request",
    description=(
        "Normalize the user's free-form input into a structured request. "
        "Extract ingredients (required) and optional constraints (dietary, meals, cooking_time). "
        "If ingredients are missing, return an error field explaining what's missing."
    ),
    parameters={
        "type": "object",
        "properties": {
            "user_text": {
                "type": "string",
                "description": "Raw user text describing ingredients and optional constraints.",
            }
        },
        "required": ["user_text"],
    },
)

tool = Tool(function_declarations=[normalize_request_func])


def normalize_request(user_text: str) -> Dict[str, Any]:
    """
    Minimal deterministic tool:
    - Tries to extract ingredients from patterns like 'ingredients: ...' or by comma-separated list.
    - Everything else stays as notes.
    This is intentionally simple to keep implementation minimal.
    """
    text = (user_text or "").strip()
    lower = text.lower()

    ingredients = []
    if "ingredients:" in lower:
        after = text[lower.index("ingredients:") + len("ingredients:"):].strip()
        after = after.splitlines()[0].strip()
        ingredients = [x.strip() for x in after.split(",") if x.strip()]
    else:
        if "," in text and len(text) <= 200:
            ingredients = [x.strip() for x in text.split(",") if x.strip()]

    if not ingredients:
        return {
            "error": "Missing ingredients. Please include 'ingredients: item1, item2, ...' or a comma-separated list.",
            "ingredients": [],
            "notes": text,
        }

    return {
        "ingredients": ingredients,
        "notes": text,
    }


# -----------------------------
# App
# -----------------------------
app = FastAPI()

# Serve your existing index.html (put it under ./static/index.html)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
def home():
    try:
        with open("static/index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>Meal Planner</h1><p>Missing static/index.html</p>"


class MenuRequest(BaseModel):
    input: str


@app.post("/api/menu")
def generate_menu(req: MenuRequest):
    user_text = (req.input or "").strip()
    if not user_text:
        raise HTTPException(status_code=400, detail="Empty input")

    # Stub output (no GCP required)
    if DEBUG_STUB:
        return {
            "result": (
                "Breakfast: Simple oatmeal (if allowed)\n"
                "Lunch: Tofu + tomato pasta\n"
                "Dinner: Stir-fried spinach tofu\n"
                "Shopping List: (optional) olive oil, garlic"
            ),
            "normalized": normalize_request(user_text),
            "stub": True,
        }

    # Cloud / Vertex AI mode
    if model is None:
        raise HTTPException(status_code=500, detail="Model not initialized (DEBUG_STUB=0, PROJECT_ID set).")

    prompt = f"""
You are a meal planning assistant.
Goal: generate a 1-day meal plan (breakfast, lunch, dinner) based on ingredients (most important) and optional constraints.
If constraints are missing, assume reasonable defaults.

First, if needed, call normalize_request(user_text=...) to extract ingredients.
Then produce the final plan.

User input:
{user_text}
""".strip()

    first = model.generate_content(prompt, tools=[tool])

    parts = first.candidates[0].content.parts if first.candidates else []
    if parts and hasattr(parts[0], "function_call") and parts[0].function_call:
        fc = parts[0].function_call
        if fc.name != "normalize_request":
            raise HTTPException(status_code=500, detail=f"Unexpected function call: {fc.name}")

        args = dict(fc.args)
        tool_out = normalize_request(user_text=args.get("user_text", ""))

        second = model.generate_content(
            [
                Content(role="user", parts=[Part.from_text(prompt)]),
                Content(role="function", parts=[Part.from_dict({"function_call": {"name": "normalize_request"}})]),
                Content(
                    role="function",
                    parts=[
                        Part.from_function_response(
                            name="normalize_request",
                            response={"content": json.dumps(tool_out)},
                        )
                    ],
                ),
            ],
            tools=[tool],
        )

        text_out = second.candidates[0].content.parts[0].text if second.candidates else ""
        return {"result": text_out, "normalized": tool_out, "stub": False}

    text_out = first.text or ""
    return {"result": text_out, "normalized": None, "config_loaded": bool(SECRET_CONFIG_TEXT), "stub": False}




















# import json
# import os
# from typing import Any, Dict, Optional

# from fastapi import FastAPI, HTTPException
# from fastapi.responses import HTMLResponse
# from fastapi.staticfiles import StaticFiles
# from pydantic import BaseModel

# import vertexai
# from vertexai.generative_models import (
#     Content,
#     FunctionDeclaration,
#     GenerativeModel,
#     Part,
#     Tool,
# )

# # Optional (EC3): Secret Manager
# def read_secret_payload(secret_resource: str) -> Optional[str]:
#     """
#     secret_resource format:
#       projects/PROJECT_ID/secrets/SECRET_NAME/versions/latest
#     """
#     try:
#         from google.cloud import secretmanager
#         client = secretmanager.SecretManagerServiceClient()
#         resp = client.access_secret_version(request={"name": secret_resource})
#         return resp.payload.data.decode("utf-8")
#     except Exception:
#         return None


# # -----------------------------
# # Config
# # -----------------------------
# # Local debug switch:
# # - DEBUG_STUB=1 (default): do NOT require GCP env vars; return stub output for local UI/API debugging.
# # - DEBUG_STUB=0: enable Vertex AI Gemini calls (requires PROJECT_ID + ADC credentials locally or Cloud Run).
# DEBUG_STUB = os.getenv("DEBUG_STUB", "1") == "1"

# PROJECT_ID = os.getenv("PROJECT_ID")
# LOCATION = os.getenv("LOCATION", "us-central1")
# MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.0-flash-001")

# # Optional: pull extra config from Secret Manager (EC3 demo)
# # e.g. SECRET_CONFIG = projects/.../secrets/MEAL_PLANNER_CONFIG/versions/latest
# SECRET_CONFIG = os.getenv("SECRET_CONFIG")
# SECRET_CONFIG_TEXT = read_secret_payload(SECRET_CONFIG) if SECRET_CONFIG else None

# # --- Cloud / Vertex AI init (commented-out behavior preserved) ---
# # In Cloud Run (or local with ADC), set:
# #   DEBUG_STUB=0
# #   PROJECT_ID=<your project id>
# #   LOCATION=us-central1
# #   MODEL_NAME=gemini-2.0-flash-001
# #
# # If you want local auth for Vertex AI:
# #   gcloud auth application-default login
# #
# # Original strict check (kept for cloud mode only):
# # if not PROJECT_ID:
# #     raise RuntimeError("Missing env var PROJECT_ID")
# #
# # vertexai.init(project=PROJECT_ID, location=LOCATION)
# # model = GenerativeModel(MODEL_NAME)

# # model = None
# # if not DEBUG_STUB:
# #     if not PROJECT_ID:
# #         raise RuntimeError("Missing env var PROJECT_ID")
# #     vertexai.init(project=PROJECT_ID, location=LOCATION)
# #     model = GenerativeModel(MODEL_NAME)

# model = None
# if not DEBUG_STUB:
#     if not PROJECT_ID:
#         # Don't crash the container; fall back to stub mode.
#         DEBUG_STUB = True
#     else:
#         vertexai.init(project=PROJECT_ID, location=LOCATION)
#         model = GenerativeModel(MODEL_NAME)


# # -----------------------------
# # Tool (EC2)
# # -----------------------------
# normalize_request_func = FunctionDeclaration(
#     name="normalize_request",
#     description=(
#         "Normalize the user's free-form input into a structured request. "
#         "Extract ingredients (required) and optional constraints (dietary, meals, cooking_time). "
#         "If ingredients are missing, return an error field explaining what's missing."
#     ),
#     parameters={
#         "type": "object",
#         "properties": {
#             "user_text": {
#                 "type": "string",
#                 "description": "Raw user text describing ingredients and optional constraints.",
#             }
#         },
#         "required": ["user_text"],
#     },
# )

# tool = Tool(function_declarations=[normalize_request_func])


# def normalize_request(user_text: str) -> Dict[str, Any]:
#     """
#     Minimal deterministic tool:
#     - Tries to extract ingredients from patterns like 'ingredients: ...' or by comma-separated list.
#     - Everything else stays as notes.
#     This is intentionally simple to keep implementation minimal.
#     """
#     text = (user_text or "").strip()
#     lower = text.lower()

#     ingredients = []
#     if "ingredients:" in lower:
#         after = text[lower.index("ingredients:") + len("ingredients:"):].strip()
#         after = after.splitlines()[0].strip()
#         ingredients = [x.strip() for x in after.split(",") if x.strip()]
#     else:
#         if "," in text and len(text) <= 200:
#             ingredients = [x.strip() for x in text.split(",") if x.strip()]

#     if not ingredients:
#         return {
#             "error": "Missing ingredients. Please include 'ingredients: item1, item2, ...' or a comma-separated list.",
#             "ingredients": [],
#             "notes": text,
#         }

#     return {
#         "ingredients": ingredients,
#         "notes": text,
#     }


# # -----------------------------
# # App
# # -----------------------------
# app = FastAPI()

# # Serve your existing index.html (put it under ./static/index.html)
# app.mount("/static", StaticFiles(directory="static"), name="static")


# @app.get("/", response_class=HTMLResponse)
# def home():
#     try:
#         with open("static/index.html", "r", encoding="utf-8") as f:
#             return f.read()
#     except FileNotFoundError:
#         return "<h1>Meal Planner</h1><p>Missing static/index.html</p>"


# class MenuRequest(BaseModel):
#     input: str


# @app.post("/api/menu")
# def generate_menu(req: MenuRequest):
#     user_text = (req.input or "").strip()
#     if not user_text:
#         raise HTTPException(status_code=400, detail="Empty input")

#     # Local-only stub output (no GCP required)
#     if DEBUG_STUB:
#         # Minimal deterministic output so you can debug the UI + API end-to-end locally.
#         return {
#             "result": (
#                 "Breakfast: Simple oatmeal (if allowed)\n"
#                 "Lunch: Tofu + tomato pasta\n"
#                 "Dinner: Stir-fried spinach tofu\n"
#                 "Shopping List: (optional) olive oil, garlic"
#             ),
#             "normalized": normalize_request(user_text),
#             "stub": True,
#         }

#     # Cloud / Vertex AI mode
#     if model is None:
#         raise HTTPException(status_code=500, detail="Model not initialized (set DEBUG_STUB=0 and PROJECT_ID)")

#     prompt = f"""
# You are a meal planning assistant.
# Goal: generate a 1-day meal plan (breakfast, lunch, dinner) based on ingredients (most important) and optional constraints.
# If constraints are missing, assume reasonable defaults.

# First, if needed, call normalize_request(user_text=...) to extract ingredients.
# Then produce the final plan.

# User input:
# {user_text}
# """.strip()

#     first = model.generate_content(prompt, tools=[tool])

#     parts = first.candidates[0].content.parts if first.candidates else []
#     if parts and hasattr(parts[0], "function_call") and parts[0].function_call:
#         fc = parts[0].function_call
#         if fc.name != "normalize_request":
#             raise HTTPException(status_code=500, detail=f"Unexpected function call: {fc.name}")

#         args = dict(fc.args)
#         tool_out = normalize_request(user_text=args.get("user_text", ""))

#         # Provide tool response back to model (pattern from Google codelab)
#         second = model.generate_content(
#             [
#                 Content(role="user", parts=[Part.from_text(prompt)]),
#                 Content(role="function", parts=[Part.from_dict({"function_call": {"name": "normalize_request"}})]),
#                 Content(
#                     role="function",
#                     parts=[
#                         Part.from_function_response(
#                             name="normalize_request",
#                             response={"content": json.dumps(tool_out)},
#                         )
#                     ],
#                 ),
#             ],
#             tools=[tool],
#         )

#         text_out = second.candidates[0].content.parts[0].text if second.candidates else ""
#         return {"result": text_out, "normalized": tool_out}

#     text_out = first.text or ""
#     return {"result": text_out, "normalized": None, "config_loaded": bool(SECRET_CONFIG_TEXT)}

# import os
# from typing import Optional

# from fastapi import FastAPI, HTTPException
# from fastapi.responses import HTMLResponse
# from fastapi.staticfiles import StaticFiles
# from pydantic import BaseModel

# from google import genai
# from google.cloud import secretmanager


# APP_TITLE = "Meal Planner"
# DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")  # 便宜快，够作业用


# class MenuRequest(BaseModel):
#     input: str


# def get_secret_from_secret_manager(secret_resource_name: str) -> str:
#     """
#     secret_resource_name 示例：
#       projects/<PROJECT_NUMBER>/secrets/<SECRET_NAME>/versions/latest
#     """
#     client = secretmanager.SecretManagerServiceClient()
#     resp = client.access_secret_version(request={"name": secret_resource_name})
#     return resp.payload.data.decode("utf-8").strip()


# def get_api_key() -> str:
#     # 方式 A（推荐）：从 Secret Manager 读
#     secret_name = os.getenv("GEMINI_API_KEY_SECRET")
#     if secret_name:
#         return get_secret_from_secret_manager(secret_name)

#     # 方式 B（更省事）：直接用环境变量
#     key = os.getenv("GEMINI_API_KEY")
#     if key:
#         return key.strip()

#     raise RuntimeError(
#         "Missing API key. Set GEMINI_API_KEY_SECRET (recommended) or GEMINI_API_KEY."
#     )


# def build_prompt(user_text: str) -> str:
#     # 简单、稳定输出：让模型返回 JSON，方便前端展示/你 report 写 evaluation
#     return f"""
# You are a meal-planning assistant.
# Generate a ONE-DAY meal plan based on the user's constraints and ingredients.

# Return STRICT JSON with this schema:
# {{
#   "breakfast": {{"title": "...", "ingredients": ["..."], "steps": ["..."], "time_minutes": 0}},
#   "lunch":     {{"title": "...", "ingredients": ["..."], "steps": ["..."], "time_minutes": 0}},
#   "dinner":    {{"title": "...", "ingredients": ["..."], "steps": ["..."], "time_minutes": 0}},
#   "shopping_list": ["..."],
#   "notes": ["..."]
# }}

# Rules:
# - Respect dietary constraints (allergies, vegan, nut-free, etc.).
# - Prefer using provided ingredients.
# - Keep steps short and practical.
# - If missing ingredients, put them in shopping_list.

# User input:
# {user_text}
# """.strip()


# app = FastAPI(title=APP_TITLE)

# # 静态前端：/ -> static/index.html
# app.mount("/static", StaticFiles(directory="static"), name="static")


# @app.get("/", response_class=HTMLResponse)
# def root():
#     try:
#         with open("static/index.html", "r", encoding="utf-8") as f:
#             return f.read()
#     except FileNotFoundError:
#         return "<h1>Missing static/index.html</h1>"


# @app.post("/api/menu")
# def menu(req: MenuRequest):
#     text = (req.input or "").strip()
#     if not text:
#         raise HTTPException(status_code=400, detail="input is required")

#     # 基础防滥用：限制输入长度（Cloud Run 上也能避免被乱打）
#     if len(text) > 6000:
#         raise HTTPException(status_code=400, detail="input too long (max 6000 chars)")

#     try:
#         api_key = get_api_key()
#         client = genai.Client(api_key=api_key)

#         prompt = build_prompt(text)

#         resp = client.models.generate_content(
#             model=DEFAULT_MODEL,
#             contents=prompt,
#         )

#         # 直接返回文本（JSON 字符串），你的前端会按 content-type 处理
#         # 这里我们返回 application/json 更方便前端直接格式化显示
#         return resp.text

#     except HTTPException:
#         raise
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


