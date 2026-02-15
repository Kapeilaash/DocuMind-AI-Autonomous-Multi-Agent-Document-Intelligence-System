"""
Run the Document Intelligence Agent server.
Ensure .env exists with OPENAI_API_KEY before running.
"""

import os
import sys

if not os.getenv("OPENAI_API_KEY"):
    print("ERROR: OPENAI_API_KEY not set.")
    print("Create a .env file with: OPENAI_API_KEY=sk-your-key")
    sys.exit(1)

import uvicorn

uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
