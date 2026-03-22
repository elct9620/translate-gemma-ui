import logging
import os

level = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, level, logging.INFO))

from translate_gemma_ui import app  # noqa: E402

if __name__ == "__main__":
    app.launch()
