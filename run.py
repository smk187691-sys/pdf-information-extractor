import os
from dotenv import load_dotenv
from app.application import app

load_dotenv()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    host = os.environ.get("HOST", "0.0.0.0")
    debug = os.environ.get("FLASK_DEBUG", "false").lower() in ("true", "1", "yes")
    print(f"Starting Universal Document RAG Assistant on http://{host}:{port}")
    app.run(host=host, port=port, debug=debug, use_reloader=False)
