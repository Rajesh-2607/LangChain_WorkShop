# config.py
# ------------------------------------------------------------------
# Loads GOOGLE_API_KEY from a local .env file. Import this ONCE at
# the top of any script, before creating a ChatGoogleGenerativeAI
# instance, so the SDK can auto-discover the key.
# ------------------------------------------------------------------
import os
from dotenv import load_dotenv

load_dotenv()  # reads .env from the current working directory

if not os.getenv("GOOGLE_API_KEY"):
    raise EnvironmentError(
        "GOOGLE_API_KEY not found. Add it to a .env file or export it "
        "in your shell before running this script."
    )