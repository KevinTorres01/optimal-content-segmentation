from src.core.env import load_env

# Load .env once, as early as possible: importing any src.core submodule runs
# this first, so API keys are available before any provider reads os.environ.
load_env()
