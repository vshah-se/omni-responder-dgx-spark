import os

# Tests exercise the full orchestrator, which now includes the Telegram notifier.
# Force it off for the whole suite so test runs never send real messages.
os.environ["TELEGRAM_DISABLED"] = "1"
