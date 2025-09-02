import os
import asyncio
import logging
from transformers import pipeline, AutoTokenizer
from twikit import Client

# ========================
# Config
# ========================
QUERY = "India"
LIMIT = 20
COOKIE_FILE = "cookies.json"
DRY_RUN = bool(os.environ.get("DRY_RUN", 0))

# ========================
# Logging setup
# ========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("tweet-bot")

# ========================
# Summarizer setup
# ========================
log.info("🤖 Loading summarizer...")
MODEL_NAME = "facebook/bart-large-cnn"
summarizer = pipeline("summarization", model=MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def chunk_text(text, max_tokens=900):
    tokens = tokenizer.encode(text, truncation=False)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk = tokens[i:i+max_tokens]
        chunks.append(tokenizer.decode(chunk, skip_special_tokens=True))
    return chunks


def summarize_tweets(tweets):
    combined = " ".join(tweets)
    log.info("📝 Summarizing tweets in chunks...")

    chunks = chunk_text(combined, max_tokens=900)
    partial_summaries = []

    for idx, chunk in enumerate(chunks, 1):
        log.info(f"⏳ Summarizing chunk {idx}/{len(chunks)}")
        s = summarizer(chunk, max_length=80, min_length=25, do_sample=False)[0]["summary_text"]
        partial_summaries.append(s)

    # Final summary from partials
    final_input = " ".join(partial_summaries)
    final_summary = summarizer(final_input, max_length=120, min_length=40, do_sample=False)[0]["summary_text"]

    # Trim for tweet
    tweet = final_summary[:277] + "…" if len(final_summary) > 280 else final_summary
    return tweet


# ========================
# Fetch tweets
# ========================
async def fetch_with_twikit(client: Client, query=QUERY, limit=LIMIT):
    """Fetch tweets using twikit with logged-in cookies."""
    log.info("🔍 Searching for tweets with query: %s", query)
    try:
        results = await client.search_tweet(query, product="Latest", count=limit)
        tweets = [t.text for t in results]
        log.info("✅ twikit returned %d tweets", len(tweets))
        return tweets
    except Exception as e:
        log.error("❌ twikit search failed: %s", e)
        return []


# ========================
# Main Bot
# ========================
async def main():
    log.info("🚀 Starting Tweet Bot")
    client = Client("en-US")

    # Load cookies
    if os.path.exists(COOKIE_FILE):
        client.load_cookies(COOKIE_FILE)
        log.info("🍪 Cookies loaded from %s", COOKIE_FILE)
    else:
        log.error("❌ No cookies.json found. Please login once and save cookies!")
        return

    # Fetch tweets
    tweets = await fetch_with_twikit(client)
    if not tweets:
        log.warning("⚠️ No tweets found")
        return

    # Summarize
    summary = summarize_tweets(tweets)
    log.info(f"📝 Final summary tweet: {summary}")

    # Post tweet
    if DRY_RUN:
        log.info("💤 DRY_RUN enabled. Would post: %s", summary[:100])
    else:
        try:
            await client.create_tweet(text=summary)
            log.info("✅ Successfully posted summary tweet")
        except Exception as e:
            log.error("❌ Failed to post tweet: %s", e)


if __name__ == "__main__":
    asyncio.run(main())
