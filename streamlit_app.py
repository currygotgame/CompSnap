import os
import re
import json
import time
import mimetypes

import requests
import streamlit as st
from google import genai
from google.genai import types

# ==============================================================
# Config — reads keys from Streamlit secrets (set these when you
# deploy, never commit real keys to GitHub)
# ==============================================================
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
SOLDCOMPS_API_KEY = st.secrets.get("SOLDCOMPS_API_KEY", "")

GEMINI_MODEL = "gemini-3.5-flash-lite"
SOLDCOMPS_URL = "https://api.sold-comps.com/v1/scrape"

client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None


# ==============================================================
# Helpers (same logic as the Colab version)
# ==============================================================
def _mime_type(name_or_url):
    mime, _ = mimetypes.guess_type(name_or_url)
    return mime or "image/jpeg"

def _download_image(url):
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    return r.content, _mime_type(url)

def _extract_json_array(text):
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return []
    try:
        return json.loads(match.group(0))
    except (json.JSONDecodeError, ValueError):
        return []

def _call_gemini(contents, retries=3):
    for attempt in range(retries):
        try:
            return client.models.generate_content(model=GEMINI_MODEL, contents=contents)
        except Exception as e:
            if "429" in str(e) and attempt < retries - 1:
                time.sleep(30)
                continue
            raise


# ==============================================================
# Pipeline functions — take raw image bytes instead of a file path,
# since Streamlit gives us an uploaded file in memory, not a path
# on disk like Colab did
# ==============================================================
def identify_item(image_bytes, mime):
    prompt = (
        "You are looking at ONE item that will be sold on eBay. "
        "Identify it as specifically as possible: brand, product line, "
        "model or style name, material, color, and any other detail that "
        "would distinguish it from similar items. "
        "Respond with ONLY the item name, nothing else — written like a "
        "real eBay search query, under 12 words."
    )
    response = _call_gemini([
        types.Part.from_bytes(data=image_bytes, mime_type=mime),
        prompt,
    ])
    return response.text.strip()


def get_sold_comps(keyword, count=40, ebay_site="ebay.com"):
    headers = {"Authorization": f"Bearer {SOLDCOMPS_API_KEY}"}
    params = {"keyword": keyword, "ebaySite": ebay_site, "count": count}
    resp = requests.get(SOLDCOMPS_URL, headers=headers, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json().get("items", [])


def filter_same_item(image_bytes, mime, comps, item_description, batch_size=25):
    matches = []

    for start in range(0, len(comps), batch_size):
        batch = comps[start:start + batch_size]

        parts = [
            f"Item in the reference photo: {item_description}\n\n"
            "Below is the reference photo, followed by numbered candidate "
            "listing photos. For each candidate, decide if it shows the "
            "SAME exact item as the reference — same product, same design "
            "or model — not just the same category or a similar-looking "
            "item.\n\nReference photo:",
            types.Part.from_bytes(data=image_bytes, mime_type=mime),
        ]

        shown_to_batch_idx = {}
        shown_num = 0

        for i, comp in enumerate(batch):
            thumb_url = comp.get("thumbnailUrl")
            if not thumb_url:
                continue
            try:
                thumb_bytes, thumb_mime = _download_image(thumb_url)
            except Exception:
                continue

            parts.append(f"\nCandidate {shown_num}: {comp.get('title', '')}")
            parts.append(types.Part.from_bytes(data=thumb_bytes, mime_type=thumb_mime))
            shown_to_batch_idx[shown_num] = i
            shown_num += 1

        if not shown_to_batch_idx:
            continue

        parts.append(
            "\n\nRespond with ONLY a JSON array of the candidate numbers "
            "that are the SAME exact item. Example: [0, 2]. "
            "If none match, respond with []."
        )

        response = _call_gemini(parts)
        matched_nums = _extract_json_array(response.text)

        for num in matched_nums:
            if num in shown_to_batch_idx:
                matches.append(batch[shown_to_batch_idx[num]])

    return matches


def compute_stats(matches, buy_under_margin=0.30):
    prices = []
    for m in matches:
        try:
            prices.append(float(m["soldPrice"]))
        except (TypeError, ValueError, KeyError):
            continue

    if not prices:
        return None

    avg = sum(prices) / len(prices)
    return {
        "count": len(prices),
        "average": round(avg, 2),
        "min": round(min(prices), 2),
        "max": round(max(prices), 2),
        "buy_under": round(avg * (1 - buy_under_margin), 2),
    }


# ==============================================================
# Streamlit UI
# ==============================================================
st.set_page_config(page_title="Estate Sale Comp Finder", page_icon="🏷️")
st.title("🏷️ Estate Sale Comp Finder")
st.caption("Upload a photo of one item — get the exact eBay sold comps for it.")

if not GEMINI_API_KEY or not SOLDCOMPS_API_KEY:
    st.error("Missing API keys. Add GEMINI_API_KEY and SOLDCOMPS_API_KEY in your app's Secrets.")
    st.stop()
if "uses" not in st.session_state:
    st.session_state.uses = 0

if st.session_state.uses >= 3:
    st.warning("Demo limit reached (3 tries). Come back tomorrow!")
    st.stop()
uploaded_file = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Your photo", width=300)

    if st.button("Find sold comps", type="primary"):
        st.session_state.uses += 1
        image_bytes = uploaded_file.getvalue()
        mime = uploaded_file.type or "image/jpeg"

        with st.spinner("Identifying item..."):
            item_name = identify_item(image_bytes, mime)
        st.success(f"Identified as: **{item_name}**")

        with st.spinner("Pulling sold comps from eBay..."):
            comps = get_sold_comps(item_name)

        if not comps:
            st.warning("No exact sold match found.")
            st.stop()

        st.write(f"Got {len(comps)} raw comps — checking which ones are the same item...")

        with st.spinner("Filtering to same-item matches..."):
            matches = filter_same_item(image_bytes, mime, comps, item_name)

        if not matches:
            st.warning("No exact sold match found.")
            st.stop()

        stats = compute_stats(matches)

        st.subheader(f"{len(matches)} matching sold listings")

        col1, col2, col3 = st.columns(3)
        col1.metric("Average", f"${stats['average']}")
        col2.metric("Range", f"${stats['min']} - ${stats['max']}")
        col3.metric("Buy under", f"${stats['buy_under']}")

        for m in matches:
            st.markdown(
                f"**[{m.get('title')}]({m.get('url')})**  \n"
                f"${m.get('soldPrice')} · sold {m.get('endedAt')}"
            )
