"""
Feature module for the RetailNext Smart Store omnichannel e-commerce prototype.
Implements semantic catalog search, deterministic event routing, omnichannel inventory
fallback recommendations, and multimodal (text, image, voice) stylist agent orchestration.
"""
import base64
import json
import os
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
import numpy as np
from openai import OpenAI
import pandas as pd
import streamlit as st

from inventory import check_local_inventory

load_dotenv()

CUSTOM_CSS = """
<style>
    .brand-logo {
        background: linear-gradient(90deg, #FF416C 0%, #FF4B2B 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 900;
        font-size: 2.2rem;
        letter-spacing: -1px;
        margin: 0;
        padding: 0;
    }
    div[data-testid="stVerticalBlock"] > div[style*="border"] {
        border-radius: 12px !important;
        box-shadow: 0 4px 10px -2px rgba(0, 0, 0, 0.1) !important;
        border: 1px solid #E2E8F0 !important;
        padding: 15px !important;
        background-color: white !important;
        transition: transform 0.2s ease-in-out;
    }
    div[data-testid="stVerticalBlock"] > div[style*="border"]:hover {
        transform: translateY(-5px);
    }
    .price-tag { font-size: 1.3rem; font-weight: 800; color: #111827; margin-bottom: 10px; }
    .product-title { height: 45px; overflow: hidden; text-overflow: ellipsis; font-weight: 600; color: #374151;}
</style>
"""

TOOLS_SCHEMA: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_catalog",
            "description": "Search the database for individual clothing items.",
            "parameters": {
                "type": "object",
                "properties": {
                    "search_query": {
                        "type": "string",
                        "description": "The style or clothing item to search for.",
                    }
                },
                "required": ["search_query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_outfit",
            "description": "Create a complete outfit for a specific event.",
            "parameters": {
                "type": "object",
                "properties": {
                    "event_type": {
                        "type": "string",
                        "description": "The occasion the user is attending.",
                    }
                },
                "required": ["event_type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_local_inventory",
            "description": "Check stock across all nearby store locations.",
            "parameters": {
                "type": "object",
                "properties": {"item_id": {"type": "string"}},
                "required": ["item_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_to_cart",
            "description": "Add an item to the shopping cart.",
            "parameters": {
                "type": "object",
                "properties": {"item_id": {"type": "string"}},
                "required": ["item_id"],
            },
        },
    },
]


def get_openai_api_key() -> Optional[str]:
    """
    Retrieves the OpenAI API key from environment variables or Streamlit secrets.

    Returns:
        Optional[str]: API key string or None if not configured.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets.get("OPENAI_API_KEY")
        except Exception:
            pass
    return api_key


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """
    Computes cosine similarity between two numeric vectors using NumPy.

    Args:
        a (List[float]): First vector.
        b (List[float]): Second vector.

    Returns:
        float: Cosine similarity score.
    """
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def detect_event_intent(text: str) -> Optional[str]:
    """
    Deterministic intent router that detects core event occasions in user input.

    Args:
        text (str): User query text.

    Returns:
        Optional[str]: Detected event name or None.
    """
    events = ["wedding", "office", "party", "vacation", "workout", "date", "interview"]
    text_lower = text.lower()
    for event in events:
        if event in text_lower:
            return event
    return None


@st.cache_data(show_spinner="Curating your personalized catalog...")
def load_and_embed_catalog(api_key: str) -> Tuple[Dict[str, Any], Dict[str, List[float]]]:
    """
    Loads `sample_styles.csv` and generates `text-embedding-3-small` vectors for each product.

    Args:
        api_key (str): OpenAI API key.

    Returns:
        Tuple[Dict[str, Any], Dict[str, List[float]]]: Product catalog and embeddings index.
    """
    client = OpenAI(api_key=api_key)
    df = pd.read_csv("sample_styles.csv").head(120)
    catalog: Dict[str, Any] = {}
    embeddings: Dict[str, List[float]] = {}

    for _, row in df.iterrows():
        item_id = str(row["id"])
        rich_desc = (
            f"{row['baseColour']} {row['masterCategory']} {row['subCategory']} - "
            f"{row['productDisplayName']}. Usage: {row['usage']} for {row['gender']}."
        )

        catalog[item_id] = {
            "id": item_id,
            "desc": rich_desc,
            "name": row["productDisplayName"],
            "price": int(np.random.randint(40, 150)),
            "image": f"sample_images/{item_id}.jpg",
            "gender": row.get("gender", "Women"),
        }
        res = client.embeddings.create(input=rich_desc, model="text-embedding-3-small")
        embeddings[item_id] = res.data[0].embedding

    return catalog, embeddings


def perform_rag_search(
    client: OpenAI,
    catalog_embeddings: Dict[str, List[float]],
    user_query: str,
    top_k: int = 16,
) -> List[str]:
    """
    Performs semantic similarity search across the embedded product catalog.

    Args:
        client (OpenAI): Initialized OpenAI client.
        catalog_embeddings (Dict[str, List[float]]): Product embedding index.
        user_query (str): Search query string.
        top_k (int): Number of top matching item IDs to return.

    Returns:
        List[str]: Ranked list of product IDs.
    """
    query_response = client.embeddings.create(
        input=user_query, model="text-embedding-3-small"
    )
    query_embedding = query_response.data[0].embedding

    similarities = [
        (cosine_similarity(query_embedding, emb), i_id)
        for i_id, emb in catalog_embeddings.items()
    ]
    similarities.sort(key=lambda x: x[0], reverse=True)
    return [item_id for _, item_id in similarities[:top_k]]


def render_retailnext_app() -> None:
    """
    Renders the RetailNext Smart Store Streamlit interface and AI Stylist agent loop.
    """
    st.set_page_config(
        page_title="RetailNext | Smart Store", page_icon="🛍️", layout="wide"
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    api_key = get_openai_api_key()
    if not api_key:
        st.error(
            "OpenAI API Key not found. Please set `OPENAI_API_KEY` in `.env` "
            "or `.streamlit/secrets.toml`."
        )
        st.stop()

    client = OpenAI(api_key=api_key)

    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "view" not in st.session_state:
        st.session_state.view = "grid"
    if "cart" not in st.session_state:
        st.session_state.cart = []
    if "grid_view" not in st.session_state:
        st.session_state.grid_view = []
    if "processed_media" not in st.session_state:
        st.session_state.processed_media = set()
    if "audio_key" not in st.session_state:
        st.session_state.audio_key = 0
    if "audio_to_play" not in st.session_state:
        st.session_state.audio_to_play = None

    if "user_profile" not in st.session_state:
        st.session_state.user_profile = {
            "name": "Sarah",
            "gender": "Women",
            "tier": "VIP Platinum",
            "preferred_style": "Casual Chic & Elegant",
            "favorite_colors": ["Navy Blue", "Black", "White"],
            "avoid": "Neon colors, heavy wool",
        }

    if "messages" not in st.session_state:
        crm = st.session_state.user_profile
        st.session_state.messages = [
            {
                "role": "system",
                "content": (
                    f"You are a warm, fashionable AI Concierge for RetailNext. "
                    f"Treat the user like a VIP.\n\n"
                    f"[CRM DATA: Name: {crm['name']}, Gender: {crm['gender']}, "
                    f"Tier: {crm['tier']}, Style: {crm['preferred_style']}, "
                    f"Loves: {', '.join(crm['favorite_colors'])}, Avoids: {crm['avoid']}].\n"
                    f"Proactively personalize recommendations based on this CRM data.\n\n"
                    f"ONLY use the 'search_catalog' or 'generate_outfit' tools if they "
                    f"explicitly ask to see clothes or outfits. CRITICAL: When using "
                    f"'add_to_cart' or 'check_local_inventory', use the exact numeric 'item_id'."
                ),
            }
        ]

    if not st.session_state.logged_in:
        _, col2, _ = st.columns([1, 2, 1])
        with col2:
            st.write("\n\n\n\n")
            st.markdown(
                "<h1 style='text-align: center;' class='brand-logo'>RetailNext</h1>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<p style='text-align: center; color: #64748B; font-size: 1.2rem;'>"
                "The future of omnichannel fashion. Powered By OpenAI.</p><br>",
                unsafe_allow_html=True,
            )
            if st.button("Secure VIP Login", type="primary", use_container_width=True):
                st.session_state.logged_in = True
                st.rerun()
        st.stop()

    product_catalog, catalog_embeddings = load_and_embed_catalog(api_key)

    if not st.session_state.grid_view:
        target_gender = st.session_state.user_profile["gender"]
        curated_keys = [
            k
            for k, v in product_catalog.items()
            if v.get("gender", "Women") == target_gender
        ]
        st.session_state.grid_view = (
            curated_keys[:16] if curated_keys else list(product_catalog.keys())[:16]
        )

    def execute_tool(tool_call: Any) -> str:
        args = json.loads(tool_call.function.arguments)

        if tool_call.function.name == "search_catalog":
            results = perform_rag_search(
                client, catalog_embeddings, args.get("search_query", ""), top_k=16
            )
            st.session_state.grid_view = results
            st.session_state.view = "grid"
            top_items = [
                {"item_id": item_id, "name": product_catalog[item_id]["name"]}
                for item_id in results[:4]
            ]
            return json.dumps(
                {
                    "status": "success",
                    "message": "Grid updated.",
                    "top_matches": top_items,
                }
            )

        if tool_call.function.name == "generate_outfit":
            event = args.get("event_type", "")
            enhanced_query = (
                f"{event} outfit {st.session_state.user_profile['preferred_style']}"
            )
            results = perform_rag_search(
                client, catalog_embeddings, enhanced_query, top_k=16
            )
            st.session_state.grid_view = results
            st.session_state.view = "grid"
            top_items = [
                {"item_id": i_id, "name": product_catalog[i_id]["name"]}
                for i_id in results[:6]
            ]
            return json.dumps(
                {
                    "status": "success",
                    "instruction": (
                        f"Select 2 to 3 items from these matches to build an outfit for a "
                        f"'{event}'. CRITICAL: You must explicitly explain to the user WHY "
                        f"these items work well together for this event, referencing their "
                        f"CRM style preferences."
                    ),
                    "matches": top_items,
                }
            )

        if tool_call.function.name == "check_local_inventory":
            item_id = str(args.get("item_id"))
            inventory_res_str = check_local_inventory(item_id)
            inventory_data = json.loads(inventory_res_str)

            total_stock = sum(inventory_data["stock_by_location"].values())
            if total_stock == 0 and item_id in product_catalog:
                item_desc = product_catalog[item_id]["desc"]
                similar = perform_rag_search(
                    client, catalog_embeddings, item_desc, top_k=4
                )
                similar = [sid for sid in similar if sid != item_id][:3]
                fallback_items = [
                    {"item_id": sid, "name": product_catalog[sid]["name"]}
                    for sid in similar
                ]
                inventory_data["fallback_recommendations"] = fallback_items
                inventory_data["instruction"] = (
                    "Item is out of stock everywhere. Apologize and highly recommend "
                    "these specific fallback alternatives to save the sale."
                )

            return json.dumps(inventory_data)

        if tool_call.function.name == "add_to_cart":
            item_id = str(args.get("item_id"))
            if item_id not in product_catalog:
                return json.dumps({"error": f"Item ID {item_id} not found."})

            inventory_res_str = check_local_inventory(item_id)
            inventory_data = json.loads(inventory_res_str)
            total_stock = sum(inventory_data["stock_by_location"].values())

            if total_stock == 0:
                item_desc = product_catalog[item_id]["desc"]
                similar = perform_rag_search(
                    client, catalog_embeddings, item_desc, top_k=4
                )
                similar = [sid for sid in similar if sid != item_id][:3]
                fallback_items = [
                    {"item_id": sid, "name": product_catalog[sid]["name"]}
                    for sid in similar
                ]
                return json.dumps(
                    {
                        "status": "blocked",
                        "error": "Out of stock",
                        "instruction": (
                            "CRITICAL: DO NOT say the item was added. It was NOT added to "
                            "the cart because it is out of stock. Apologize to the user and "
                            "proactively recommend these specific fallback alternatives to "
                            "save the sale."
                        ),
                        "fallback_recommendations": fallback_items,
                    }
                )

            st.session_state.cart.append(product_catalog[item_id])
            return json.dumps(
                {
                    "status": "success",
                    "message": f"Successfully added {product_catalog[item_id]['name']} to cart.",
                }
            )

        return json.dumps({"error": "Unknown tool call"})

    nav1, nav2, nav3 = st.columns([1, 3, 1])
    with nav1:
        if st.button("RetailNext", use_container_width=True):
            st.session_state.grid_view = list(product_catalog.keys())[:16]
            st.session_state.view = "grid"
            st.rerun()
    with nav2:
        search_query = st.text_input(
            "🔍",
            placeholder="Semantic Search: Try 'A breezy outfit'...",
            label_visibility="collapsed",
        )
        if search_query:
            st.session_state.grid_view = perform_rag_search(
                client, catalog_embeddings, search_query
            )
            st.session_state.view = "grid"
    with nav3:
        total_price = sum(item["price"] for item in st.session_state.cart)
        if st.button(
            f"🛒 Cart ({len(st.session_state.cart)}) | ${total_price}",
            type="primary",
            use_container_width=True,
        ):
            st.session_state.view = "cart"
            st.rerun()

    st.divider()

    if st.session_state.view == "grid":
        st.markdown(f"### ✨ Curated for {st.session_state.user_profile['name']}")
        cols = st.columns(4)
        for index, item_id in enumerate(st.session_state.grid_view):
            item = product_catalog.get(item_id)
            if not item:
                continue

            with cols[index % 4]:
                with st.container(border=True):
                    if os.path.exists(item["image"]):
                        st.image(item["image"], use_container_width=True)
                    st.markdown(
                        f"<div class='product-title'>{item['name']}</div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"<div class='price-tag'>${item['price']}</div>",
                        unsafe_allow_html=True,
                    )

                    btn1, btn2 = st.columns(2)
                    with btn1:
                        if st.button(
                            "🛒 Add", key=f"add_{item_id}", use_container_width=True
                        ):
                            st.session_state.cart.append(item)
                            st.rerun()
                    with btn2:
                        if st.button(
                            "💬 Ask", key=f"ask_{item_id}", use_container_width=True
                        ):
                            st.session_state.messages.append(
                                {
                                    "role": "system",
                                    "content": f"[System: User clicked 'Ask' on item {item_id}.]",
                                }
                            )
                            st.session_state.messages.append(
                                {
                                    "role": "assistant",
                                    "content": (
                                        f"I see you're looking at the **{item['name']}**! "
                                        f"Need me to check local inventory?"
                                    ),
                                }
                            )

    elif st.session_state.view == "cart":
        st.markdown("### 🛒 Your Shopping Cart")
        if not st.session_state.cart:
            st.info("Your cart is currently empty.")
        else:
            c1, c2 = st.columns([2, 1])
            with c1:
                for item in st.session_state.cart:
                    st.markdown(f"- **{item['name']}** : ${item['price']}")
            with c2:
                st.markdown(f"### Total: ${total_price}")
                if st.button(
                    "Proceed to Secure Checkout",
                    type="primary",
                    use_container_width=True,
                ):
                    st.session_state.view = "checkout"
                    st.rerun()

    elif st.session_state.view == "checkout":
        st.markdown("### 📦 Secure Checkout")
        st.markdown("#### 1. Shipping Address")
        st.radio(
            "Select destination:",
            [
                "🏠 123 Innovation Drive, San Francisco, CA",
                "🏢 456 Cloud Blvd, Seattle, WA",
            ],
        )

        st.markdown("#### 2. Payment Method")
        st.radio(
            "Select payment:",
            ["💳 Visa ending in 4242", "🍏 Apple Pay", "🅿️ PayPal"],
        )

        st.markdown(f"### Total to Pay: ${total_price}")

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Back to Cart", use_container_width=True):
                st.session_state.view = "cart"
                st.rerun()
        with c2:
            if st.button(
                "Confirm Order & Pay", type="primary", use_container_width=True
            ):
                st.session_state.view = "success"
                st.rerun()

    elif st.session_state.view == "success":
        st.success("🎉 Order Placed Successfully!")
        st.balloons()
        st.markdown(
            "Your items are being prepared for shipment. A receipt has been sent to your email."
        )
        if st.button("Continue Shopping", type="primary"):
            st.session_state.cart = []
            st.session_state.view = "grid"
            st.rerun()

    with st.sidebar:
        st.markdown("### ✨ AI Stylist")

        if st.session_state.audio_to_play:
            st.audio(
                st.session_state.audio_to_play, format="audio/mp3", autoplay=True
            )
            st.session_state.audio_to_play = None

        chat_container = st.container(height=450, border=False)
        with chat_container:
            for msg in st.session_state.messages:
                if (
                    msg.get("role") in ["user", "assistant"]
                    and msg.get("content")
                    and not str(msg.get("content")).startswith("[")
                ):
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])

        with st.container(border=True):
            st.markdown(
                "<p style='font-size: 0.8rem; color: #64748B; margin-bottom: 0px;'>Media Uploads</p>",
                unsafe_allow_html=True,
            )
            col_img, col_mic = st.columns(2)
            with col_img:
                uploaded_file = st.file_uploader(
                    "Image",
                    type=["jpg", "jpeg", "png"],
                    label_visibility="collapsed",
                )
            with col_mic:
                audio_value = st.audio_input(
                    "Voice",
                    label_visibility="collapsed",
                    key=f"mic_{st.session_state.audio_key}",
                )

        user_prompt = st.chat_input("Message your stylist...")

        input_triggered = False
        new_user_message = ""

        if (
            uploaded_file is not None
            and uploaded_file.name not in st.session_state.processed_media
        ):
            with st.spinner("Analyzing image..."):
                st.session_state.processed_media.add(uploaded_file.name)
                base64_image = base64.b64encode(uploaded_file.read()).decode("utf-8")
                vision_res = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Describe this clothing item briefly.",
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}"
                                    },
                                },
                            ],
                        }
                    ],
                )
                image_desc = vision_res.choices[0].message.content or ""

                st.session_state.grid_view = perform_rag_search(
                    client, catalog_embeddings, image_desc, top_k=16
                )
                st.session_state.view = "grid"

                st.session_state.messages.append(
                    {
                        "role": "user",
                        "content": f"*(Uploaded Image: {uploaded_file.name})*",
                    }
                )
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": (
                            f"📸 I analyzed your image! It looks like: **{image_desc}**. "
                            f"I've updated your grid to match!"
                        ),
                    }
                )
                st.rerun()

        elif audio_value and audio_value not in st.session_state.processed_media:
            with st.spinner("Listening..."):
                st.session_state.processed_media.add(audio_value)
                transcription = client.audio.transcriptions.create(
                    model="whisper-1", file=("audio.wav", audio_value)
                )
                new_user_message = f"🎤 {transcription.text}"
                input_triggered = True
                st.session_state.audio_key += 1

        elif user_prompt:
            new_user_message = user_prompt
            input_triggered = True

        if input_triggered:
            st.session_state.messages.append(
                {"role": "user", "content": new_user_message}
            )
            with chat_container.chat_message("user"):
                st.markdown(new_user_message)

            with chat_container.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    event_detected = detect_event_intent(new_user_message)
                    if event_detected:
                        st.session_state.messages.append(
                            {
                                "role": "system",
                                "content": (
                                    f"[SYSTEM ALERT: Deterministic Python Router detected "
                                    f"the event '{event_detected}'. You MUST invoke the "
                                    f"'generate_outfit' tool immediately.]"
                                ),
                            }
                        )

                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=st.session_state.messages,
                        tools=TOOLS_SCHEMA,
                        tool_choice="auto",
                    )
                    msg = response.choices[0].message
                    st.session_state.messages.append(
                        msg.model_dump(exclude_none=True)
                    )

                    if msg.tool_calls:
                        for tool_call in msg.tool_calls:
                            tool_result = execute_tool(tool_call)
                            st.session_state.messages.append(
                                {
                                    "tool_call_id": tool_call.id,
                                    "role": "tool",
                                    "name": tool_call.function.name,
                                    "content": tool_result,
                                }
                            )

                        final_res = client.chat.completions.create(
                            model="gpt-4o", messages=st.session_state.messages
                        )
                        final_reply = final_res.choices[0].message.content or ""
                        st.markdown(final_reply)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": final_reply}
                        )
                    else:
                        final_reply = msg.content or ""
                        st.markdown(final_reply)

                    if "🎤" in new_user_message and final_reply:
                        tts = client.audio.speech.create(
                            model="tts-1", voice="alloy", input=final_reply
                        )
                        tts.stream_to_file("response.mp3")
                        st.session_state.audio_to_play = "response.mp3"

                    st.rerun()

