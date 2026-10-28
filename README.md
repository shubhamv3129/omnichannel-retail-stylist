# 🛍️ RetailNext Smart Store — Agentic Omnichannel E-Commerce Prototype

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.38+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o_%7C_Whisper_%7C_TTS-412991?style=for-the-badge&logo=openai&logoColor=white)
![RAG](https://img.shields.io/badge/Vector_Search-text--embedding--3--small-10B981?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)

**RetailNext Smart Store** is an omnichannel AI fashion storefront and personal stylist built with **Streamlit** and **OpenAI (`gpt-4o`, `text-embedding-3-small`, `whisper-1`, `tts-1`)**. It combines CRM-personalized semantic product search, **"Snap & Shop"** visual similarity discovery, bidirectional voice commerce, deterministic event routing, and automated out-of-stock fallback recommendations.

---

## 📐 System Architecture

```mermaid
flowchart TD
    User["VIP Customer (Sarah)"] -->|Text / Voice / Photo| UI["Streamlit Storefront & AI Stylist Sidebar"]
    UI -->|Voice Input| Whisper["OpenAI Whisper-1\nSpeech-to-Text"]
    UI -->|Photo Upload| Vision["GPT-4o Vision\nGarment Description"]
    Vision --> RAG["Semantic Catalog Search\n(text-embedding-3-small + Cosine Similarity)"]

    Whisper --> Router["Deterministic Event Router\n(detect_event_intent)"]
    UI -->|Chat Prompt| Router
    Router --> Agent["GPT-4o Stylist Agent\n+ CRM Personalization Profile"]

    Agent -->|Tool: search_catalog| RAG
    Agent -->|Tool: generate_outfit| RAG
    Agent -->|Tool: check_local_inventory / add_to_cart| ERP["Omnichannel Inventory Engine\n(Downtown / Uptown / Suburbs)"]
    ERP -->|If Stock == 0| Fallback["Automatic Vector Fallback\nRecommend Top-3 Similar In-Stock Items"]
    Agent -->|If Voice Query| TTS["OpenAI TTS-1 (alloy)\nSpoken Stylist Reply"]
```

---

## ✨ Enterprise Capabilities

- **CRM-Aware Personalization**: Injects customer tier (`VIP Platinum`), preferred style (`Casual Chic & Elegant`), favorite colors, and fabric exclusions directly into the agent's system context.
- **Deterministic Event Intent Router**: Detects high-value occasions (`wedding`, `office`, `party`, `vacation`, `workout`, `date`, `interview`) in Python and forces the agent to assemble a multi-item outfit with explicit styling rationale.
- **Zero-Stock Save-the-Sale Guardrail**: Hard-checks omnichannel store inventory (`inventory.py`) before adding to cart; if an item is out of stock across all locations (e.g., demo SKU `14325`), blocks the cart addition and uses vector similarity to pitch 3 in-stock alternatives.
- **Multimodal Discovery ("Snap & Shop" + Voice)**: Accepts uploaded clothing photos (via `gpt-4o` Vision) and spoken queries (via `whisper-1` and `tts-1`).

---

## 📁 Project Structure

```text
omnichannel-retail-stylist/
├── main.py                      # Primary application entry point
├── app.py                       # Streamlit entry point alias
├── feature_storefront.py        # Storefront UI, vector search, tool execution, and multimodal stylist loop
├── inventory.py                 # Mock multi-location ERP inventory service with stockout simulation
├── sample_styles.csv            # Fashion product catalog metadata
├── sample_images/               # Product catalog imagery
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment variable template
├── .streamlit/
│   ├── config.toml              # Light luxury storefront theme
│   └── secrets.toml.example     # Streamlit secrets template
└── README.md                    # Project documentation
```

---

## ⚙️ Configuration

| Variable | Required | Description |
| :--- | :---: | :--- |
| `OPENAI_API_KEY` | Yes | OpenAI API key (set in `.env` or `.streamlit/secrets.toml`) |

---

## 🚀 Quickstart

```bash
git clone https://github.com/shubhamv3129/omnichannel-retail-stylist.git
   cd omnichannel-retail-stylist
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Or: cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Set your OPENAI_API_KEY

streamlit run main.py
```
