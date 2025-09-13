from app.config import get_settings


def main() -> None:
    s = get_settings()
    print(
        "Settings loaded:",
        {
            "LLM_PROVIDER": s.LLM_PROVIDER,
            "OPENAI_MODEL": s.OPENAI_MODEL,
            "EMBEDDING_MODEL": s.EMBEDDING_MODEL,
        },
    )
    try:
        import chromadb
        import llama_index

        _ = (chromadb, llama_index)
        print("Imports OK: chromadb, llama_index")
    except Exception as e:
        raise SystemExit(f"Import failure: {e}") from None


if __name__ == "__main__":
    main()
