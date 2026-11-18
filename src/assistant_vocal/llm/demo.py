"""Démo : discussion avec le LLM en streaming, avec mesure de la latence."""
from assistant_vocal.config import load_config
from assistant_vocal.llm import LLMClient, LLMError, make_message

SYSTEM = (
    "Tu es un assistant vocal pour un entretien professionnel. "
    "Réponds en français, en une ou deux phrases courtes, sur un ton naturel et chaleureux, "
    "sans liste, sans markdown, comme à l'oral."
    "Tu parles à l'oral : réponds en français, en deux ou trois phrases courtes maximum, "
    "sans liste, sans numérotation, sans markdown. "
    "Si le sujet est vaste, donne l'idée principale et propose de continuer "
)


def main() -> None:
    llm = LLMClient.from_config(load_config())
    try:
        print("Chargement du modèle en mémoire...")
        llm.warmup()
    except LLMError as exc:
        print(f"Erreur : {exc}")
        return

    history = [make_message("system", SYSTEM)]
    print("Écris un message (ligne vide ou Ctrl+C pour quitter).")
    while True:
        try:
            text = input("\nToi > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            break
        history.append(make_message("user", text))
        print("IA  > ", end="", flush=True)
        reply = []
        try:
            for piece in llm.stream(history):
                print(piece, end="", flush=True)
                reply.append(piece)
        except LLMError as exc:
            print(f"\nErreur : {exc}")
            break
        history.append(make_message("assistant", "".join(reply)))
        s = llm.last_stats
        if s:
            first = f"{s.first_token_s:.2f} s" if s.first_token_s is not None else "n/a"
            print(f"\n   1er mot : {first} | {s.tokens_per_s:.1f} tokens/s | total {s.total_s:.2f} s")
    print("\nAu revoir.")


if __name__ == "__main__":
    main()