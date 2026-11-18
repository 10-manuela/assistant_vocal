"""Démo : gérer des conversations en ligne de commande (base de test séparée)."""
from assistant_vocal.config import ROOT_DIR
from assistant_vocal.storage import ConversationStore

AIDE = """Commandes :
  new [titre]        créer une conversation
  list               lister les conversations
  add <id> <texte>   ajouter un message utilisateur
  show <id>          afficher les messages
  delete <id>        supprimer (messages inclus)
  quit               quitter"""


def main() -> None:
    store = ConversationStore(ROOT_DIR / "data" / "demo.db")
    print(AIDE)
    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue
        cmd, _, rest = line.partition(" ")
        try:
            if cmd == "new":
                c = store.create_conversation(rest or None)
                print(f"Créée : [{c.id}] {c.title}")
            elif cmd == "list":
                for c in store.list_conversations():
                    print(f"[{c.id}] {c.title}  (activité : {c.updated_at[:16]})")
            elif cmd == "add":
                cid, _, text = rest.partition(" ")
                store.add_message(int(cid), "user", text)
                print("Message ajouté.")
            elif cmd == "show":
                for m in store.get_messages(int(rest)):
                    print(f"  {m.role:<9} {m.content}")
            elif cmd == "delete":
                print("Supprimée." if store.delete_conversation(int(rest)) else "Introuvable.")
            elif cmd == "quit":
                break
            else:
                print(AIDE)
        except Exception as exc:
            print(f"Erreur : {exc}")
    store.close()
    print("Au revoir.")


if __name__ == "__main__":
    main()