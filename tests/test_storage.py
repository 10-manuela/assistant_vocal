import sqlite3

import pytest

from assistant_vocal.storage import ConversationStore


@pytest.fixture
def store():
    s = ConversationStore(":memory:")
    yield s
    s.close()


def test_create_and_list(store):
    a = store.create_conversation("A")
    b = store.create_conversation("B")
    ids = [c.id for c in store.list_conversations()]
    assert ids == [b.id, a.id]  # la plus récente d'abord


def test_default_title(store):
    assert store.create_conversation().title.startswith("Conversation du")


def test_last_conversation_follows_activity(store):
    a = store.create_conversation("A")
    store.create_conversation("B")
    store.add_message(a.id, "user", "bonjour")  # A redevient la plus récente
    assert store.get_last_conversation().id == a.id


def test_last_conversation_empty(store):
    assert store.get_last_conversation() is None


def test_messages_order_and_emotion(store):
    c = store.create_conversation()
    store.add_message(c.id, "user", "Je suis stressé", emotion="stress", emotion_score=0.82)
    store.add_message(c.id, "assistant", "Respirons un instant.")
    msgs = store.get_messages(c.id)
    assert [m.role for m in msgs] == ["user", "assistant"]
    assert msgs[0].emotion == "stress"
    assert msgs[0].emotion_score == pytest.approx(0.82)
    assert msgs[1].emotion is None


def test_messages_limit_returns_last_ones(store):
    c = store.create_conversation()
    for i in range(5):
        store.add_message(c.id, "user", f"m{i}")
    assert [m.content for m in store.get_messages(c.id, limit=2)] == ["m3", "m4"]


def test_delete_cascades_to_messages(store):
    c = store.create_conversation()
    store.add_message(c.id, "user", "salut")
    assert store.delete_conversation(c.id) is True
    assert store.get_conversation(c.id) is None
    count = store.conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    assert count == 0
    assert store.delete_conversation(c.id) is False


def test_archive_hides_conversation(store):
    c = store.create_conversation()
    store.archive_conversation(c.id)
    assert store.list_conversations() == []
    assert len(store.list_conversations(include_archived=True)) == 1


def test_invalid_role_rejected(store):
    c = store.create_conversation()
    with pytest.raises(sqlite3.IntegrityError):
        store.add_message(c.id, "robot", "x")


def test_unknown_conversation_rejected(store):
    with pytest.raises(sqlite3.IntegrityError):
        store.add_message(999, "user", "x")


def test_settings(store):
    assert store.get_setting("voix", "défaut") == "défaut"
    store.set_setting("voix", "femme")
    store.set_setting("voix", "homme")
    assert store.get_setting("voix") == "homme"


def test_data_survives_restart(tmp_path):
    db = tmp_path / "test.db"
    s1 = ConversationStore(db)
    c = s1.create_conversation("Persistante")
    s1.add_message(c.id, "user", "je reste")
    s1.close()

    s2 = ConversationStore(db)
    assert s2.get_conversation(c.id).title == "Persistante"
    assert s2.get_messages(c.id)[0].content == "je reste"
    s2.close()