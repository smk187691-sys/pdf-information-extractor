"""End-to-end test suite for the multi-document RAG app."""
import requests
import io
import os
import sys

BASE = "http://127.0.0.1:5000"
ROOT = os.path.dirname(os.path.abspath(__file__))
PASS, FAIL = 0, 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name} {detail}")


def upload(path):
    with open(path, "rb") as f:
        files = {"file": (os.path.basename(path), f,
                 "application/pdf" if path.endswith(".pdf") else "application/vnd.openxmlformats-officedocument.wordprocessingml.document")}
        return requests.post(f"{BASE}/upload", files=files, headers={"X-Requested-With": "XMLHttpRequest"}, timeout=300)


def part1():
    agr_path = os.path.join(ROOT, "test_samples", "agriculture_report.pdf")
    ml_path = os.path.join(ROOT, "test_samples", "machine_learning_notes.docx")

    print("\n== SETUP: reset docs + clear chats ==")
    requests.post(f"{BASE}/reset-documents", timeout=60)
    requests.delete(f"{BASE}/chats", timeout=60)
    r = requests.get(f"{BASE}/documents", timeout=30).json()
    check("start clean", r["success"] and len(r["documents"]) == 0, str(r))

    print("\n== TEST 1: upload agriculture_report.pdf ==")
    r = upload(agr_path)
    d = r.json()
    check("upload success", r.status_code == 200 and d.get("success"), str(d))
    check("has document_id doc_*", str(d.get("document_id", "")).startswith("doc_"), str(d))
    agr_id = d.get("document_id")
    check("chunks created", d.get("chunks", 0) >= 1, str(d))
    check("pages extracted", d.get("pages") == 1, str(d))
    phys = [f for f in os.listdir(os.path.join(ROOT, "uploads")) if f.startswith(str(agr_id))]
    check("physical file stored with doc id prefix", len(phys) == 1, str(phys))

    docs = requests.get(f"{BASE}/documents", timeout=30).json()["documents"]
    check("listed in /documents", len(docs) == 1 and docs[0]["document_id"] == agr_id, str(docs))
    check("registry fields present",
          all(k in docs[0] for k in ("filename", "file_type", "page_count", "chunk_count", "upload_time", "status")),
          str(docs[0]))

    print("\n== TEST 2: upload machine_learning_notes.docx ==")
    r = upload(ml_path)
    d2 = r.json()
    check("upload success", r.status_code == 200 and d2.get("success"), str(d2))
    ml_id = d2.get("document_id")
    check("second doc has own id", ml_id and ml_id != agr_id, f"{agr_id} vs {ml_id}")
    docs = requests.get(f"{BASE}/documents", timeout=30).json()["documents"]
    check("both documents exist", len(docs) == 2, str(docs))

    print("\n== TEST 20: duplicate upload detected ==")
    r = upload(agr_path)
    d = r.json()
    check("duplicate rejected", not d.get("success") and d.get("duplicate"), str(d))
    docs = requests.get(f"{BASE}/documents", timeout=30).json()["documents"]
    check("still only 2 docs", len(docs) == 2, str(len(docs)))

    print("\n== TEST 21: invalid file type rejected ==")
    files = {"file": ("evil.exe", io.BytesIO(b"MZ fake"), "application/octet-stream")}
    r = requests.post(f"{BASE}/upload", files=files, headers={"X-Requested-With": "XMLHttpRequest"}, timeout=60)
    d = r.json()
    check("exe rejected", not d.get("success"), str(d))

    return agr_id, ml_id


def part2(agr_id, ml_id):
    print("\n== TEST 3: ask agriculture question ==")
    r = requests.post(f"{BASE}/chat", json={"question": "By how much does drip irrigation reduce evaporation?"}, timeout=180)
    d = r.json()
    check("chat API success", r.status_code == 200 and d.get("success"), str(d)[:300])
    chat1 = d.get("chat_id", "")
    check("chat_id returned", chat1.startswith("chat_"), str(chat1))
    check("answer returned", bool(d.get("answer")), str(d.get("answer", ""))[:120])
    sources = d.get("sources", [])
    check("sources include agriculture", any("agriculture" in (s.get("document") or "") for s in sources), str(sources))
    agr_src = next((s for s in sources if "agriculture" in (s.get("document") or "")), {})
    check("page is real number or N/A (not fabricated)", agr_src.get("page") in (1, "1", "N/A", None), str(agr_src))

    print("\n== TEST 9/11: chat sessions & titles ==")
    chats = requests.get(f"{BASE}/chats", timeout=30).json()["chats"]
    check("chat listed", any(c["chat_id"] == chat1 for c in chats), str(chats))
    t = next((c["title"] for c in chats if c["chat_id"] == chat1), "")
    check("title generated from question", bool(t) and t != "New Chat", t)

    r = requests.post(f"{BASE}/chat", json={"question": "What is overfitting?", "chat_id": chat1}, timeout=180)
    d = r.json()
    check("continue same chat", r.status_code == 200 and d.get("chat_id") == chat1, str(d)[:200])

    r = requests.post(f"{BASE}/chat", json={"question": "Explain bias variance tradeoff briefly."}, timeout=180)
    d = r.json()
    chat2 = d.get("chat_id", "")
    check("second chat created", d.get("success") and bool(chat2) and chat2 != chat1, str(d)[:200])

    m1 = requests.get(f"{BASE}/chats/{chat1}", timeout=30).json()
    check("chat1 has 4 messages", len(m1.get("messages", [])) == 4, str(len(m1.get("messages", []))))
    m2 = requests.get(f"{BASE}/chats/{chat2}", timeout=30).json()
    check("chat2 has 2 messages", len(m2.get("messages", [])) == 2, str(len(m2.get("messages", []))))
    check("messages store role/content/timestamp",
          all(k in m2["messages"][0] for k in ("role", "content", "timestamp", "sources")), str(m2["messages"][0]))

    print("\n== TEST 13: document filter (search only ML doc) ==")
    r = requests.post(f"{BASE}/chat",
                      json={"question": "What is overfitting and how can it be prevented?",
                            "document_ids": [ml_id]}, timeout=180)
    d = r.json()
    if d.get("success"):
        srcs = d.get("sources", [])
        only_ml = all((s.get("document_id") or ml_id) == ml_id for s in srcs) if srcs else True
        check("retrieval restricted to selected doc", only_ml, str(srcs))
    else:
        check("retrieval restricted to selected doc", False, str(d)[:200])
    return chat1


def part3(agr_id):
    print("\n== TEST 4: delete agriculture document ==")
    r = requests.delete(f"{BASE}/documents/{agr_id}", timeout=300)
    d = r.json()
    check("delete success", r.status_code == 200 and d.get("success"), str(d))
    docs = requests.get(f"{BASE}/documents", timeout=30).json()["documents"]
    check("doc gone from list", all(x["document_id"] != agr_id for x in docs), str(docs))
    phys = [f for f in os.listdir(os.path.join(ROOT, "uploads")) if f.startswith(str(agr_id))]
    check("physical file deleted", len(phys) == 0, str(phys))

    # verify FAISS docstore directly (no agriculture vectors left)
    sys.path.insert(0, ROOT)
    from app.componenets.vector_store import load_vector_store, _load_all_indexed_documents
    db = load_vector_store()
    if db is None:
        check("FAISS after delete", True, "index removed entirely")
    else:
        left = [x for x in _load_all_indexed_documents(db)
                if (x.metadata or {}).get("document_id") == agr_id
                or "agriculture" in ((x.metadata or {}).get("source") or "")]
        check("FAISS has no agriculture vectors", len(left) == 0, f"{len(left)} left")
        check("ML vectors survived", len(_load_all_indexed_documents(db)) >= 1)

    r = requests.post(f"{BASE}/chat", json={"question": "By how much does drip irrigation reduce evaporation?"}, timeout=180)
    d = r.json()
    if d.get("success"):
        srcs = d.get("sources", [])
        check("no agriculture in sources after delete",
              not any("agriculture" in (s.get("document") or "") for s in srcs), str(srcs))
        check("answer says info not found (grounded)",
              any(p in d.get("answer", "").lower() for p in
                  ("not found", "do not contain", "does not contain", "not available",
                   "no information", "unable", "cannot find")),
              str(d.get("answer", ""))[:200])
    else:
        check("chat works after delete", False, str(d)[:300])

    print("\n== delete missing document ==")
    r = requests.delete(f"{BASE}/documents/doc_doesnotexist", timeout=60)
    check("404 for missing doc", r.status_code == 404 and not r.json().get("success"), r.text[:200])


def part4():
    print("\n== TEST 7: clear all chats ==")
    r = requests.delete(f"{BASE}/chats", timeout=60)
    check("clear ok", r.json().get("success"), r.text)
    chats = requests.get(f"{BASE}/chats", timeout=30).json()["chats"]
    check("chats empty", len(chats) == 0, str(chats))
    docs = requests.get(f"{BASE}/documents", timeout=30).json()["documents"]
    check("documents remain", len(docs) >= 1, str(docs))

    print("\n== TEST 6: delete one chat keeps others ==")
    r = requests.post(f"{BASE}/chat", json={"question": "What is bias?"}, timeout=180)
    c1 = r.json().get("chat_id")
    r = requests.post(f"{BASE}/chat", json={"question": "What is variance?"}, timeout=180)
    c2 = r.json().get("chat_id")
    r = requests.delete(f"{BASE}/chats/{c1}", timeout=60)
    check("delete one ok", r.json().get("success"), r.text)
    remaining = [c["chat_id"] for c in requests.get(f"{BASE}/chats", timeout=30).json()["chats"]]
    check("other chat remains", c2 in remaining and c1 not in remaining, str(remaining))
    r = requests.delete(f"{BASE}/chats/{c1}", timeout=60)
    check("second delete -> 404", r.status_code == 404, r.text)
    r = requests.get(f"{BASE}/chats/nope123", timeout=60)
    check("GET missing chat -> 404", r.status_code == 404, r.text)

    print("\n== TEST 21: empty question ==")
    r = requests.post(f"{BASE}/chat", json={"question": "   "}, timeout=60)
    check("400 on empty question", r.status_code == 400, r.text[:200])

    print("\n== TEST 19: index page renders ==")
    r = requests.get(BASE, timeout=60)
    check("GET / 200", r.status_code == 200, str(r.status_code))
    check("has Documents section", "sidebarDocList" in r.text)
    check("has Chat History section", "chatHistoryList" in r.text)


def restart_check():
    """Test 5: verify persistence across a Flask restart (run after restart)."""
    print("\n== TEST 5: restart persistence ==")
    docs = requests.get(f"{BASE}/documents", timeout=30).json()["documents"]
    check("remaining docs survive restart", len(docs) == 1 and "machine_learning" in docs[0]["filename"], str(docs))
    check("deleted doc stays deleted", all("agriculture" not in d["filename"] for d in docs), str(docs))

    sys.path.insert(0, ROOT)
    from app.componenets.vector_store import load_vector_store, _load_all_indexed_documents
    db = load_vector_store()
    check("FAISS index loads after restart", db is not None)
    if db is not None:
        check("ML vectors still present",
              any("machine_learning" in ((x.metadata or {}).get("source") or "")
                  for x in _load_all_indexed_documents(db)))

    chats = requests.get(f"{BASE}/chats", timeout=30).json()["chats"]
    check("chat history survives restart", len(chats) >= 1, str(chats))

    print(f"\n===== RESTART RESULTS: {PASS} passed, {FAIL} failed =====")
    return FAIL


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "full"
    if mode == "restart":
        sys.exit(1 if restart_check() else 0)
    else:
        agr_id, ml_id = part1()
        part2(agr_id, ml_id)
        part3(agr_id)
        part4()
        print(f"\n===== RESULTS: {PASS} passed, {FAIL} failed =====")
        sys.exit(1 if FAIL else 0)
