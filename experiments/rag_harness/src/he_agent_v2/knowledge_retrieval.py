"""本地、可审计的形态卡检索；不进行联网搜索，不读取任何样本标签。"""

import hashlib
import json


class KnowledgeLibrary:
    def __init__(self, folder):
        """核对知识库文件清单与哈希，再载入卡片；不接受未登记或被改写的知识。"""
        self.folder = folder
        manifest = json.loads((folder / "manifest.json").read_text())
        for name, digest in manifest["files"].items():
            if name not in {"cards.jsonl", "sources.json"}:
                raise ValueError("Unexpected knowledge file")
            if hashlib.sha256((folder / name).read_bytes()).hexdigest() != digest:
                raise ValueError("Knowledge checksum mismatch")
        if set(manifest["files"]) != {"cards.jsonl", "sources.json"}:
            raise ValueError("Incomplete knowledge manifest")
        self.manifest = manifest
        sources = json.loads((folder / "sources.json").read_text())
        self.cards = [json.loads(x) for x in (folder / "cards.jsonl").read_text().splitlines() if x]
        seen = set()
        for card in self.cards:
            if card["id"] in seen or card["modality"] != "ultrasound":
                raise ValueError("Invalid knowledge identity/modality")
            seen.add(card["id"])
            if not card["source_ids"] or any(x not in sources for x in card["source_ids"]):
                raise ValueError("Unknown medical source")
            if not card["locator"] or not card["limitations"] or len(self.render(card)) > 600:
                raise ValueError("Incomplete or oversized medical card")

    @staticmethod
    def render(card):
        """含适用限制的完整卡片；绝不截断句子凑字数。"""
        return f"[{card['id']}] {card['text']} 限制：{card['limitations']}"

    def retrieve(self, branches, limit=2, budget=600):
        """按分支轮转选卡，优先每分支一张；同分按审核优先级及 ID 固定排序。"""
        if not 0 <= limit <= 2 or not 0 <= budget <= 600:
            raise ValueError("Knowledge budget exceeds frozen protocol")
        queues = [[c for c in self.cards if b in c["branches"]] for b in branches]
        for queue in queues:
            queue.sort(key=lambda c: (c["priority"], c["id"]))
        selected, used = [], set()
        while any(queues) and len(selected) < limit:
            for queue in queues:
                if not queue or len(selected) >= limit:
                    continue
                card = queue.pop(0)
                if card["id"] in used:
                    continue
                proposed = selected + [card]
                text = "\n".join(self.render(c) for c in proposed)
                if len(text) <= budget:
                    selected.append(card)
                    used.add(card["id"])
        return selected
