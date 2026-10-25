"""Тесты главы 2 на примере Barista Agent. Запуск: python -m unittest test_ch02.py"""
import tempfile
import threading
import time
import unittest
from pathlib import Path

from memory import (CompactConfig, Episode, EpisodicWriter, MemoryFile, Procedural, Working,
                    build_context, collect_prefetch, compact, start_prefetch)
from retrieval import Index, assemble, chunk_text, compare_costs, edge_order
from router import ModelSpec, Request, ThinkingPolicy, Verdict, route, turn_cost

# Цены условные, соотношения — как у типичного провайдера: выход в 5 раз дороже входа,
# чтение кэша — 10% входа, запись в кэш — 125% входа.
SMALL = ModelSpec("barista-small", "small", 1.0, 5.0, 0.1, 1.25)
BIG = ModelSpec("barista-frontier", "frontier", 5.0, 25.0, 0.5, 6.25)
VISION = ModelSpec("barista-vision", "frontier", 6.0, 30.0, 0.6, 7.5, frozenset({"text", "image"}))
MODELS = [SMALL, BIG, VISION]


def clf(label, conf=0.95):
    return lambda text: Verdict(label, conf)


def broken(text):
    raise TimeoutError("classifier down")


class RouterTests(unittest.TestCase):
    def test_simple_goes_to_small_model(self):
        d = route(Request("Сколько стоит латте?"), MODELS, clf("simple"))
        self.assertEqual((d.model.name, d.reason), ("barista-small", "classifier"))

    def test_complex_goes_to_frontier(self):
        d = route(Request("Спланируй заказ на офис"), MODELS, clf("complex"))
        self.assertEqual(d.model.name, "barista-frontier")

    def test_modality_filter(self):
        d = route(Request("Что на фото?", frozenset({"text", "image"})), MODELS, clf("simple"))
        self.assertEqual(d.model.name, "barista-vision")   # маленькая модель картинку не примет

    def test_classifier_failure_falls_back_to_rules(self):
        d = route(Request("Сколько стоит капучино?"), MODELS, broken)
        self.assertEqual((d.model.name, d.reason), ("barista-small", "rules"))
        self.assertIn("classifier_error:TimeoutError", d.trace)

    def test_low_confidence_unknown_text_goes_to_default(self):
        d = route(Request("Хм, а вот такое"), MODELS, clf("simple", 0.4))
        self.assertEqual((d.model.name, d.reason), ("barista-frontier", "default"))

    def test_cache_aware_stays_on_expensive_model(self):
        # Длинный диалог на frontier-модели: переход на дешёвую потребует перечитать
        # 100k токенов без кэша и записать их в новый кэш — это дороже, чем остаться.
        req = Request("Сколько стоит латте?", current_model="barista-frontier",
                      history_tokens=100_000, new_tokens=50, expected_output=100)
        d = route(req, MODELS, clf("simple"))
        self.assertEqual((d.model.name, d.reason), ("barista-frontier", "cache_stay"))

    def test_short_dialog_switches_to_cheap(self):
        req = Request("Сколько стоит латте?", current_model="barista-frontier",
                      history_tokens=500, new_tokens=50, expected_output=100)
        self.assertEqual(route(req, MODELS, clf("simple")).model.name, "barista-small")

    def test_turn_cost(self):
        # 1M из кэша + 1M без кэша + 1M выхода на BIG = 0.5 + 5 + 25
        self.assertAlmostEqual(turn_cost(BIG, 1_000_000, 1_000_000, 1_000_000), 30.5)

    def test_thinking_policy(self):
        p = ThinkingPolicy(plan_iterations=2, budget_tokens=4000)
        self.assertEqual([p.budget(i) for i in (1, 2, 3, 4)], [4000, 4000, 0, 0])
        self.assertEqual(p.budget(5, last_step_failed=True), 2000)


MENU_DOC = ("Латте — эспрессо и молоко, 250 рублей. Капучино — 230 рублей. "
            "Безлактозные опции: овсяное и миндальное молоко, доплата 40 рублей. "
            "Какао без кофеина, 220 рублей. Травяной чай без кофеина, 180 рублей. ") * 3
POLICY_DOC = "Кофейня работает с 8 до 22. Возврат денег возможен в течение 15 минут после оплаты."


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.chunks = chunk_text("menu", MENU_DOC, size=200, overlap=40) + chunk_text("policy", POLICY_DOC)
        self.index = Index(self.chunks)

    def test_chunks_overlap_and_cover_text(self):
        c = chunk_text("menu", MENU_DOC, size=200, overlap=40)
        self.assertGreater(len(c), 1)
        self.assertTrue(all(len(x.text) <= 200 for x in c))
        self.assertIn(c[0].text[-20:].split()[-1], c[1].text)   # хвост первого есть во втором

    def test_search_finds_relevant_source(self):
        top = self.index.search("возврат денег после оплаты", k=1)
        self.assertEqual(top[0][0].source, "policy")

    def test_assemble_respects_budget(self):
        found = self.index.search("без кофеина молоко", k=10)
        text, ids = assemble("без кофеина", found, budget_tokens=120)
        self.assertLessEqual(len(text) // 4, 120 + 30)
        self.assertLess(len(ids), len(found))

    def test_edge_order_puts_best_at_edges(self):
        self.assertEqual(edge_order([1, 2, 3, 4, 5]), [1, 3, 5, 4, 2])

    def test_long_context_vs_rag(self):
        # Документ на 200k токенов, 20 ходов, RAG приносит 3k токенов на ход
        no_cache = compare_costs(200_000, 3_000, 20, price_in=5.0)
        with_cache = compare_costs(200_000, 3_000, 20, price_in=5.0, price_cache_read=0.5)
        self.assertGreater(no_cache.ratio, 50)
        self.assertGreater(with_cache.ratio, 5)            # кэш сокращает разрыв, но не убирает
        self.assertLess(with_cache.long_context, no_cache.long_context)


FILES = [MemoryFile(f"pref_{i}", f"предпочтение гостя {i}", f"тело {i}") for i in range(10)]


class MemoryTests(unittest.TestCase):
    def test_prefetch_limits_and_filters(self):
        sel = lambda q, manifest: ["pref_1", "unknown", "pref_1"] + [n for n, _ in manifest]
        got = collect_prefetch(start_prefetch("латте", FILES, sel, limit=5), wait_s=1)
        self.assertEqual(len(got), 5)
        self.assertEqual(got[0].name, "pref_1")           # дубликаты и выдуманные имена отброшены

    def test_prefetch_does_not_block(self):
        release = threading.Event()
        def slow(q, m):
            release.wait(2)
            return ["pref_0"]
        t0 = time.monotonic()
        fut = start_prefetch("латте", FILES, slow)
        self.assertLess(time.monotonic() - t0, 0.1)        # запуск мгновенный
        self.assertEqual(collect_prefetch(fut, wait_s=0.05), [])  # не дождались — работаем без памяти
        release.set()

    def test_episodic_writer_at_least_once_is_idempotent(self):
        ep = Episode("гость с аллергией", "предложил латте", "жалоба", "уточнять молоко до заказа", "m1")
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "episodes.jsonl"
            msgs = [{"role": "user", "content": "у меня аллергия на лактозу"}]
            w = EpisodicWriter(log, lambda fresh: [ep])
            self.assertEqual(w.run(msgs), 1)
            # «Перезапуск» после сбоя: курсор потерян, сообщения обрабатываются повторно
            w2 = EpisodicWriter(log, lambda fresh: [ep])
            self.assertEqual(w2.run(msgs), 0)
            self.assertEqual(len(w2.lessons()), 1)

    def test_episodic_cursor_not_advanced_on_failure(self):
        def boom(fresh):
            raise RuntimeError("extractor down")
        with tempfile.TemporaryDirectory() as d:
            w = EpisodicWriter(Path(d) / "e.jsonl", boom)
            with self.assertRaises(RuntimeError):
                w.run([{"role": "user", "content": "x"}])
            self.assertEqual(w.cursor, 0)

    def test_offload_keeps_head_and_path(self):
        big = "X" * 20_000
        with tempfile.TemporaryDirectory() as d:
            out, steps = compact([{"role": "tool", "content": big}], CompactConfig(window=100_000),
                                 Path(d), summarize=lambda m: "")
            self.assertEqual(steps, ["offload:0"])
            self.assertTrue(out[0]["content"].startswith("X" * 2000))
            self.assertIn("tool_0.txt", out[0]["content"])
            self.assertEqual((Path(d) / "tool_0.txt").read_text(), big)

    def test_compaction_ladder_escalates(self):
        msgs = [{"role": "tool", "content": "y" * 1600} for _ in range(10)] + \
               [{"role": "user", "content": "z" * 400} for _ in range(6)]
        cfg = CompactConfig(window=4_000, keep_recent=6)
        with tempfile.TemporaryDirectory() as d:
            out, steps = compact(msgs, cfg, Path(d), summarize=lambda m: f"{len(m)} сообщений")
        self.assertTrue(any(s.startswith("micro") for s in steps))
        self.assertNotIn("auto", steps)                     # после microcompact уже влезло
        self.assertEqual(out[-6:], msgs[-6:])               # последние сообщения не тронуты

    def test_autocompact_when_micro_not_enough(self):
        msgs = [{"role": "user", "content": "q" * 4000} for _ in range(10)]
        cfg = CompactConfig(window=4_000, keep_recent=2)
        with tempfile.TemporaryDirectory() as d:
            out, steps = compact(msgs, cfg, Path(d), summarize=lambda m: f"{len(m)} сообщений")
        self.assertIn("auto", steps)
        self.assertEqual(len(out), 3)
        self.assertIn("8 сообщений", out[0]["content"])

    def test_context_static_first_dynamic_last(self):
        proc = Procedural("Ты — бариста.", ["get_menu: меню и цены"])
        work = Working([{"role": "user", "content": "Хочу латте"}])
        a = build_context(proc, "<knowledge>A</knowledge>", [], [], work)
        b = build_context(proc, "<knowledge>B</knowledge>", FILES[:1], [], work)
        self.assertEqual(a[:2], b[:2])                      # system + история не зависят от хода
        self.assertIn("B", b[-1]["content"])


if __name__ == "__main__":
    unittest.main()
