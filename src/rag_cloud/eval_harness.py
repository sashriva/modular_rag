from __future__ import annotations

"""Small evaluation harness for the cloud RAG stack.

This module runs the existing query pipeline against a golden dataset and scores:
- faithfulness: whether the answer is grounded in retrieved contexts
- answer_relevancy: whether the answer addresses the question
- context_recall: whether expected source documents were retrieved
"""

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from rag_cloud.clients import Clients
from rag_cloud.config import Settings
from rag_cloud.pipeline import RAGPipeline


@dataclass
class GoldenSample:
    id: str
    query: str
    ground_truth: str
    relevant_sources: list[str]
    expected_answer_contains: list[str]


@dataclass
class EvalSampleResult:
    sample_id: str
    query: str
    answer: str
    cache: str
    source_names: list[str]
    faithfulness: float
    answer_relevancy: float
    context_recall: float
    notes: dict[str, str]


class EvalHarness:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.clients = Clients(settings)
        self.pipeline = RAGPipeline(self.clients, settings)
        self.judge_model = settings.llm_model

    @staticmethod
    def load_dataset(dataset_path: str | Path) -> list[GoldenSample]:
        raw = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
        return [GoldenSample(**row) for row in raw]

    @staticmethod
    def _extract_json_object(text: str) -> dict[str, Any]:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError(f"Judge response did not contain JSON: {text}")
        return json.loads(match.group(0))

    def _judge(self, *, prompt: str) -> dict[str, Any]:
        response = self.clients.groq.chat.completions.create(
            model=self.judge_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=250,
        )
        content = response.choices[0].message.content or "{}"
        return self._extract_json_object(content)

    def score_faithfulness(self, query: str, answer: str, contexts: list[str]) -> tuple[float, str]:
        prompt = (
            "You are evaluating whether an answer is fully grounded in retrieved context.\n"
            "Return ONLY valid JSON in the shape {\"score\": number, \"reason\": string}.\n"
            "Score from 0.0 to 1.0 where 1.0 means every material claim in the answer is supported by the context.\n\n"
            f"Question:\n{query}\n\n"
            f"Answer:\n{answer}\n\n"
            f"Contexts:\n{chr(10).join(contexts)}\n"
        )
        judged = self._judge(prompt=prompt)
        return float(judged["score"]), str(judged.get("reason", ""))

    def score_answer_relevancy(self, query: str, answer: str, ground_truth: str) -> tuple[float, str]:
        prompt = (
            "You are evaluating whether an answer addresses the user question.\n"
            "Return ONLY valid JSON in the shape {\"score\": number, \"reason\": string}.\n"
            "Score from 0.0 to 1.0 where 1.0 means the answer is direct, relevant, and aligned with the expected answer.\n\n"
            f"Question:\n{query}\n\n"
            f"Answer:\n{answer}\n\n"
            f"Expected answer:\n{ground_truth}\n"
        )
        judged = self._judge(prompt=prompt)
        return float(judged["score"]), str(judged.get("reason", ""))

    @staticmethod
    def score_context_recall(expected_sources: list[str], retrieved_sources: list[str]) -> float:
        if not expected_sources:
            return 1.0
        expected = {s.lower() for s in expected_sources}
        retrieved = {s.lower() for s in retrieved_sources}
        return len(expected & retrieved) / len(expected)

    def evaluate_sample(self, sample: GoldenSample) -> EvalSampleResult:
        result = self.pipeline.query_dict(sample.query, tenant_id="eval")
        answer = str(result["answer"])
        source_names = [str(s.get("source", "")) for s in result["sources"]]
        contexts = [str(s.get("text", "")) for s in result["sources"]]

        faithfulness, faithfulness_reason = self.score_faithfulness(sample.query, answer, contexts)
        answer_relevancy, relevancy_reason = self.score_answer_relevancy(
            sample.query,
            answer,
            sample.ground_truth,
        )
        context_recall = self.score_context_recall(sample.relevant_sources, source_names)

        return EvalSampleResult(
            sample_id=sample.id,
            query=sample.query,
            answer=answer,
            cache=str(result["cache"]),
            source_names=source_names,
            faithfulness=faithfulness,
            answer_relevancy=answer_relevancy,
            context_recall=context_recall,
            notes={
                "faithfulness_reason": faithfulness_reason,
                "relevancy_reason": relevancy_reason,
            },
        )

    def run(self, dataset_path: str | Path) -> dict[str, Any]:
        dataset = self.load_dataset(dataset_path)
        sample_results = [self.evaluate_sample(sample) for sample in dataset]

        summary = {
            "faithfulness": round(mean([r.faithfulness for r in sample_results]), 3),
            "answer_relevancy": round(mean([r.answer_relevancy for r in sample_results]), 3),
            "context_recall": round(mean([r.context_recall for r in sample_results]), 3),
        }

        return {
            "run_at": datetime.now(timezone.utc).isoformat(),
            "dataset_path": str(dataset_path),
            "summary": summary,
            "samples": [asdict(r) for r in sample_results],
        }

    @staticmethod
    def save(report: dict[str, Any], output_dir: str | Path) -> Path:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        file_path = output_path / f"eval_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
        file_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return file_path
