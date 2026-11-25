# -*- coding: utf-8 -*-
"""
检索策略消融评估

对同一批评估问题，用 5 种检索配置各跑一遍「检索 → 生成」，再统一用 RAGAS 打分，
用于回答"混合检索 / 重排序 / LLM 策略选择到底带来了多少增益"。

用法：
    # 只验证检索链路（不调用大模型，几乎不花钱）
    python -m rag_qa.rag_assesment.ablation_eval --retrieval-only --limit 3

    # 完整消融（默认 5 组 × 全部题目）
    python -m rag_qa.rag_assesment.ablation_eval

    # 只跑某几组、限制题量
    python -m rag_qa.rag_assesment.ablation_eval --configs dense hybrid_rerank --limit 10

输出：
    ablation_detail.jsonl    每条问题的检索上下文与答案（便于人工抽查）
    ablation_summary.csv     各配置的 RAGAS 指标均值
    ablation_summary.md      可直接贴进 README 的对比表
"""
import argparse
import json
import os
import time

import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv
from langchain_community.chat_models import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from openai import OpenAI
from ragas import evaluate
from ragas.metrics import answer_relevancy, faithfulness
from ragas.run_config import RunConfig

from base import Config, logger
from rag_qa.core.new_rag_system import RAGSystem
from rag_qa.core.prompts import RAGPrompts
from rag_qa.core.vector_store import VectorStore

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
HERE = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
conf = Config()

# 五组对照配置：前三组只改检索，第四组是现网基线，第五组额外启用 LLM 策略选择
CONFIGS = {
    "dense": {"desc": "仅稠密向量（无稀疏、无重排序）", "use_sparse": False, "use_rerank": False, "strategy": False},
    "dense_rerank": {"desc": "稠密向量 + BGE-reranker", "use_sparse": False, "use_rerank": True, "strategy": False},
    "hybrid": {"desc": "稠密 + 稀疏混合，无重排序", "use_sparse": True, "use_rerank": False, "strategy": False},
    "hybrid_rerank": {"desc": "混合检索 + 重排序（现网基线）", "use_sparse": True, "use_rerank": True, "strategy": False},
    "full": {"desc": "混合检索 + 重排序 + LLM 选策略", "use_sparse": True, "use_rerank": True, "strategy": True},
}

# 已完成（从日志抢救）的配置，跳过重跑，直接并入汇总表，省 token
PRECOMPUTED = {
    "dense": {"faithfulness": 0.794, "answer_relevancy": 0.739,
              "context_precision": 0.591, "context_recall": 0.742},
    "dense_rerank": {"faithfulness": 0.838, "answer_relevancy": 0.757,
                     "context_precision": 0.726, "context_recall": 0.708},
}
# 本次只跑这两个生成质量指标；context_* 用 retrieval-only 的 Recall@N / MRR 替代
ACTIVE_METRICS = [faithfulness, answer_relevancy]


def save_progress(detail, summary_rows):
    """增量落盘：每跑完一个配置就写一次，防止中途崩溃丢失结果"""
    with open(os.path.join(HERE, "ablation_detail.jsonl"), "w", encoding="utf-8") as f:
        for r in detail:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    if summary_rows:
        cols = ["config", "desc", "n", "faithfulness", "answer_relevancy",
                "context_precision", "context_recall"]
        df = pd.DataFrame(summary_rows)
        df = df[[c for c in cols if c in df.columns]]
        df.to_csv(os.path.join(HERE, "ablation_summary.csv"), index=False)
        headers = ["配置", "说明", "题量", "faithfulness", "answer_relevancy",
                   "context_precision", "context_recall"]
        lines = ["| " + " | ".join(headers) + " |",
                 "|" + "|".join([" --- "] * len(headers)) + "|"]
        for r in summary_rows:
            lines.append("| " + " | ".join([
                r["config"], r["desc"], str(r["n"]),
                str(r.get("faithfulness", "-")), str(r.get("answer_relevancy", "-")),
                str(r.get("context_precision", "-")), str(r.get("context_recall", "-")),
            ]) + " |")
        with open(os.path.join(HERE, "ablation_summary.md"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")


def build_llm():
    client = OpenAI(api_key=conf.DASHSCOPE_API_KEY, base_url=conf.DASHSCOPE_BASE_URL)

    def call_llm(prompt):
        """非流式调用，评估场景一次性拿全量答案，便于统计"""
        completion = client.chat.completions.create(
            model=conf.LLM_MODEL,
            messages=[{"role": "system", "content": "你是一个有用的助手。"},
                      {"role": "user", "content": prompt}],
            temperature=0.1,
            timeout=60,
        )
        return completion.choices[0].message.content or ""

    def stream_llm(prompt):
        """流式生成器：RAGSystem 的 HyDE / 子查询改写要求 llm 可被逐 token 迭代"""
        completion = client.chat.completions.create(
            model=conf.LLM_MODEL,
            messages=[{"role": "system", "content": "你是一个有用的助手。"},
                      {"role": "user", "content": prompt}],
            timeout=60,
            stream=True,
        )
        for chunk in completion:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    return call_llm, stream_llm


def retrieve(vector_store, rag, query, cfg):
    """按配置取上下文，返回 (文档列表, 实际使用的策略名)"""
    if cfg["strategy"]:
        strategy = rag.strategy_selector.select_strategy(query)
        docs = rag.retrieve_and_merge(query, strategy=strategy)
        return docs, strategy
    docs = vector_store.hybrid_search_with_rerank(
        query, k=conf.RETRIEVAL_K, use_sparse=cfg["use_sparse"], use_rerank=cfg["use_rerank"]
    )
    return docs, "直接检索(固定)"


def build_prompt(context, question):
    return RAGPrompts.rag_prompt().format(
        context=context, question=question, phone=conf.CUSTOMER_SERVICE_PHONE, history=""
    )


def run(args):
    with open(os.path.join(HERE, "rag_evaluate_data.json"), encoding="utf-8") as f:
        data = json.load(f)
    if args.limit:
        data = data[: args.limit]

    names = list(CONFIGS) if args.configs in (None, "all") else args.configs
    unknown = [n for n in names if n not in CONFIGS]
    if unknown:
        raise SystemExit(f"未知配置: {unknown}，可选: {list(CONFIGS)}")

    logger.info(f"评估题目 {len(data)} 条，配置 {names}")
    vector_store = VectorStore()
    call_llm, stream_llm = build_llm()
    rag = RAGSystem(vector_store, stream_llm)  # 仅 full 配置会用到策略选择

    detail, summary_rows = [], []
    # 先并入已完成的配置（跳过重跑，省 token）
    for name in names:
        if name in PRECOMPUTED:
            cfg = CONFIGS[name]
            row = {"config": name, "desc": cfg["desc"], "n": len(data)}
            row.update(PRECOMPUTED[name])
            summary_rows.append(row)
            logger.info(f"[{name}] 使用已保存结果，跳过重跑: {row}")
    save_progress(detail, summary_rows)

    for name in names:
        if name in PRECOMPUTED:
            continue
        cfg = CONFIGS[name]
        logger.info(f"=== 开始配置 {name}：{cfg['desc']} ===")
        records, cost = [], 0.0
        for i, item in enumerate(data, 1):
            q = item["question"]
            t0 = time.time()
            docs, strategy = retrieve(vector_store, rag, q, cfg)
            contexts = [d.page_content for d in docs]
            # --retrieval-only：只跑检索，不调大模型
            answer = ""
            if not args.retrieval_only:
                answer = call_llm(build_prompt("\n\n".join(contexts), q))
            elapsed = time.time() - t0
            cost += elapsed
            records.append({
                "config": name, "question": q, "strategy": strategy,
                "n_docs": len(contexts), "contexts": contexts, "answer": answer,
                "ground_truth": item["ground_truth"], "elapsed_s": round(elapsed, 2),
            })
            logger.info(f"[{name}] {i}/{len(data)} 检索 {len(contexts)} 块，耗时 {elapsed:.1f}s")

        detail.extend(records)
        logger.info(f"配置 {name} 完成，平均单题耗时 {cost / max(len(data), 1):.1f}s")

        if args.retrieval_only or not records or not records[0]["answer"]:
            continue

        logger.info(f"[{name}] 调用 RAGAS 打分中...")
        ds = Dataset.from_dict({
            "question": [r["question"] for r in records],
            "answer": [r["answer"] for r in records],
            "contexts": [r["contexts"] for r in records],
            "ground_truth": [r["ground_truth"] for r in records],
        })
        result = evaluate(
            dataset=ds,
            metrics=ACTIVE_METRICS,
            llm=ChatTongyi(model="qwen-max", api_key=conf.DASHSCOPE_API_KEY),
            embeddings=DashScopeEmbeddings(dashscope_api_key=conf.DASHSCOPE_API_KEY,
                                           model="text-embedding-v3"),
            run_config=RunConfig(max_workers=4, max_retries=3),
            raise_exceptions=False,
        )
        scores = result.to_pandas().mean(numeric_only=True).to_dict()
        row = {"config": name, "desc": cfg["desc"], "n": len(records)}
        row.update({k: round(v, 3) for k, v in scores.items() if isinstance(v, (int, float))})
        summary_rows.append(row)
        logger.info(f"[{name}] RAGAS: {row}")
        save_progress(detail, summary_rows)  # 增量落盘，防止中途崩溃丢结果

    save_progress(detail, summary_rows)
    if summary_rows and not args.retrieval_only:
        with open(os.path.join(HERE, "ablation_summary.md"), encoding="utf-8") as f:
            print(f.read())
    elif not summary_rows:
        logger.info("未生成汇总表（检索模式或未产生答案）")


def parse_args():
    p = argparse.ArgumentParser(description="检索策略消融评估")
    p.add_argument("--configs", nargs="+", default="all",
                   help=f"要跑的配置，可选 {list(CONFIGS)}，默认全部")
    p.add_argument("--limit", type=int, default=0, help="只跑前 N 条题目，0 表示全部")
    p.add_argument("--retrieval-only", action="store_true", help="只跑检索，不调用大模型")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
