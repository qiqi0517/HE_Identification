"""最多两次调用的受控工作流；最终预测只来自独立复核。"""

import time

from he_agent import build_request, parse_prediction
from models import ModelError

from .fixed_references import StaleData, UnsafeReferences, sha
from .knowledge_retrieval import KnowledgeLibrary
from .routing import route
from .schemas import parse_observation
from .transport import MeteredClient

MODES = ("reference", "review", "knowledge_review")


def infer_one(
    query,
    reference_set,
    library,
    client,
    prompts,
    mode="review",
    retries=2,
    dry_run=False,
    emit=None,
):
    """query 不含真值；emit 逐事件落盘，不存原始回复、密钥或隐藏推理。"""
    if mode not in MODES:
        raise ValueError("Unknown Agent 2.0 mode")
    emit = emit or (lambda kind, value: None)
    started = time.perf_counter()
    meter = MeteredClient(client, retries)
    result = {
        "status": "error",
        "p_he": None,
        "evidence": None,
        "fatal": False,
        "stage1": None,
        "stage2": None,
        "references": None,
        "branches": [],
        "card_ids": [],
        "routing_fallback": False,
        "calls": meter.calls,
    }

    def stage(number, request, parser):
        """执行指定阶段，解析观察或复核协议，记录状态、响应元数据与耗时。"""
        state = {"status": "error", "prediction": None}
        begin = time.perf_counter()
        try:
            response = meter.generate(request)
            state["response"] = {
                "id": response.response_id,
                "model": response.model,
                "completed": response.completed,
                "finish_reason": response.finish_reason,
            }
            state["prediction"] = parser(response)
            state["status"] = "abstain" if state["prediction"]["p_he"] is None else "ok"
        except ModelError as exc:
            state.update(error="model_error", fatal=exc.fatal)
        except (ValueError, TypeError, KeyError):
            state.update(error="invalid_response", fatal=False)
        except Exception:
            state.update(error="unexpected_client_error", fatal=False)
        state["latency_seconds"] = time.perf_counter() - begin
        result[f"stage{number}"] = state
        emit(f"stage{number}", state)
        return state

    try:
        references, reference_log = reference_set.prepare(query)
        result["references"] = reference_log
        emit("references", reference_log)
        image = query.path.read_bytes()
        if sha(image) != query.sha256:
            raise StaleData("Query changed before request")
        if dry_run:
            # 验证两种请求能承载原图，但没有模型观察，不能伪造路由/医学卡结果。
            result["dry_request"] = {
                "image_parts": sum(
                    hasattr(p, "data")
                    for p in build_request(
                        image, query.mime, prompts["observe"], references
                    ).content
                ),
                "planned_logical_calls": 1 if mode == "reference" else 2,
                "stage2_routing": "requires_live_observation"
                if mode == "knowledge_review"
                else None,
            }
            result["status"] = "dry_run"
            return result
        if mode == "reference":
            final = stage(
                2, build_request(image, query.mime, prompts["basic"], references), parse_prediction
            )
        else:
            first = stage(
                1,
                build_request(image, query.mime, prompts["observe"], references),
                parse_observation,
            )
            if first.get("fatal"):
                result.update(fatal=True, error="fatal_stage1")
                return result
            observation = first["prediction"]
            branches = route(observation)
            cards = library.retrieve(branches) if mode == "knowledge_review" else []
            result.update(
                branches=branches,
                card_ids=[c["id"] for c in cards],
                routing_fallback=observation is None,
            )
            card_text = "\n".join(KnowledgeLibrary.render(c) for c in cards)
            emit(
                "routing",
                {
                    "branches": branches,
                    "card_ids": result["card_ids"],
                    "knowledge_characters": len(card_text),
                    "routing_fallback": observation is None,
                },
            )
            prompt = prompts["basic"] + "\n" + prompts["refine"]
            if card_text:
                prompt += "\n以下是条件性背景资料，并非当前图片的检查发现：\n" + card_text
            # 不传 observation、初判概率或理由；同一原图和同一组固定参考图重新判断。
            final = stage(2, build_request(image, query.mime, prompt, references), parse_prediction)
        result["status"] = final["status"]
        result["fatal"] = final.get("fatal", False)
        if final["prediction"] is not None:
            result.update(final["prediction"])
        else:
            result["error"] = final.get("error", "invalid_response")
    except Exception as exc:
        # 不保存任意异常正文：第三方库异常可能包含请求或本地敏感路径。
        result.update(
            error=(
                "stale_data"
                if isinstance(exc, StaleData)
                else "unsafe_references"
                if isinstance(exc, UnsafeReferences)
                else "pipeline_error"
            ),
            fatal=isinstance(exc, StaleData),
        )
    finally:
        result["latency_seconds"] = time.perf_counter() - started
        emit("outcome", {k: result[k] for k in ("status", "fatal", "calls", "latency_seconds")})
    return result
