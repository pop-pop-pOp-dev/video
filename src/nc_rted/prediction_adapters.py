"""Concrete adapters for bound ReactVAU VAD and HIVAU runtime objects.

The construction of these objects belongs to the accepted runtime binding.  At
prediction time this module calls the original public evaluator methods and
copies their complete raw outputs; it never reconstructs prompts, Fast scores,
memory updates, fusion, smoothing, or decoding itself.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping
import torch

from .bridge import EvidenceSlowBridge
from .batches import pack_observation_blocks
from .detection_provider import DetectionMemoryReplay, DetectionProtocol
from .prediction_media import FullBlindDetectionReader
from .prediction_inputs import ModelArtifact, VadRequest, VauRequest
from .prediction_worker import PredictionExecutionError


@dataclass
class BoundReactVAUModel:
    """One independently loaded Slow instance and its two bound evaluator routes."""
    group: str
    seed: int | None
    evidence_enabled: bool
    bridge: EvidenceSlowBridge
    vad_detector: Any
    hivau_inference: Any
    load_report: Mapping[str, Any]
    residency: Any


class ReactVAUVadAdapter:
    """Invoke the inherited evaluator's complete VAD route without output clipping."""
    def predict(self, request: VadRequest, model: BoundReactVAUModel, *, protocol: dict[str, Any]) -> dict[str, Any]:
        vad = protocol.get("vad")
        if not isinstance(vad, dict) or set(vad) != {"target_fps", "query_interval", "batch_size"}:
            raise PredictionExecutionError("bound VAD protocol is incomplete")
        primary_error = None
        try:
            model.residency.activate_language()
            result = model.vad_detector.detect_video(request.media_path, target_fps=vad["target_fps"],
                                                      query_interval=vad["query_interval"], batch_size=vad["batch_size"], verbose=False)
        except BaseException as error:
            primary_error = error
            raise
        finally:
            try: model.residency.stage_language()
            except BaseException:
                if primary_error is None: raise
        if not isinstance(result, dict):
            raise PredictionExecutionError("inherited VAD evaluator returned no record")
        scores, fast, indices = result.get("query_scores"), result.get("paligemma_scores"), result.get("query_frame_indices")
        online_frames = result.get("online_smoothed_frame_scores")
        if not all(isinstance(value, list) for value in (scores, fast, indices, online_frames)) or len(scores) != len(fast) or len(scores) != len(indices):
            raise PredictionExecutionError("inherited VAD evaluator omitted full query/causal smoothing output")
        triggered = {item.get("query_idx"): item for item in result.get("trigger_details", []) if isinstance(item, dict)}
        if any(detail.get("cot_reasoning") for detail in triggered.values()):
            raise PredictionExecutionError("inherited VAD CoT detail is truncated; a lossless bound route is required")
        queries = []
        for index, (final, pg, frames) in enumerate(zip(scores, fast, indices)):
            if not isinstance(frames, list) or not frames:
                raise PredictionExecutionError("inherited VAD evaluator emitted an incomplete query frame group")
            detail = triggered.get(index)
            queries.append({"query_index": index, "frame_indices": frames, "fast_score": pg, "final_score": final,
                            "triggered": detail is not None, "slow_score": None if detail is None else detail.get("streamforest_score"),
                            "fused_score": None if detail is None else detail.get("fused_score")})
        # The evaluator exposes the causal expansion at original-frame granularity.
        # Keep it untouched rather than substituting Gaussian compatibility output.
        total_frames = result.get("total_frames")
        if type(total_frames) is not int or total_frames < 1:
            raise PredictionExecutionError("inherited VAD evaluator omitted original-frame geometry")
        return {"queries": queries, "causal_smoothed_scores": online_frames, "total_frames": total_frames, "raw": result}


class ReactVAUVauAdapter:
    """Invoke `ReactVAUInference.generate` with the official question unchanged."""
    def generate(self, request: VauRequest, model: BoundReactVAUModel, *, protocol: dict[str, Any]) -> dict[str, Any]:
        hivau = protocol["hivau"]
        runtime = model.hivau_inference
        # The official wrapper has these values at construction, so accepting a
        # mismatched object would silently change sampling or Fast prompt context.
        if getattr(runtime, "target_fps", None) != hivau["target_fps"] or getattr(runtime, "query_interval", None) != hivau["query_interval"] or getattr(runtime, "paligemma_batch_size", None) != hivau["paligemma_batch_size"]:
            raise PredictionExecutionError("HIVAU runtime differs from its bound sampling protocol")
        pipeline = getattr(runtime, "pipeline", None)
        if getattr(pipeline, "context_mode", None) != "none":
            raise PredictionExecutionError("HIVAU Fast score prompt context is enabled")
        primary_error = None
        try:
            model.residency.activate_language()
            result = runtime.generate(video_path=request.media_path, question=request.question,
                                      max_new_tokens=hivau["max_new_tokens"], task=hivau["task"])
        except BaseException as error:
            primary_error = error
            raise
        finally:
            try: model.residency.stage_language()
            except BaseException:
                if primary_error is None: raise
        if not isinstance(result, dict) or not isinstance(result.get("response"), str):
            raise PredictionExecutionError("inherited HIVAU inference returned no response")
        # The released wrapper exposes text but not generated IDs. The bound runtime
        # must expose them so predictions remain lossless rather than pretending IDs.
        token_ids = result.get("token_ids")
        if not isinstance(token_ids, list):
            raise PredictionExecutionError("bound HIVAU runtime did not expose untruncated generated token IDs")
        return {"text": result["response"], "token_ids": token_ids, "raw": result}


class BlindPromptTokenizer:
    """Build the evaluator's open-ended Qwen generation prompt exactly."""
    def __init__(self, tokenizer, data_args, conv_templates, tokenizer_image_token, image_token, image_token_index):
        self.tokenizer, self.data_args = tokenizer, data_args
        self.conv_templates, self.tokenizer_image_token = conv_templates, tokenizer_image_token
        self.image_token, self.image_token_index = image_token, image_token_index

    def encode(self, *, question: str, context) -> Any:
        from .bridge import SlowInputs
        if getattr(self.data_args, "is_multimodal", False) is not True:
            raise PredictionExecutionError("inherited blind tokenization is not multimodal")
        conversation = self.conv_templates["qwen_2"].copy()
        conversation.append_message(conversation.roles[0], self.image_token + "\n" + context.time_message + question)
        conversation.append_message(conversation.roles[1], None)
        input_ids = self.tokenizer_image_token(conversation.get_prompt(), self.tokenizer, self.image_token_index,
                                               return_tensors="pt").unsqueeze(0)
        if input_ids.ndim != 2 or input_ids.shape[0] != 1 or int((input_ids == -200).sum()) != 1:
            raise PredictionExecutionError("inherited blind tokenization lost the image sentinel")
        return SlowInputs(input_ids.to(context.visual_embeddings.device), context.visual_embeddings, context.images,
                          context.image_sizes, labels=None,
                          attention_mask=input_ids.ne(getattr(self.tokenizer, "pad_token_id", -1)).to(context.visual_embeddings.device))


class BlindDetectionRunner:
    """Concrete inherited VAD replay plus answer-free Slow yes/no probability."""
    def __init__(self, *, bridge: EvidenceSlowBridge, reader, protocol: DetectionProtocol, observation_reader,
                 prompt_tokenizer: BlindPromptTokenizer, yes_token_ids: tuple[int, ...], no_token_ids: tuple[int, ...],
                 fusion: str, fusion_alpha: float, smoother):
        self.bridge, self.reader, self.protocol, self.observation_reader = bridge, reader, protocol, observation_reader
        self.prompt_tokenizer, self.yes_token_ids, self.no_token_ids = prompt_tokenizer, yes_token_ids, no_token_ids
        self.fusion, self.fusion_alpha, self.smoother = fusion, fusion_alpha, smoother
        self._slow_head_active = False

    def _last_token_head_call(self, prepared):
        arguments = prepared.arguments
        raw = getattr(self.bridge, "raw_slow", None)
        if (not isinstance(arguments, dict) or arguments.get("labels") is not None or raw is None
                or not isinstance(raw, torch.nn.Module) or not isinstance(self.bridge.slow, torch.nn.Module)
                or raw.training or self.bridge.slow.training or torch.is_grad_enabled() or self._slow_head_active):
            raise PredictionExecutionError("bound Slow scoring route is not inference-safe")
        output_embeddings = getattr(raw, "get_output_embeddings", None)
        if not callable(output_embeddings):
            raise PredictionExecutionError("bound Slow model has no output-head accessor")
        head = output_embeddings()
        config = getattr(raw, "config", None)
        if (head is not getattr(raw, "lm_head", None) or not isinstance(head, torch.nn.Linear) or head.bias is not None
                or getattr(config, "hidden_size", None) != head.in_features or getattr(config, "vocab_size", None) != head.out_features
                or head._forward_pre_hooks or head._forward_hooks or torch.nn.utils.parametrize.is_parametrized(head)):
            raise PredictionExecutionError("bound Slow output head differs from the supported inference shape")
        calls = 0

        def last_token_only(module, args):
            nonlocal calls
            calls += 1
            if calls != 1 or len(args) != 1 or not isinstance(args[0], torch.Tensor):
                raise PredictionExecutionError("bound Slow output head was reentered")
            hidden = args[0]
            if (hidden.ndim != 3 or hidden.shape[0] != 1 or hidden.shape[1] < 1 or hidden.shape[2] != module.in_features
                    or hidden.dtype != module.weight.dtype or hidden.device != module.weight.device):
                raise PredictionExecutionError("bound Slow output head input differs from the expected hidden state")
            return (hidden[:, -1:, :],)

        handle = head.register_forward_pre_hook(last_token_only)
        self._slow_head_active = True
        try:
            result = self.bridge.slow(**arguments, use_cache=False, return_dict=True)
            logits = getattr(result, "logits", None)
            if (calls != 1 or not isinstance(logits, torch.Tensor) or logits.ndim != 3
                    or tuple(logits.shape[:2]) != (1, 1) or logits.shape[2] != head.out_features
                    or logits.dtype != head.weight.dtype or logits.device != head.weight.device):
                raise PredictionExecutionError("bound Slow output head did not return one final-token vocabulary row")
            return logits
        finally:
            handle.remove()
            self._slow_head_active = False

    def _slow_probability(self, inputs, observations, *, enabled: bool) -> float:
        with torch.no_grad():
            prepared = self.bridge.prepare(inputs, observations if enabled else None, enabled=enabled)
            logits = self._last_token_head_call(prepared)[0, -1].float()
            yes = logits[list(self.yes_token_ids)].max()
            no = logits[list(self.no_token_ids)].max()
            return float((yes - torch.logaddexp(yes, no)).exp().item())

    def _fuse(self, fast: float, slow: float) -> float:
        if self.fusion == "replace": return slow
        if self.fusion == "weighted": return self.fusion_alpha * fast + (1.0 - self.fusion_alpha) * slow
        if self.fusion == "adaptive": return (1.0 - slow) * fast + slow * slow
        raise PredictionExecutionError("unknown inherited VAD fusion")

    def detect(self, request: VadRequest) -> dict[str, Any]:
        replay = DetectionMemoryReplay(self.bridge.slow, self.protocol)
        prefix = self.reader(request.dataset, request.media_id, expected_media_path=request.media_path,
                             expected_media_sha256=request.media_sha256)
        queries, raw_scores = [], []
        try:
            for query in prefix.queries:
                triggered = query.fast_score >= self.protocol.trigger_threshold
                if triggered and self.protocol.rt_anomaly and query.dense_patches is None:
                    raise PredictionExecutionError("blind VAD reader omitted required four-frame RT anomaly patches for a trigger")
                captured = replay.step(query, capture=triggered, image_height=prefix.image_height, image_width=prefix.image_width)
                slow = None
                if captured is not None:
                    context, question = captured
                    enabled = getattr(self.bridge, "prediction_evidence_enabled", True)
                    observations = None
                    if enabled:
                        if self.observation_reader is None:
                            raise PredictionExecutionError("evidence-enabled VAD route has no observation reader")
                        observed = self.observation_reader(request.dataset, request.media_id, context.observed_seconds)
                        observations = pack_observation_blocks([observed.features], task="detection", dtype=context.visual_embeddings.dtype,
                                                               device=context.visual_embeddings.device)
                    slow = self._slow_probability(self.prompt_tokenizer.encode(question=question, context=context), observations,
                                                  enabled=enabled)
                final = query.fast_score if slow is None else self._fuse(query.fast_score, slow)
                raw_scores.append(final)
                queries.append({"query_index": query.index, "frame_indices": list(query.frame_indices), "fast_score": query.fast_score,
                                "triggered": triggered, "slow_score": slow, "final_score": final})
        finally:
            close = getattr(prefix.queries, "close", None)
            if close is not None: close()
        self.smoother.reset()
        if not raw_scores or prefix.frame_count is None or prefix.sample_interval is None:
            raise PredictionExecutionError("blind VAD replay omitted complete original-frame geometry")
        causal_queries = [self.smoother.step(score) for score in raw_scores]
        # This is the released order: causal smoothing at query resolution,
        # four sampled frames per query, then each sampled score to originals.
        sampled = [score for score in causal_queries for _ in range(4)]
        causal = [score for score in sampled for _ in range(prefix.sample_interval)][:prefix.frame_count]
        if len(causal) != prefix.frame_count:
            raise PredictionExecutionError("blind VAD causal expansion omitted original frames")
        return {"queries": queries, "causal_smoothed_scores": causal, "total_frames": prefix.frame_count,
                "sample_interval": prefix.sample_interval}


class BlindVauRunner:
    """Answer-free HIVAU generation over the complete original Stage2 media route.

    ``media_reader`` is the accepted identity-only Stage2 decoder. It must return
    the original visual memory/context and an observation batch covering every
    eight-second block, including the tail; it has no conversation or answer
    input surface.
    """
    def __init__(self, *, bridge: EvidenceSlowBridge, media_reader, prompt_tokenizer: BlindPromptTokenizer, tokenizer,
                 generation_config: Mapping[str, Any]):
        self.bridge, self.media_reader = bridge, media_reader
        self.prompt_tokenizer, self.tokenizer = prompt_tokenizer, tokenizer
        self.generation_config = dict(generation_config)
        forbidden = {"penalty_alpha", "top_k", "top_p", "typical_p", "constraints", "force_words_ids", "prefix_allowed_tokens_fn", "assistant_model"}
        if (self.generation_config.get("do_sample") is not False or self.generation_config.get("num_beams", 1) != 1 or
                self.generation_config.get("num_return_sequences", 1) != 1 or forbidden.intersection(self.generation_config)):
            raise PredictionExecutionError("official HIVAU blind generation must be greedy")
        self.generation_config["do_sample"] = False
        self.generation_config["num_beams"] = 1
        self.generation_config["num_return_sequences"] = 1

    def generate(self, request: VauRequest) -> dict[str, Any]:
        material = self.media_reader.read(media_path=request.media_path, media_sha256=request.media_sha256)
        from .task_inputs import FrozenTaskContext
        context = FrozenTaskContext(material.visual_embeddings, material.images, material.image_sizes,
                                    material.observed_seconds, material.sampled_frame_times, material.time_message)
        observations = material.observations
        enabled = getattr(self.bridge, "prediction_evidence_enabled", True)
        if enabled and (observations is None or observations.features.shape[1] < 1):
            raise PredictionExecutionError("HIVAU evidence reader omitted observed media blocks")
        inputs = self.prompt_tokenizer.encode(question=request.question, context=context)
        with torch.no_grad():
            output = self.bridge.generate(inputs, observations if enabled else None, enabled=enabled, generation_config=self.generation_config)
        token_ids = output[0].detach().cpu().tolist()
        text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        return {"text": text, "token_ids": token_ids}
