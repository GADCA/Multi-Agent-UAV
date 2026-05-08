"""LLM controller with async multiprocessing worker."""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from multiprocessing import Process, Queue
from queue import Empty
from pathlib import Path
from typing import Any

from openai import OpenAI
from sim.global_state_manager import GlobalStateManager
from sim.uav import UAV


@dataclass
class LLMConfig:
    provider: str
    api_key: str
    api_key_env: str
    base_url: str
    model: str
    temperature: float
    response_format_json: bool
    trigger_interval_sec: float
    trigger_interval_frames: int
    trigger_on_events: bool
    deepseek_reasoning_effort: str
    deepseek_enable_thinking: bool
    enabled: bool
    heartbeat_interval_sec: float
    llm_request_timeout_sec: float
    llm_debug_logs: bool
    llm_env_file: str

    @classmethod
    def from_json_file(cls, path: str) -> "LLMConfig":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)

        if "profiles" in d:
            active_profile = str(d.get("active_profile", "siliconflow_flash"))
            profiles = d.get("profiles", {})
            if active_profile not in profiles:
                raise ValueError(f"active_profile '{active_profile}' not found in llm_config.json profiles.")
            p = profiles[active_profile]
            provider = str(p.get("provider", "siliconflow"))
            api_key = str(p.get("api_key", ""))
            api_key_env = str(p.get("api_key_env", ""))
            base_url = str(p.get("base_url", "https://api.siliconflow.cn/v1"))
            model = str(p.get("model", "deepseek-ai/DeepSeek-V4-Flash"))
        else:
            # backward-compatible single-profile mode
            provider = str(d.get("provider", "siliconflow"))
            api_key = str(d.get("api_key", ""))
            api_key_env = str(d.get("api_key_env", "SILICONFLOW_API_KEY"))
            base_url = str(d.get("base_url", "https://api.siliconflow.cn/v1"))
            model = str(d.get("model", "deepseek-ai/DeepSeek-V4-Flash"))

        return cls(
            provider=provider,
            api_key=api_key,
            api_key_env=api_key_env,
            base_url=base_url,
            model=model,
            temperature=float(d.get("temperature", 0.1)),
            response_format_json=bool(d.get("response_format_json", True)),
            trigger_interval_sec=float(d.get("trigger_interval_sec", 10.0)),
            trigger_interval_frames=int(d.get("trigger_interval_frames", 500)),
            trigger_on_events=bool(d.get("trigger_on_events", True)),
            deepseek_reasoning_effort=str(d.get("deepseek_reasoning_effort", "high")),
            deepseek_enable_thinking=bool(d.get("deepseek_enable_thinking", True)),
            enabled=bool(d.get("enabled", False)),
            heartbeat_interval_sec=float(d.get("heartbeat_interval_sec", 5.0)),
            llm_request_timeout_sec=float(d.get("llm_request_timeout_sec", 60.0)),
            llm_debug_logs=bool(d.get("llm_debug_logs", True)),
            llm_env_file=str(d.get("llm_env_file", "llm.env")),
        )


def _load_local_env_file(env_file: str) -> None:
    p = Path(env_file)
    if not p.exists():
        return
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        k = key.strip()
        v = value.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


def _safe_parse_text(raw_text: str) -> dict[str, Any]:
    default = {"reasoning": "解析错误或网络异常，触发本能悬停", "commands": []}
    if not raw_text:
        return default
    text = raw_text.strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        text = match.group(0)
    try:
        data: Any = json.loads(text)
    except json.JSONDecodeError:
        return default
    if not isinstance(data, dict):
        return default
    reasoning = data.get("reasoning", "")
    commands = data.get("commands", [])
    if not isinstance(reasoning, str):
        reasoning = str(reasoning)
    if not isinstance(commands, list):
        commands = []
    commands = [c.strip() for c in commands if isinstance(c, str) and c.strip()]
    return {"reasoning": reasoning, "commands": commands}

def _llm_worker_loop(request_q: Queue, response_q: Queue) -> None:
    # Worker-local log file to capture child-process events (helps when stdout is not visible).
    log_path = Path("llm_worker.log")
    def _wlog(msg: str) -> None:
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        try:
            # Prefer append if available
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"[{ts}] {msg}\n")
        except Exception:
            pass

    try:
        with open(log_path, "a", encoding="utf-8"):
            pass
    except Exception:
        pass
    print("[LLM WORKER] 子进程已启动，等待请求...")
    _wlog("[LLM WORKER] started")
    while True:
        req = request_q.get()
        if req is None:
            print("[LLM WORKER] 收到退出信号。")
            _wlog("[LLM WORKER] received shutdown signal")
            break
        default = {"reasoning": "解析错误或网络异常，触发本能悬停", "commands": []}
        try:
            req_id = req.get("request_id", -1)
            print(f"[LLM WORKER] 收到请求 request_id={req_id}，准备调用模型...")
            _wlog(f"received request_id={req_id}")
            client_timeout = float(req.get("timeout", 120.0))  # Use explicit timeout
            _wlog(f"calling API model={req['model']} base_url={req['base_url']} timeout={client_timeout}s")
            
            client = OpenAI(
                api_key=req["api_key"] or os.environ.get(req["api_key_env"], ""),
                base_url=req["base_url"],
                timeout=client_timeout,
            )
            kwargs: dict[str, Any] = dict(
                model=req["model"],
                messages=[
                    {"role": "system", "content": req["system_prompt"]},
                    {"role": "user", "content": req["state_json"]},
                ],
                stream=False,
                temperature=req["temperature"],
            )
            if req["response_format_json"]:
                kwargs["response_format"] = {"type": "json_object"}
            if "api.deepseek.com" in req["base_url"]:
                kwargs["reasoning_effort"] = req["deepseek_reasoning_effort"]
                if req["deepseek_enable_thinking"]:
                    kwargs["extra_body"] = {"thinking": {"type": "enabled"}}

            response = None
            last_error: Exception | None = None
            for attempt in range(1, 4):
                _wlog(f"[ATTEMPT {attempt}/3] Starting API call (timeout={client_timeout}s)")
                
                try:
                    res = client.chat.completions.create(**kwargs)
                    response = res.choices[0].message.content or ""
                    _wlog(f"[ATTEMPT {attempt}/3] ✓ SUCCESS - API returned response")
                    break
                except Exception as api_err:
                    last_error = api_err
                    _wlog(f"[ATTEMPT {attempt}/3] ✗ FAILED - {type(api_err).__name__}: {api_err}")

                if attempt < 3:
                    wait_sec = 2 ** (attempt - 1)
                    _wlog(f"[ATTEMPT {attempt}/3] Waiting {wait_sec}s before retry...")
                    time.sleep(wait_sec)
                else:
                    _wlog(f"[ATTEMPT {attempt}/3] All retries exhausted, giving up")

            if response is None:
                raise last_error or Exception("Failed to get API response after 3 attempts")

            content = response or ""
            _wlog(f"completed request_id={req_id} (response_len={len(content)})")
            parsed = _safe_parse_text(content)
            response_q.put(
                {
                    "ok": True,
                    "decision": parsed,
                    "trigger_reason": req["trigger_reason"],
                    "request_id": req_id,
                }
            )
            print(f"[LLM WORKER] 请求完成 request_id={req_id}，结果已返回主进程。")
            _wlog(f"response queued for request_id={req_id}")
        except Exception as e:
            response_q.put(
                {
                    "ok": False,
                    "decision": default,
                    "trigger_reason": req.get("trigger_reason", "unknown"),
                    "request_id": req.get("request_id", -1),
                    "error": str(e),
                }
            )
            print(f"[LLM WORKER] 请求失败 request_id={req.get('request_id', -1)} error={e}")
            _wlog(f"request failed request_id={req.get('request_id', -1)} error={e}")


class LLMController:
    def __init__(self, config_path: str = "llm_config.json", prompt_path: str = "UAV_SKILLS.md") -> None:
        self.config = LLMConfig.from_json_file(config_path)
        _load_local_env_file(self.config.llm_env_file)
        self.enabled = self.config.enabled
        self.trigger_interval_sec = self.config.trigger_interval_sec
        self.trigger_interval_frames = self.config.trigger_interval_frames
        self.trigger_on_events = self.config.trigger_on_events
        self._elapsed_since_last_trigger = 0.0
        self._frames_since_last_trigger = 0
        self._first_trigger_pending = True
        self._inflight = False
        self._inflight_since = 0.0
        self._inflight_deadline = 0.0
        self._request_counter = 0
        self._inflight_request_id = -1
        self._last_decision_history: list[dict] = []  # Track previous decisions for coherence
        self._last_worker_status_display = 0.0  # Throttle worker status display (max once per 10s)
        self._last_displayed_inflight_id = -1  # Track which request_id we last displayed

        self.system_prompt = Path(prompt_path).read_text(encoding="utf-8")
        self.api_key = self.config.api_key.strip() or os.environ.get(self.config.api_key_env, "")
        self.api_key_configured = bool(self.api_key)

        print(f"[LLM] 当前激活模型: {self.config.model} | provider={self.config.provider} | base_url={self.config.base_url}")
        if self.config.llm_debug_logs:
            print(f"[LLM] 项目级环境变量文件: {self.config.llm_env_file}")
        if not self.api_key_configured and self.enabled:
            print(
                "[LLM WARNING] 未检测到 API Key。"
                f"请在 {self.config.llm_env_file} 里设置 {self.config.api_key_env}=<your_key>。"
            )

        self._request_q: Queue | None = None
        self._response_q: Queue | None = None
        self._worker: Process | None = None
        if self.enabled and self.api_key_configured:
            self._request_q = Queue()
            self._response_q = Queue()
            self._worker = Process(target=_llm_worker_loop, args=(self._request_q, self._response_q), daemon=True)
            self._worker.start()
            if self.config.llm_debug_logs:
                print(f"[LLM] 并行进程已启动，pid={self._worker.pid}")

    def _start_worker(self) -> None:
        """(Re)start the worker process and queues."""
        old_pid = getattr(self._worker, "pid", None)
        # Terminate old worker process if exists.
        if self._worker is not None:
            try:
                if self.config.llm_debug_logs:
                    print(f"[LLM] 终止旧子进程 pid={old_pid} 并重启。")
                self._worker.terminate()
                self._worker.join(timeout=2.0)
            except Exception:
                pass
        # Reuse existing queues if present, to avoid losing responses queued by previous worker.
        if self._request_q is None:
            self._request_q = Queue()
        if self._response_q is None:
            self._response_q = Queue()
        # Start new worker using the same queue objects.
        self._worker = Process(target=_llm_worker_loop, args=(self._request_q, self._response_q), daemon=True)
        self._worker.start()
        new_pid = getattr(self._worker, "pid", None)
        self._last_displayed_inflight_id = -1  # Reset display tracking
        print(f"[WORKER] 子进程已启动 pid={new_pid}")
        if self.config.llm_debug_logs:
            print(f"[LLM] 并行进程已启动，old_pid={old_pid} new_pid={new_pid}")

    def _should_trigger(self, global_state_manager: GlobalStateManager) -> bool:
        by_time = self._elapsed_since_last_trigger >= self.trigger_interval_sec
        by_frame = self._frames_since_last_trigger >= self.trigger_interval_frames
        by_event = self.trigger_on_events and len(global_state_manager.active_events) > 0
        return by_time or by_frame or by_event

    @staticmethod
    def _trim_reasoning(reasoning: str, max_len: int = 1000) -> str:
        r = (reasoning or "").replace("\n", " ").strip()
        return r if len(r) <= max_len else r[:max_len] + "..."

    def _submit_request(self, state_json: str, trigger_reason: str) -> None:
        if self._request_q is None:
            return
        self._request_counter += 1
        request_id = self._request_counter
        # Inject decision history into state_json for coherent reasoning
        import json as json_lib
        try:
            state_dict = json_lib.loads(state_json)
            state_dict["decision_history"] = self._last_decision_history
            state_json_with_history = json_lib.dumps(state_dict, ensure_ascii=False)
        except Exception:
            state_json_with_history = state_json
        self._request_q.put(
            {
                "request_id": request_id,
                "api_key": self.api_key,
                "api_key_env": self.config.api_key_env,
                "base_url": self.config.base_url,
                "model": self.config.model,
                "temperature": self.config.temperature,
                "response_format_json": self.config.response_format_json,
                "deepseek_reasoning_effort": self.config.deepseek_reasoning_effort,
                "deepseek_enable_thinking": self.config.deepseek_enable_thinking,
                "system_prompt": self.system_prompt,
                "state_json": state_json_with_history,
                "timeout": float(self.config.llm_request_timeout_sec),
                "trigger_reason": trigger_reason,
            }
        )
        self._inflight = True
        self._inflight_since = time.monotonic()
        self._inflight_deadline = self._inflight_since + self.config.llm_request_timeout_sec
        self._inflight_request_id = request_id
        try:
            qsize = self._request_q.qsize()
        except Exception:
            qsize = None
        worker_alive = bool(self._worker and self._worker.is_alive())
        worker_pid = getattr(self._worker, "pid", None)
        print(f"\n--- 唤醒大模型进行决策 --- request_id={request_id} 触发原因: {trigger_reason}")
        print(f"[LLM DEBUG] worker_pid={worker_pid} alive={worker_alive} request_qsize={qsize}")
        if self.config.llm_debug_logs:
            print(
                f"[LLM] 已发送状态JSON给模型（长度={len(state_json)}字符），等待返回..."
                f" timeout={self.config.llm_request_timeout_sec:.1f}s"
            )

    def _poll_response(self) -> dict[str, Any] | None:
        if self._response_q is None:
            return None
        try:
            resp = self._response_q.get_nowait()
        except Empty:
            # Silently wait for response without diagnostic logs
            return None
        # We got a response — clear inflight state and process it.
        self._inflight = False
        self._inflight_since = 0.0
        self._inflight_deadline = 0.0
        self._inflight_request_id = -1
        self._elapsed_since_last_trigger = 0.0
        self._frames_since_last_trigger = 0
        self._last_worker_status_display = 0.0
        if not resp.get("ok", False):
            print(f"[LLM ERROR] API请求异常 request_id={resp.get('request_id', -1)}: {resp.get('error', 'unknown')}")
        elif self.config.llm_debug_logs:
            print(f"[LLM] 收到模型响应 request_id={resp.get('request_id', -1)}，开始解析。")
        decision = resp.get("decision", {"reasoning": "解析错误或网络异常，触发本能悬停", "commands": []})
        reasoning = self._trim_reasoning(decision.get("reasoning", ""))
        commands = decision.get("commands", [])
        if not isinstance(commands, list):
            commands = []
        commands = [c for c in commands if isinstance(c, str) and c.strip()]
        print(f"[LLM 意图] {reasoning if reasoning else '（空）'}")
        print(f"[LLM 指令] {commands if commands else '[]'}")
        # Save this decision to history for next iteration's coherence
        self._last_decision_history.append({
            "reasoning": reasoning,
            "commands": commands,
            "timestamp_sec": round(time.monotonic(), 1)
        })
        # Keep only last 5 decisions to avoid token explosion
        if len(self._last_decision_history) > 5:
            self._last_decision_history = self._last_decision_history[-5:]
        return {"reasoning": reasoning, "commands": commands}

    def step(self, uav_list: list[UAV], dt: float, global_state_manager: GlobalStateManager) -> dict[str, Any]:
        if not self.enabled:
            return {"reasoning": "LLM未启用，跳过决策。", "commands": []}
        if not self.api_key_configured:
            return {"reasoning": "未配置API Key，触发本能悬停。", "commands": []}

        polled = self._poll_response()
        if polled is not None:
            return polled

        # Inflight watchdog: recover if worker/API hangs too long.
        if self._inflight:
            now = time.monotonic()
            elapsed = now - self._inflight_since
            
            # Throttled status display (once per 10s max, when request is still inflight)
            if now - self._last_worker_status_display >= 10.0:
                self._last_worker_status_display = now
                print(f"[WORKER] 请求中... request_id={self._inflight_request_id} (elapsed={elapsed:.1f}s / timeout={self.config.llm_request_timeout_sec:.1f}s)")
            
            if now >= self._inflight_deadline:
                print(
                    f"[LLM WARN] 请求超时 request_id={self._inflight_request_id} "
                    f"(elapsed={elapsed:.1f}s, timeout={self.config.llm_request_timeout_sec:.1f}s)，"
                    "重置inflight状态并等待下一次触发。"
                )
                # Reset inflight flags
                self._inflight = False
                self._inflight_since = 0.0
                self._inflight_deadline = 0.0
                self._inflight_request_id = -1
                self._elapsed_since_last_trigger = 0.0
                self._frames_since_last_trigger = 0
                self._last_worker_status_display = 0.0
                # Try to recover a potentially stuck worker by restarting it.
                try:
                    if self.config.llm_debug_logs:
                        print("[LLM] 超时发生，尝试重启子进程以恢复状态。")
                    self._start_worker()
                except Exception as e:
                    print(f"[LLM ERROR] 重启子进程失败: {e}")

            return {"reasoning": "", "commands": []}

        self._elapsed_since_last_trigger += float(dt)
        self._frames_since_last_trigger += 1

        trigger_reason = ""
        if self._first_trigger_pending:
            self._first_trigger_pending = False
            trigger_reason = "startup_immediate"
        elif self._should_trigger(global_state_manager):
            parts = []
            if self.trigger_on_events and len(global_state_manager.active_events) > 0:
                parts.append("active_events")
            if self._elapsed_since_last_trigger >= self.trigger_interval_sec:
                parts.append("timer")
            if self._frames_since_last_trigger >= self.trigger_interval_frames:
                parts.append("frame_fallback")
            trigger_reason = "+".join(parts) if parts else "policy_trigger"

        if trigger_reason and not self._inflight:
            state_json = global_state_manager.get_llm_state_json(uav_list=uav_list)
            self._elapsed_since_last_trigger = 0.0
            self._frames_since_last_trigger = 0
            self._submit_request(state_json, trigger_reason)

        return {"reasoning": "", "commands": []}
