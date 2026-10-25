"""Authenticated, fixed-profile AVT1 service. Importing this module loads no model.

The engine owns inference cancellation checkpoints. HTTP cancellation never
proves GPU completion: a separate joiner releases the slot after thread exit.
"""
from __future__ import annotations

import asyncio
from collections import OrderedDict
from contextlib import asynccontextmanager
from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
import numbers
import queue
import re
import secrets
import struct
import threading
from typing import Annotated
import uuid

from fastapi import Depends, FastAPI, Path, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from vtuber_engine_support import InferenceCancelled
from vtuber_profile import SERVICE_VERSION


MAX_TEXT_CHARS = 512
MAX_PCM_BYTES = 16777216
MAX_PCM_FRAME = 16384
MAX_REQUEST_BYTES = 8192
ID_PATTERN = r"^[A-Za-z0-9_.:-]{1,128}$"
HASH_PATTERN = r"^[a-f0-9]{64}$"


class SynthesisRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    request_id: str = Field(pattern=ID_PATTERN)
    speech_unit_hash: str = Field(pattern=HASH_PATTERN)
    text_hash: str = Field(pattern=HASH_PATTERN)
    text: str = Field(min_length=1, max_length=MAX_TEXT_CHARS)
    expected_instance_id: str = Field(min_length=1, max_length=128)
    expected_profile_fingerprint: str = Field(pattern=HASH_PATTERN)


class _Rejected(Exception):
    def __init__(self, status: int, code: str):
        self.status, self.code = status, code


class _InvalidAudio(Exception):
    pass


def _frame(kind: int, payload: bytes) -> bytes:
    return struct.pack(">BI", kind, len(payload)) + payload


def _json_frame(kind: int, value: dict) -> bytes:
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(payload) > 8192:
        raise RuntimeError("protocol metadata exceeds its fixed limit")
    return _frame(kind, payload)


@dataclass
class _Job:
    request: SynthesisRequest
    loop: asyncio.AbstractEventLoop
    cancel_event: threading.Event = field(default_factory=threading.Event)
    stopped: threading.Event = field(default_factory=threading.Event)
    response_done: threading.Event = field(default_factory=threading.Event)
    space_available: threading.Event = field(default_factory=threading.Event)
    frames: queue.Queue = field(default_factory=lambda: queue.Queue(maxsize=2))
    available: asyncio.Event = field(default_factory=asyncio.Event)
    outcome: str = "generation_failed"
    unsafe: bool = False
    worker: threading.Thread | None = None

    def notify(self) -> None:
        try:
            self.loop.call_soon_threadsafe(self.available.set)
        except RuntimeError:
            # Event-loop shutdown must not prevent model-thread cleanup.
            self.cancel_event.set()
            self.space_available.set()

    def cancel(self) -> None:
        self.cancel_event.set()
        self.space_available.set()
        self.notify()

    def wait_capacity(self) -> bool:
        while self.frames.full() and not self.cancel_event.is_set():
            self.space_available.clear()
            if self.frames.full() and not self.cancel_event.is_set():
                self.space_available.wait()
        return not self.cancel_event.is_set()

    def put(self, frame: bytes) -> bool:
        while not self.cancel_event.is_set():
            try:
                self.frames.put(frame, timeout=0.025)
                self.notify()
                return True
            except queue.Full:
                continue
        return False

    async def next_frame(self) -> bytes | None:
        while True:
            if self.cancel_event.is_set():
                return None
            if self.stopped.is_set() and self.outcome != "success":
                return None  # Never drain buffered audio after a known failure.
            try:
                frame = self.frames.get_nowait()
                self.space_available.set()
                return frame
            except queue.Empty:
                if self.stopped.is_set():
                    return None
            self.available.clear()
            # Recheck after clearing, including a producer racing this clear.
            if self.cancel_event.is_set() or self.stopped.is_set() or not self.frames.empty():
                continue
            await self.available.wait()


class _Service:
    def __init__(self, engine, profile: dict):
        self.engine = engine
        self.profile = deepcopy(profile)
        self.instance_id = uuid.uuid4().hex
        self.lock = threading.Lock()
        self.active: _Job | None = None
        self.finished: OrderedDict[str, None] = OrderedDict()
        self.quarantined = False
        self.shutting_down = False

    def capabilities(self) -> dict:
        return {"protocol": 1, "service_version": SERVICE_VERSION,
                "instance_id": self.instance_id, "profile_fingerprint": self.profile["fingerprint"],
                "format": "pcm_s16le", "channels": 1, "sample_width": 2,
                "max_text_chars": MAX_TEXT_CHARS, "max_pcm_bytes": MAX_PCM_BYTES,
                "streaming_options": deepcopy(self.profile["streaming_options"])}

    def start(self, request: SynthesisRequest) -> _Job:
        with self.lock:
            if self.quarantined or self.shutting_down:
                raise _Rejected(503, "instance_unavailable")
            if request.request_id in self.finished:
                raise _Rejected(409, "request_id_used")
            if self.active is not None:
                raise _Rejected(409, "busy")
            job = _Job(request, asyncio.get_running_loop())
            self.active = job  # Reserve before creating or starting any worker.
            worker = threading.Thread(target=self._run, args=(job,), name="vtuber-inference", daemon=True)
            job.worker = worker
            try:
                worker.start()
                threading.Thread(target=self._join, args=(job,), name="vtuber-inference-join", daemon=True).start()
            except BaseException:
                # A missing joiner cannot establish safe reusable ownership.
                self.quarantined = True
                job.cancel()
                raise _Rejected(503, "instance_unavailable") from None
            return job

    def cancel(self, request_id: str) -> dict:
        with self.lock:
            job = self.active if self.active and self.active.request.request_id == request_id else None
            if job is None:
                return {"state": "already_finished" if request_id in self.finished else "unknown",
                        "worker_stopped": request_id in self.finished}
        # Mutate this exact job, never whichever job happens to become active later.
        job.cancel()
        return {"state": "cancellation_requested", "worker_stopped": job.stopped.is_set()}

    def shutdown(self) -> None:
        with self.lock:
            self.shutting_down = True
            job = self.active
        if job:
            job.cancel()

    def _join(self, job: _Job) -> None:
        job.worker.join()
        with self.lock:
            if job.unsafe:
                self.quarantined = True
            # The event is set only after join, not inside the inference worker.
            job.stopped.set()
            self._release_finished(job)
        job.notify()

    def response_finished(self, job: _Job) -> None:
        with self.lock:
            job.response_done.set()
            self._release_finished(job)

    def _release_finished(self, job: _Job) -> None:
        # Caller holds self.lock. Slow responses also occupy the only slot, so
        # completed inference cannot create an unbounded population of senders.
        if not job.stopped.is_set() or not job.response_done.is_set() or self.active is not job:
            return
        self.active = None
        self.finished[job.request.request_id] = None
        while len(self.finished) > 128:
            self.finished.popitem(last=False)

    def _run(self, job: _Job) -> None:
        iterator = None
        rate = None
        total = sequence = 0
        natural_end = False
        try:
            if job.cancel_event.is_set():
                job.outcome = "cancelled"
                return
            params = deepcopy(self.profile["inference_params"])
            params.update(text=job.request.text, _cancel_event=job.cancel_event, _strict_errors=True)
            iterator = iter(self.engine.run(params))
            while not job.cancel_event.is_set():
                # Do not prefetch another engine output while both frame slots
                # are occupied; an output may itself contain several PCM frames.
                if not job.wait_capacity():
                    break
                try:
                    value = next(iterator)
                except StopIteration:
                    natural_end = True
                    break
                if job.cancel_event.is_set():
                    break
                if not isinstance(value, (tuple, list)) or len(value) != 2:
                    raise _InvalidAudio()
                sample_rate, audio = value
                if (isinstance(sample_rate, (bool, np.bool_)) or not isinstance(sample_rate, numbers.Integral)
                        or not 8000 <= sample_rate <= 192000 or not isinstance(audio, np.ndarray)
                        or audio.ndim != 1 or audio.dtype.kind != "i" or audio.dtype.itemsize != 2 or audio.size == 0):
                    raise _InvalidAudio()
                sample_rate = int(sample_rate)
                if rate is not None and rate != sample_rate:
                    raise _InvalidAudio()
                if total + audio.nbytes > MAX_PCM_BYTES:
                    raise _InvalidAudio()
                if rate is None:
                    rate = sample_rate
                    start = {"protocol": 1, "request_id": job.request.request_id,
                             "speech_unit_hash": job.request.speech_unit_hash, "text_hash": job.request.text_hash,
                             "instance_id": self.instance_id, "profile_fingerprint": self.profile["fingerprint"],
                             "sample_rate": rate, "channels": 1, "sample_width": 2, "format": "pcm_s16le"}
                    if not job.put(_json_frame(1, start)):
                        break
                pcm = audio.astype("<i2", copy=False).tobytes(order="C")
                for offset in range(0, len(pcm), MAX_PCM_FRAME):
                    part = pcm[offset:offset + MAX_PCM_FRAME]
                    if not job.put(_frame(2, struct.pack(">I", sequence) + part)):
                        break
                    total += len(part)
                    sequence += 1
            if job.cancel_event.is_set():
                job.outcome = "cancelled"
            elif natural_end and total:
                job.outcome = "success"
            else:
                job.outcome = "invalid_audio"
        except InferenceCancelled:
            job.outcome = "cancelled"
        except _InvalidAudio:
            job.outcome = "invalid_audio"
        except BaseException:
            job.outcome = "generation_failed"
            job.unsafe = True
        finally:
            if iterator is not None:
                try:
                    close = getattr(iterator, "close", None)
                    if close is not None:
                        close()
                except BaseException:
                    # Even InferenceCancelled from close does not prove the
                    # internal owner was cleaned up successfully.
                    job.outcome = "generation_failed"
                    job.unsafe = True


class _StreamResponse(Response):
    media_type = "application/x-ai-vtuber-pcm"

    def __init__(self, service: _Service, job: _Job):
        self.service = service
        self.job = job
        self.status_code = 200
        self.background = None
        self.init_headers({"cache-control": "no-store", "x-content-type-options": "nosniff"})

    async def _send_stream(self, send) -> None:
        await send({"type": "http.response.start", "status": self.status_code, "headers": self.raw_headers})
        await send({"type": "http.response.body", "body": b"AVT1", "more_body": True})
        digest = hashlib.sha256()
        pcm_bytes = chunks = 0
        while True:
            frame = await self.job.next_frame()
            if frame is None:
                break
            if self.job.cancel_event.is_set():
                break
            await send({"type": "http.response.body", "body": frame, "more_body": True})
            if frame[0] == 2:
                pcm = frame[9:]
                digest.update(pcm)
                pcm_bytes += len(pcm)
                chunks += 1
        req = self.job.request
        if self.job.stopped.is_set() and self.job.outcome == "success" and not self.job.cancel_event.is_set() and chunks:
            terminal = _json_frame(3, {"request_id": req.request_id, "speech_unit_hash": req.speech_unit_hash,
                                      "text_hash": req.text_hash, "pcm_bytes": pcm_bytes,
                                      "pcm_frames": pcm_bytes // 2, "chunks": chunks,
                                      "pcm_sha256": digest.hexdigest()})
        else:
            code = "cancelled" if self.job.cancel_event.is_set() else self.job.outcome
            terminal = _json_frame(4, {"request_id": req.request_id, "code": code})
        await send({"type": "http.response.body", "body": terminal, "more_body": False})

    async def __call__(self, scope, receive, send) -> None:
        async def disconnected():
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    self.job.cancel()
                    return

        sending = asyncio.create_task(self._send_stream(send))
        listening = asyncio.create_task(disconnected())
        response_delivered = False
        try:
            done, _ = await asyncio.wait({sending, listening}, return_when=asyncio.FIRST_COMPLETED)
            if listening in done:
                self.job.cancel()
                # Propagate genuine receive errors after requesting cleanup.
                await listening
            else:
                await sending
                response_delivered = True
        except OSError:
            self.job.cancel()
        finally:
            if not response_delivered:
                self.job.cancel()
            for task in (sending, listening):
                if not task.done():
                    task.cancel()
            await asyncio.gather(sending, listening, return_exceptions=True)
            self.service.response_finished(self.job)


def create_app(engine, profile: dict, token: str) -> FastAPI:
    """Build a service around an already-created fixed-profile synchronous engine."""
    if not isinstance(token, str) or not token or len(token) > 1024 or any(c.isspace() for c in token):
        raise ValueError("a nonempty bounded bearer token is required")
    if (not isinstance(profile, dict) or not isinstance(profile.get("fingerprint"), str)
            or not re.fullmatch(HASH_PATTERN, profile["fingerprint"])
            or not isinstance(profile.get("inference_params"), dict)
            or not isinstance(profile.get("streaming_options"), dict)):
        raise ValueError("a fixed fingerprint, inference_params and streaming_options are required")
    # Validate exposed metadata without inspecting/loading model assets.
    if len(json.dumps(profile["streaming_options"], allow_nan=False).encode()) > 4096:
        raise ValueError("streaming_options exceeds the capability metadata limit")
    service = _Service(engine, profile)
    expected_token = token.encode("utf-8")

    @asynccontextmanager
    async def lifespan(app):
        yield
        service.shutdown()

    app = FastAPI(title="AI VTuber stream service", version=SERVICE_VERSION, lifespan=lifespan,
                  docs_url=None, redoc_url=None, openapi_url=None)
    app.state.vtuber_service = service

    async def authenticate(request: Request):
        value = request.headers.get("authorization", "")
        scheme, separator, credential = value.partition(" ")
        if (not separator or scheme.lower() != "bearer" or len(credential) > 1024
                or not secrets.compare_digest(credential.encode("utf-8"), expected_token)):
            raise _Rejected(401, "unauthorized")

    @app.exception_handler(_Rejected)
    async def rejected(request, error):
        headers = {"WWW-Authenticate": "Bearer"} if error.status == 401 else None
        return JSONResponse({"error": error.code}, status_code=error.status, headers=headers)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request, error):
        return JSONResponse({"error": "invalid_request"}, status_code=422)

    auth = [Depends(authenticate)]

    @app.get("/ai-vtuber/v1/capabilities", dependencies=auth)
    async def capabilities():
        return service.capabilities()

    @app.post("/ai-vtuber/v1/cancel/{request_id}", dependencies=auth)
    async def cancel(request_id: Annotated[str, Path(pattern=ID_PATTERN)]):
        return service.cancel(request_id)

    @app.post("/ai-vtuber/v1/tts", dependencies=auth)
    async def tts(request: Request):
        data = bytearray()
        async for block in request.stream():
            if len(data) + len(block) > MAX_REQUEST_BYTES:
                raise _Rejected(400, "request_too_large")
            data.extend(block)
        try:
            req = SynthesisRequest.model_validate(json.loads(data))
        except (ValueError, ValidationError, RecursionError):
            raise _Rejected(422, "invalid_request") from None
        if not req.text.strip() or "\x00" in req.text:
            raise _Rejected(400, "invalid_text")
        if req.expected_instance_id != service.instance_id or req.expected_profile_fingerprint != service.profile["fingerprint"]:
            raise _Rejected(409, "identity_mismatch")
        try:
            text_bytes = req.text.encode("utf-8")
        except UnicodeEncodeError:
            raise _Rejected(422, "invalid_text") from None
        if hashlib.sha256(text_bytes).hexdigest() != req.text_hash:
            raise _Rejected(400, "text_hash_mismatch")
        return _StreamResponse(service, service.start(req))

    return app
