"""Bounded asynchronous playback for ACT's per-control-step EE deltas.

This is not diffusion/flow RTC: predictions are independent, without prefix
conditioning or blending. Align a replacement to the number of actions actually
executed since its observation, never sum skipped deltas or replay a delta while
waiting. Only the control thread owns the robot; the worker owns inference.
"""

from copy import deepcopy
from dataclasses import dataclass
import math
import queue
import threading
import time


@dataclass(frozen=True)
class Chunk:
    epoch: int
    step: int
    observed_at: float
    actions: tuple


class AsyncChunkInference:
    """One inference in flight, one completed result, and one active chunk.

    Call submit/action/reset from a single control thread. ``step`` counts motor
    actions actually sent, not wall-clock ticks: startup and position holds must
    not skip increments. Wall-clock age is enforced separately. A reset rejects
    an in-flight result without waiting for the model or touching its state.
    """

    def __init__(self, predict, *, replan_steps=10, max_age_s=3.0, clock=time.monotonic):
        if not isinstance(replan_steps, int) or replan_steps < 1:
            raise ValueError("replan_steps must be a positive integer")
        if not math.isfinite(max_age_s) or max_age_s <= 0:
            raise ValueError("max_age_s must be positive and finite")
        self.predict = predict
        self.replan_steps = replan_steps
        self.max_age_s = max_age_s
        self.clock = clock
        self._condition = threading.Condition()
        self._epoch = 0
        self._closed = False
        self._busy = False
        self._request = None
        self._result = None
        self._error = None
        self._active = None
        self._last_request_step = None
        self._last_action_step = -1
        self.inferences = 0
        self.discarded = 0
        self.latency_s = 0.0
        # Let an already-running native torch call finish before interpreter
        # teardown. Killing a daemon inside libtorch can abort the process.
        self._thread = threading.Thread(target=self._run, name="ACTChunkInference")
        self._thread.start()

    def submit(self, observation, step, observed_at):
        """Snapshot a fresh observation only when another inference is useful."""
        with self._condition:
            if self._closed or self._busy or self._result is not None or self._error is not None:
                return False
            if self.clock() - observed_at > self.max_age_s:
                return False
            if (self._active is not None and self._last_request_step is not None
                    and step - self._last_request_step < self.replan_steps):
                return False
            self._request = (self._epoch, step, observed_at, deepcopy(observation))
            self._last_request_step = step
            self._busy = True
            self._condition.notify()
            return True

    def action(self, step):
        """Return one finite seven-wide delta, or None to hold position.

        Late rows are discarded when a new chunk replaces the old future. No
        averaging is applied to rotations or gripper deltas. Existing downstream
        EE bounds, IK and joint clamps must still be applied by the caller.
        """
        with self._condition:
            if self._closed:
                return None
            if self._error is not None:
                raise RuntimeError("ACT inference worker failed") from self._error
            now = self.clock()
            if self._result is not None:
                candidate = self._result
                self._result = None
                index = step - candidate.step
                if (candidate.epoch == self._epoch and 0 <= index < len(candidate.actions)
                        and now - candidate.observed_at <= self.max_age_s):
                    self._active = candidate
                else:
                    self.discarded += 1
            if self._active is None:
                return None
            index = step - self._active.step
            if now - self._active.observed_at > self.max_age_s or index >= len(self._active.actions):
                self._active = None
                return None
            if index < 0 or step <= self._last_action_step:
                raise ValueError("Action cursor must advance after each executed delta")
            self._last_action_step = step
            return self._active.actions[index]

    def reset(self):
        """Invalidate all motion at halt/resume, including unfinished inference."""
        with self._condition:
            self._epoch += 1
            self._active = None
            self._result = None
            self._last_request_step = None
            self._last_action_step = -1
            # A worker failure is terminal; reset must not silently clear it.

    def fresh(self):
        """Recheck the selected chunk after potentially slow IK."""
        with self._condition:
            return (not self._closed and self._active is not None
                    and self.clock() - self._active.observed_at <= self.max_age_s)

    def close(self, timeout=5.0):
        """Stop requests and wait at most timeout; return whether the worker exited.

        The worker never owns hardware. If native inference outlasts this wait,
        Python waits for that call at process exit, after robot disconnection.
        """
        with self._condition:
            self._closed = True
            self._active = None
            self._result = None
            self._condition.notify_all()
        self._thread.join(timeout=timeout)
        return not self._thread.is_alive()

    def _run(self):
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._closed or self._request is not None)
                if self._closed:
                    return
                epoch, step, observed_at, observation = self._request
                self._request = None
            started = self.clock()
            try:
                rows = tuple(tuple(float(x) for x in row) for row in self.predict(observation))
                if not rows or any(len(row) != 7 or not all(map(math.isfinite, row)) for row in rows):
                    raise ValueError("ACT chunk must be nonempty, finite and seven-wide")
                result = Chunk(epoch, step, observed_at, rows)
            except Exception as exc:
                with self._condition:
                    self._error = exc
                    self._busy = False
                return
            with self._condition:
                self.inferences += 1
                self.latency_s = self.clock() - started
                self._busy = False
                if self._closed:
                    return
                if epoch == self._epoch:
                    self._result = result
                else:
                    self.discarded += 1


def run_async_control(
    *, engine, events, observe, prepare_action, send_action, hold_position,
    reset_motion=lambda: None,
    fps=30.0, duration=0.0, starvation_timeout=3.0,
    clock=time.monotonic, sleep=time.sleep,
):
    """Run robot callbacks only on the calling thread, with bounded action age.

    ``observe`` returns (policy_frame, raw_robot_observation). ``prepare_action``
    applies EE safety and IK; ``send_action`` applies the robot's joint clamp.
    Check the input queue again after potentially slow observation/IK callbacks.
    """
    running = False
    step = 0
    started = clock()
    waiting_since = None

    def consume_events():
        nonlocal running, step, waiting_since
        changed = False
        while True:
            try:
                event = events.get_nowait()
            except queue.Empty:
                return True, changed
            if event == "stop":
                engine.reset()
                reset_motion()
                hold_position()
                return False, True
            running = not running
            engine.reset()
            reset_motion()
            step = 0
            waiting_since = None
            changed = True
            if not running:
                hold_position()
            print("Running." if running else "Halted.")

    try:
        while duration <= 0 or clock() - started < duration:
            tick = clock()
            keep_going, _ = consume_events()
            if not keep_going:
                return
            if not running:
                sleep(min(0.05, 1.0 / fps))
                continue
            observed_at = clock()
            frame, raw = observe()
            keep_going, changed = consume_events()
            if not keep_going:
                return
            if changed:
                continue
            action = engine.action(step)
            engine.submit(frame, step, observed_at)
            if action is None:
                if waiting_since is None:
                    waiting_since = clock()
                    hold_position()
                    print("Holding position while waiting for a fresh policy chunk.")
                elif clock() - waiting_since > starvation_timeout:
                    raise RuntimeError("No fresh policy action within the inference wait budget")
            else:
                target = prepare_action(action, raw)
                keep_going, changed = consume_events()
                if not keep_going:
                    return
                if changed:
                    continue
                # A slow IK solve must not resurrect a result past its age limit.
                if not engine.fresh():
                    engine.reset()
                    reset_motion()
                    hold_position()
                    continue
                send_action(target)
                step += 1
                waiting_since = None
            sleep(max(0.0, 1.0 / fps - (clock() - tick)))
    finally:
        engine.reset()
        reset_motion()
        hold_position()
