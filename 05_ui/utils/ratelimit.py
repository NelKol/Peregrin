# debounce_throttle.py
import functools
import time
from typing import Callable, Optional

from shiny import reactive


def Debounce(delay_secs: float) -> Callable[[Callable[[], object]], Callable[[], object]]:
    """
    Decorator: delay propagating invalidations until `delay_secs` have passed
    *since the last* upstream invalidation.
    """
    def wrapper(f: Callable[[], object]) -> Callable[[], object]:
        when = reactive.Value[Optional[float]](None)
        trigger = reactive.Value(0)

        @reactive.calc
        def cached():
            # Wrap f in a calc so Shiny handles dependency tracking/caching.
            return f()

        @reactive.effect(priority=102)
        def primer():
            """
            Each time `cached()` invalidates, push the deadline out by `delay_secs`.
            """
            try:
                # Touch cached() to register dependency, but ignore its value/errors.
                cached()
            except Exception:
                ...
            finally:
                when.set(time.time() + delay_secs)

        @reactive.effect(priority=101)
        def timer():
            """
            If the deadline is in the future, wait; if it has passed, fire `trigger`.
            """
            deadline = when()
            if deadline is None:
                return

            remaining = deadline - time.time()
            if remaining <= 0:
                with reactive.isolate():
                    when.set(None)
                    trigger.set(trigger() + 1)
            else:
                reactive.invalidate_later(remaining)

        @reactive.calc
        @reactive.event(trigger, ignore_none=False)
        @functools.wraps(f)
        def debounced():
            return cached()

        return debounced

    return wrapper


def Throttle(delay_secs: float) -> Callable[[Callable[[], object]], Callable[[], object]]:
    """
    Decorator: propagate at most once every `delay_secs` even if upstream
    invalidates more frequently (first event gets through ASAP; subsequent ones
    are held until the window elapses).
    """
    def wrapper(f: Callable[[], object]) -> Callable[[], object]:
        last_signaled = reactive.Value[Optional[float]](None)  # when upstream last invalidated
        last_triggered = reactive.Value[Optional[float]](None) # when we last let a value through
        trigger = reactive.Value(0)

        @reactive.calc
        def cached():
            return f()

        @reactive.effect(priority=102)
        def primer():
            """
            Record that upstream invalidated (even if `cached()` errors).
            """
            try:
                cached()
            except Exception:
                ...
            finally:
                last_signaled.set(time.time())

        @reactive.effect(priority=101)
        def timer():
            """
            If we've never triggered, or the window has elapsed, trigger now.
            Otherwise, schedule a wake-up for the remaining window.
            """
            # Read to form dependencies
            signaled = last_signaled()
            triggered = last_triggered()

            # Nothing to do until we've seen at least one upstream signal.
            if signaled is None:
                return

            now = time.time()
            if triggered is None:
                # First-ever trigger goes through immediately.
                last_triggered.set(now)
                with reactive.isolate():
                    trigger.set(trigger() + 1)
                return

            elapsed = now - triggered
            if elapsed >= delay_secs:
                last_triggered.set(now)
                with reactive.isolate():
                    trigger.set(trigger() + 1)
            else:
                reactive.invalidate_later(delay_secs - elapsed)

        @reactive.calc
        @reactive.event(trigger, ignore_none=False)
        @functools.wraps(f)
        def throttled():
            return cached()

        return throttled

    return wrapper

def MakeThrottled(fn, delay_secs: float):
    """
    Return a callable that defers/merges calls to `fn` so that `fn` executes
    at most once per `delay_secs`. If multiple calls occur while waiting,
    the most recent args/kwargs are used.
    """
    last_fired = reactive.Value(None)          # type: Optional[float]
    pending = reactive.Value(None)             # type: Optional[tuple[tuple, dict]]
    tick = reactive.Value(0)                   # drives the worker

    def request(*args, **kwargs):
        # Record the latest request and poke the worker
        with reactive.isolate():
            pending.set((args, kwargs))
            tick.set(tick() + 1)

    @reactive.effect
    @reactive.event(tick, ignore_none=False)
    def _worker():
        job = pending()
        if job is None:
            return
        now = time.time()
        last = last_fired()
        if last is None or (now - last) >= delay_secs:
            # Fire now
            last_fired.set(now)
            args, kwargs = job
            fn(*args, **kwargs)
        else:
            # Not yet—wake up when the window elapses and re-check
            reactive.invalidate_later(delay_secs - (now - last))
            # Re-poke ourselves so the event fires after the sleep
            with reactive.isolate():
                tick.set(tick() + 1)

    return request

