"""
check_idle_floor.py — how long does your GPU take to reach true idle?

Run this with LM Studio loaded but NOT generating, and don't touch the
machine while it runs.

WHY: P_idle is subtracted from every power reading in s₁₄, so getting it
wrong biases every joules-per-token figure. On the reference machine the
same card with the same model resident read 2.66 W when genuinely idle and
34.8 W three seconds after activity — a 13x difference, and using the
second understates energy per token by roughly half.

The settle detection added to calibrate_idle waits for consecutive readings
to agree, and still returned 21.36 W. So either the card takes longer than
60s to drop, or 21 W IS its floor with a model resident and the 2.66 W was
measured before the weights were loaded.

This tells you which, by watching for two minutes.
"""
import time
try:
    import pynvml
except ImportError:
    raise SystemExit("nvidia-ml-py not installed")

pynvml.nvmlInit()
h = pynvml.nvmlDeviceGetHandleByIndex(0)

print("Watching GPU power for 120 seconds. Don't use the machine.\n")
print(f"  {'elapsed':>8} {'watts':>8}   {'note'}")
print("  " + "-" * 42)

start = time.perf_counter()
readings = []
last_print = 0.0
while True:
    el = time.perf_counter() - start
    if el > 120:
        break
    w = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
    readings.append((el, w))
    if el - last_print >= 5:
        recent = [x for t, x in readings if t > el - 5]
        note = ""
        if len(readings) > 40:
            older = [x for t, x in readings if el - 15 < t <= el - 10]
            if older and abs(sum(recent)/len(recent) - sum(older)/len(older)) < 0.5:
                note = "steady"
        print(f"  {el:>7.0f}s {w:>8.2f}   {note}")
        last_print = el
    time.sleep(0.2)

lo = min(w for _, w in readings)
hi = max(w for _, w in readings)
final = sum(w for t, w in readings if t > 110) / max(1, len([1 for t, _ in readings if t > 110]))
print(f"\n  minimum seen: {lo:.2f} W")
print(f"  maximum seen: {hi:.2f} W")
print(f"  last 10s mean: {final:.2f} W")
print()
if final <= 5:
    print("  The card DOES reach deep idle — it just needs longer than 60s.")
    print("  Raise max_wait_s in calibrate_idle().")
elif abs(final - lo) < 2:
    print(f"  {final:.2f} W appears to be the floor with a model resident.")
    print("  The earlier 2.66 W was probably measured before the weights loaded.")
    print("  Use this as P_idle — it is the correct baseline for a loaded model.")
else:
    print("  Still falling at 120s. The card is slow to drop clocks; measure")
    print("  idle after a longer wait, or accept a conservative P_idle and note it.")
