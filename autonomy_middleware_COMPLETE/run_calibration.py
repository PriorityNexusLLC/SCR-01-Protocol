"""
run_calibration.py — run the calibration tests without typing long commands.

Put this file in the SAME folder that contains the `autonomy_middleware`
folder, then run:

    python3 run_calibration.py

It will show a menu. Pick a number, press Enter.

No long command lines to copy, so nothing can get mangled by formatting.
"""

import sys
import time


def line():
    print("-" * 62)


def test_1_baseline():
    print("\nTEST 1 — Load Index resting baseline (X_base)")
    line()
    print("Make sure your computer is idle: close browsers, games, LM Studio.")
    print("Background things like antivirus are fine.")
    input("\nPress Enter when ready...")

    try:
        from autonomy_middleware.load_index import LoadIndexMonitor
    except ImportError as e:
        print(f"\nCouldn't load the middleware: {e}")
        print("You're probably in the wrong folder. This script must sit in the")
        print("folder that CONTAINS 'autonomy_middleware', not inside it.")
        return

    monitor = LoadIndexMonitor(max_context_tokens=62000)

    print(f"\nGPU detected: {monitor.gpu_available}")
    if not monitor.gpu_available:
        print("  No NVIDIA GPU found. Not fatal -- it will use CPU only --")
        print("  but tell Claude, since thermal calibration won't apply.")

    print("\nTaking three baseline readings, about a minute apart.")
    print("Please don't use the computer while this runs.\n")

    results = []
    for i in range(1, 4):
        print(f"  Reading {i} of 3...", end=" ", flush=True)
        monitor.X_base = None
        x = monitor.calibrate(samples=10)
        results.append(x)
        print(f"X_base = {x:.6f}")
        if i < 3:
            time.sleep(60)

    line()
    print("RESULT — send these three numbers to Claude:")
    for i, x in enumerate(results, 1):
        print(f"  Reading {i}: {x:.6f}")
    print(f"  GPU detected: {monitor.gpu_available}")
    line()

    # A near-zero baseline is the EXPECTED result on idle hardware, not a
    # fault. An earlier version of this check compared spread against a
    # fraction of the maximum, which fires trivially when every reading is
    # near zero -- it wrongly told the user to redo perfectly good runs.
    biggest = max(results)
    if biggest < 0.02:
        print("This is the expected result: your machine really is idle, so the")
        print("baseline is essentially zero. Λ measures load ABOVE this, so a")
        print("near-zero baseline is exactly what it should be.")
    elif (biggest - min(results)) > 0.05:
        print("NOTE: these readings differ by more than 0.05 in absolute terms,")
        print("which suggests something was running during one of them.")
        print("Worth redoing if you can — but send all the numbers either way.")
    else:
        print("Readings are consistent. Good baseline.")


def test_2_thermal():
    print("\nTEST 2 — GPU thermal range")
    line()
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    except Exception as e:
        print(f"Can't read GPU temperature: {e}")
        print("If you don't have an NVIDIA card, skip this test.")
        return

    def temp():
        return pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)

    print("Step 1 of 2: idle temperature.")
    print("Make sure LM Studio is NOT generating anything right now.")
    input("Press Enter when ready...")
    idle = temp()
    print(f"  Idle temperature: {idle} C")

    print("\nStep 2 of 2: temperature under load.")
    print("Now go to LM Studio and give Qwen something long, for example:")
    print('  "Write a detailed 2000-word explanation of how photosynthesis works."')
    print("\nStart it generating, then come back here and press Enter.")
    input("Press Enter once it's generating...")

    print("\nWatching temperature. Press Ctrl+C when generation finishes.\n")
    peak = idle
    try:
        while True:
            t = temp()
            peak = max(peak, t)
            print(f"  current {t} C   (peak so far {peak} C)")
            time.sleep(5)
    except KeyboardInterrupt:
        pass

    line()
    print("RESULT — send these to Claude:")
    print(f"  Idle temperature: {idle} C")
    print(f"  Peak temperature: {peak} C")
    print(f"  Range: {peak - idle} C")
    line()


def test_3_speed():
    print("\nTEST 3 — Generation speed baseline")
    line()
    print("This one you read off LM Studio's screen; nothing runs here.")
    print()
    print("For each of three rounds:")
    print("  1. Start a BRAND NEW chat in LM Studio (important -- long chats are slower)")
    print("  2. Paste this prompt:")
    print()
    print("     Write a 300-word description of a lighthouse at dawn.")
    print()
    print("  3. When it finishes, look at the small readout under the response.")
    print("     It looks something like:  5.40 tok/sec  1684 tokens  13.97s")
    print()

    rounds = []
    for i in range(1, 4):
        print(f"\n--- Round {i} of 3 ---")
        input("Press Enter once you have the numbers on screen...")
        try:
            tps = float(input("  tokens per second (e.g. 5.40): ").strip())
            toks = int(input("  token count (e.g. 1684): ").strip())
            secs = float(input("  seconds (e.g. 13.97): ").strip())
            rounds.append((tps, toks, secs))
        except ValueError:
            print("  Couldn't read that as a number -- skipping this round.")

    if not rounds:
        print("\nNo readings captured.")
        return

    line()
    print("RESULT — send this to Claude:")
    for i, (tps, toks, secs) in enumerate(rounds, 1):
        print(f"  Round {i}: {tps} tok/sec, {toks} tokens, {secs}s")
    avg = sum(r[0] for r in rounds) / len(rounds)
    print(f"  Average: {avg:.2f} tokens/sec")
    line()


def main():
    while True:
        print("\n" + "=" * 62)
        print("CALIBRATION TESTS")
        print("=" * 62)
        print("  1  Load Index resting baseline   (5 min, no LM Studio needed)")
        print("  2  GPU thermal range             (15 min, LM Studio needed)")
        print("  3  Generation speed baseline     (5 min, LM Studio needed)")
        print("  q  Quit")
        choice = input("\nPick one: ").strip().lower()

        if choice == "1":
            test_1_baseline()
        elif choice == "2":
            test_2_thermal()
        elif choice == "3":
            test_3_speed()
        elif choice in ("q", "quit", "exit"):
            print("\nDone. Send the results to Claude whenever you're ready.")
            return
        else:
            print("  Didn't recognise that -- type 1, 2, 3, or q.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nStopped.")
