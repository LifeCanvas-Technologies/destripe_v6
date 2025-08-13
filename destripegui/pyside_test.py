import sys, time, threading
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QProgressBar, QLabel
)

# ---- Shared state ----
progress = [0, 0]
done = [False, False]
canceled = [False, False]
locks = [threading.Lock(), threading.Lock()]
threads = [None, None]

def run_task(idx: int, duration_ms: int):
    steps = 100
    delay = duration_ms / steps / 1000.0
    for i in range(steps + 1):
        with locks[idx]:
            if canceled[idx]:
                break
            progress[idx] = i
        time.sleep(delay)
    with locks[idx]:
        # Mark done only if not canceled
        done[idx] = not canceled[idx]

def start_task(idx: int, duration_ms: int, btn_start: QPushButton, btn_cancel: QPushButton, bar: QProgressBar):
    if not btn_start.isEnabled():
        return  # already running
    with locks[idx]:
        progress[idx] = 0
        done[idx] = False
        canceled[idx] = False
    bar.setRange(0, 100)
    bar.setValue(0)
    bar.setFormat("%p%")
    btn_start.setEnabled(False)
    btn_cancel.setEnabled(True)

    t = threading.Thread(target=run_task, args=(idx, duration_ms), daemon=True)
    threads[idx] = t
    t.start()

def cancel_task(idx: int):
    with locks[idx]:
        canceled[idx] = True

def make_ui():
    w = QWidget()
    w.setWindowTitle("PySide6: Threaded Tasks (no classes) + Cancel")

    # Widgets
    lbl1, btn1_start, btn1_cancel, bar1 = QLabel("Task 1"), QPushButton("Start"), QPushButton("Cancel"), QProgressBar()
    lbl2, btn2_start, btn2_cancel, bar2 = QLabel("Task 2"), QPushButton("Start"), QPushButton("Cancel"), QProgressBar()
    for bar in (bar1, bar2):
        bar.setRange(0, 100); bar.setValue(0)
    for b in (btn1_cancel, btn2_cancel):
        b.setEnabled(False)

    # Layout
    v = QVBoxLayout(w)

    r1 = QHBoxLayout()
    r1.addWidget(lbl1); r1.addStretch(1)
    r1.addWidget(btn1_start); r1.addWidget(btn1_cancel)
    v.addLayout(r1); v.addWidget(bar1)

    r2 = QHBoxLayout()
    r2.addWidget(lbl2); r2.addStretch(1)
    r2.addWidget(btn2_start); r2.addWidget(btn2_cancel)
    v.addLayout(r2); v.addWidget(bar2)

    # Button wiring
    btn1_start.clicked.connect(lambda: start_task(0, 3000, btn1_start, btn1_cancel, bar1))
    btn2_start.clicked.connect(lambda: start_task(1, 5000, btn2_start, btn2_cancel, bar2))
    btn1_cancel.clicked.connect(lambda: cancel_task(0))
    btn2_cancel.clicked.connect(lambda: cancel_task(1))

    # UI updater timer
    def tick():
        for idx, (btn_start, btn_cancel, bar) in enumerate((
            (btn1_start, btn1_cancel, bar1),
            (btn2_start, btn2_cancel, bar2),
        )):
            with locks[idx]:
                p = progress[idx]
                is_done = done[idx]
                is_canceled = canceled[idx]

            # Update progress
            bar.setValue(p)

            # Handle completion
            if (is_done or is_canceled) and not btn_start.isEnabled():
                if is_canceled:
                    bar.setFormat("Canceled")
                else:
                    bar.setFormat("Done")
                    bar.setValue(100)
                btn_start.setEnabled(True)
                btn_cancel.setEnabled(False)

    timer = QTimer(w)
    timer.timeout.connect(tick)
    timer.start(50)  # ~20 fps

    return w

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = make_ui()
    window.resize(520, 220)
    window.show()
    sys.exit(app.exec())
