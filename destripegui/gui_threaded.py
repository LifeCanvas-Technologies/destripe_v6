import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
import random
import queue

def default_worker(task_id, q, stop_event, delay_fn=lambda: 0.03 + random.random() * 0.02):
    # Simulate variable-length work with 100 steps
    for i in range(101):
        if stop_event.is_set():
            q.put(("status", task_id, "Cancelled"))
            q.put(("done", task_id, "cancelled"))
            return
        time.sleep(delay_fn())
        if i % 2 == 0:
            q.put(("progress", task_id, i))
            q.put(("status", task_id, f"Step {i}/100"))   
    q.put(("status", task_id, "Done"))
    q.put(("done", task_id, "success"))

def slower_worker(task_id, q, stop_event):
    # Same logic, just slower delays
    return default_worker(task_id, q, stop_event, delay_fn=lambda: 0.05 + random.random() * 0.05)

class TaskUI:
    def __init__(self, parent, title, task_id, message_queue, worker_func=None):
        self.parent = parent
        self.task_id = task_id
        self.q = message_queue
        self.stop_event = None
        self.thread = None
        self.worker_func = worker_func or default_worker

        frame = ttk.LabelFrame(parent, text=title, padding=10)
        frame.pack(fill="x", padx=10, pady=6)

        top = ttk.Frame(frame)
        top.pack(fill="x")
        self.start_btn = ttk.Button(top, text="Start", command=self.start_task)
        self.start_btn.pack(side="left")

        self.cancel_btn = ttk.Button(top, text="Cancel", command=self.cancel_task, state="disabled")
        self.cancel_btn.pack(side="left", padx=(6, 0))

        self.progress = ttk.Progressbar(frame, orient="horizontal", mode="determinate", length=400, maximum=100)
        self.progress.pack(fill="x", pady=8)

        self.status = ttk.Label(frame, text="Idle")
        self.status.pack(anchor="w")

    def start_task(self):
        if self.thread and self.thread.is_alive():
            return
        self.progress["value"] = 0
        self.status.config(text="Starting...")
        self.start_btn.config(state="disabled")
        self.cancel_btn.config(state="normal")
        self.stop_event = threading.Event()

        self.thread = threading.Thread(
            target=self.worker_func,
            args=(self.task_id, self.q, self.stop_event),
            daemon=True
        )
        self.thread.start()

    def cancel_task(self):
        if self.stop_event:
            self.stop_event.set()
            self.status.config(text="Cancelling...")

class App:
    def __init__(self, root):
        self.root = root
        root.title("Threaded Tasks with Progress Bars")
        root.geometry("520x280")
        self.q = queue.Queue()

        # Task A uses default speed
        self.task1 = TaskUI(root, "Task A (fast-ish)", "task_a", self.q, worker_func=default_worker)
        # Task B uses slower worker
        self.task2 = TaskUI(root, "Task B (slow-ish)", "task_b", self.q, worker_func=slower_worker)

        # Poll queue
        self.root.after(80, self.process_queue)

    def process_queue(self):
        print(random.random())
        try:
            while True:
                msg_type, task_id, payload = self.q.get_nowait()
                task = self.task1 if task_id == "task_a" else self.task2

                if msg_type == "progress":
                    task.progress["value"] = payload
                elif msg_type == "status":
                    task.status.config(text=str(payload))
                elif msg_type == "done":
                    task.start_btn.config(state="normal")
                    task.cancel_btn.config(state="disabled")
                    if payload == "success":
                        task.status.config(text="Completed ✓")
                    elif payload == "cancelled":
                        task.status.config(text="Cancelled ✖")
        except queue.Empty:
            pass
        finally:
            self.root.after(80, self.process_queue)

if __name__ == "__main__":
    root = tk.Tk()
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass
    App(root)
    root.mainloop()
