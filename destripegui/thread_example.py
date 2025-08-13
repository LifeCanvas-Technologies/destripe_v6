import tkinter as tk
import threading
import time
import queue

# Create a shared queue
task_queue = queue.Queue()

def producer_task():
    """Simulates a task that produces data and puts it into the queue."""
    for i in range(5):
        item = f"Data item {i}"
        task_queue.put(item)
        print(f"Produced: {item}")
        time.sleep(1) # Simulate some processing time
    task_queue.put(None) # Sentinel value to signal the end of tasks

def consumer_task(gui_label):
    """Retrieves data from the queue and updates the GUI."""
    while True:
        item = task_queue.get()
        if item is None: # Exit if sentinel value is received
            print("Consumer received None, exiting.")
            break
        print(f"Consumed: {item}")
        gui_label.config(text=f"Processed: {item}") # Update GUI label
        task_queue.task_done() # Signal that the task is done

def other_task():
    for i in range(1, 10):
        print(i)
        time.sleep(.5)

def start_threads_with_queue():
    """Starts the producer and consumer threads."""
    threads = threading.enumerate()
    for t in threads:
        print('Thread: ----- {}'.format(t))
    number_of_threads = threading.active_count()
    print('current threads: {}'.format(number_of_threads))
    if number_of_threads < 2:
        print('starting thread')
        thread = threading.Thread(target=other_task, daemon=True)
        thread.start()
    root.after(1000, start_threads_with_queue)

    # producer_thread = threading.Thread(target=producer_task, daemon=True) # Set as daemon
    # consumer_thread = threading.Thread(target=consumer_task, args=(gui_label,), daemon=True) # Set as daemon

    # producer_thread.start()
    # consumer_thread.start()
    # print("Producer and consumer threads started!")

root = tk.Tk()
root.title("Tkinter Threading with Queue Example")

status_label = tk.Label(root, text="Waiting for tasks...")
status_label.pack(pady=20)

start_button = tk.Button(root, text="Start Producer/Consumer", command=lambda: start_threads_with_queue())
start_button.pack()

root.after(1000, start_threads_with_queue)
root.mainloop()

