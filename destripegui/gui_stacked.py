import tkinter as tk
from tkinter import ttk

def create_scrollable_treeview(parent, columns, data, label_text):
    # Outer frame to hold label + Treeview + Scrollbars
    outer_frame = tk.Frame(parent)

    # Label above the Treeview
    label = tk.Label(outer_frame, text=label_text, font=("Arial", 12, "bold"))
    label.pack(anchor="w", pady=(0, 2))

    # Frame for Treeview + Scrollbars
    frame = tk.Frame(outer_frame)

    # Create Treeview
    tree = ttk.Treeview(frame, columns=columns, show='headings', height=10)

    # Define headings and column widths
    for col in columns:
        tree.heading(col, text=col)
        tree.column(col, width=100, anchor="center")

    # Insert data rows
    for row in data:
        tree.insert("", tk.END, values=row)

    # Add vertical scrollbar
    vsb = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=vsb.set)

    # Add horizontal scrollbar
    hsb = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
    tree.configure(xscrollcommand=hsb.set)

    # Layout inside frame
    tree.grid(row=0, column=0, sticky="nsew")
    vsb.grid(row=0, column=1, sticky="ns")
    hsb.grid(row=1, column=0, sticky="ew")

    frame.grid_rowconfigure(0, weight=1)
    frame.grid_columnconfigure(0, weight=1)
    frame.pack(fill="both", expand=True)

    return outer_frame

# Sample data
sample_data1 = [(f"R{i}C1", f"R{i}C2", f"R{i}C3", f"R{i}C4", f"R{i}C5") for i in range(1, 21)]
sample_data2 = [(f"A{i}C1", f"A{i}C2", f"A{i}C3", f"A{i}C4", f"A{i}C5") for i in range(1, 15)]

# Main window
root = tk.Tk()
root.title("Two Scrollable Multi-Column Listboxes with Labels")

columns = ("Col1", "Col2", "Col3", "Col4", "Col5")

# First Treeview with label
tree_frame1 = create_scrollable_treeview(root, columns, sample_data1, "First Listbox")
tree_frame1.pack(fill="both", expand=True, padx=5, pady=5)

# Second Treeview with label
tree_frame2 = create_scrollable_treeview(root, columns, sample_data2, "Second Listbox")
tree_frame2.pack(fill="both", expand=True, padx=5, pady=5)

root.mainloop()
