import pandas as pd
from tkinter import Tk
from pandastable import Table

df = pd.DataFrame({'Name': ['Alice', 'Bob'], 'Score': [85, 90]})


def edit_table_gui(data_frame):
  root = Tk()
  root.title('Edit Table - Close window when done')
  frame = root.frame() if hasattr(root, 'frame') else None
  # Alternative using ttk/frame properly:
  import tkinter as tk

  frame = tk.Frame(root)
  frame.pack(fill='both', expand=True)
  table = Table(frame, dataframe=data_frame, showtoolbar=True, showindex=True)
  table.show()
  root.mainloop()
  return table.model.df


# Script pauses here until you close the GUI window
df = edit_table_gui(df)
print('Updated DataFrame:')
print(df)