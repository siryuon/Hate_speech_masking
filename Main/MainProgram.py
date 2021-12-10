from tkinter import *
from utils import *

model = create_model(embedding_matrix = ft_model.wv.vectors, maxlen=35)
model.load_weights('./ckpt/ckpt')

root = Tk()
root.title('Text classification')
root.geometry('400x400')
root.resizable(False, False)

lbl1 = Label(root)
lbl1['text'] = 'Input your text'
lbl1.pack()

lbl2 = Label(root)
lbl3 = Label(root)

def result(n):
  lbl2.config(text='Origin Text: ' + ent.get())
  origin, masked = pred_mask(model, ent.get())
  lbl3.config(text='Masked Text: ' + masked)

ent = Entry(root)
ent.bind('<Return>', result)
ent.pack()
lbl2.pack()
lbl3.pack()
root.mainloop()
