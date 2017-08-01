from tkinter import Tk, Label, Button, filedialog, messagebox

from PIL import ImageTk, Image


class GUI:
    def __init__(self, master):
        self.master = master
        master.title("Cancer Classifier")

        self.label = Label(master, text="This is our first GUI!")
        self.label.pack()

        self.greet_button = Button(master, text="Select File", command=self.select_file)
        self.greet_button.pack()

        self.classify_button = Button(master,text="Classify", command = self.classify)
        self.classify_button.pack()

        self.close_button = Button(master, text="Close", command=master.quit)
        self.close_button.pack()


        self.img = None
        self.img_label = None

    def select_file(self):
        filename = filedialog.askopenfilename(initialdir="/", title="Select file",
                                              filetypes=(("png files", "*.png"), ("all files", "*.*")))

        if filename !="":
            self.img = ImageTk.PhotoImage(Image.open(filename))

            # The Label widget is a standard Tkinter widget used to display a text or image on the screen.
            self.img_label = Label(self.master, image=self.img)

            # The Pack geometry manager packs widgets in rows or columns.
            self.img_label.pack()
            print(filename)

    def classify(self):
        messagebox.showinfo("Results", "I don'tknow")


root = Tk()
gui = GUI(root)
root.mainloop()
