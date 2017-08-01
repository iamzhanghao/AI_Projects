from tkinter import *
from tkinter import ttk, filedialog, messagebox
# Tk, Label, Button, filedialog, messagebox, Canvas
# import matplotlib as mp
from os import listdir
from PIL import ImageTk, Image


class GUI:
    def __init__(self, master):

        # Variables
        self.imageNameMsg = StringVar()
        self.entryFolderName = StringVar()
        self.entryResultName = StringVar()
        self.totalImgNumber = 0
        self.curImgNumber = 0
        self.imageList = []
        self.imageFolderDir = ''
        self.cancerResultList = []

        self.img = None
        self.img_label = None
        self.defaultImgDir = r"C:/Users/IceFox/AI_core/Project/GUI/TestGUI01/40X/"
        self.ImgDir = ""

        # GUI
        self.master = master
        self.master.resizable(width=FALSE, height=FALSE)
        self.master.title("Cancer Classifier")

        # Frame
        self.frame = ttk.Frame(self.master, padding=(5, 5, 12, 0))
        self.frame.pack(fill=BOTH, expand=1)

        for row in range(9):
            self.frame.grid_rowconfigure(row, minsize=15)
        for column in range(6):
            self.frame.grid_columnconfigure(column, minsize=15)
        self.frame.grid_columnconfigure(0, minsize=200)

        # Title (maybe not needed?)
        self.labelTitle = Label(self.frame, text="Cancer Classifier v1.0")
        self.labelTitle.grid(row=0, column=3, sticky=E)

        # dir entry & load
        self.labelFolder = Label(self.frame, text="Image Folder Dir:")
        self.labelFolder.grid(row=1, column=0, sticky=E)
        self.entryFolder = Entry(self.frame, textvariable=self.entryFolderName)
        self.entryFolder.grid(row=1, column=1, columnspan=4, sticky=W + E)
        self.btnFolder = Button(self.frame, text="Load", command=self.LoadFromDir)
        self.btnFolder.grid(row=1, column=5, sticky=W)
        self.btnFolder = Button(self.frame, text="DebugLoad", command=self.LoadDir)
        self.btnFolder.grid(row=1, column=6, sticky=W)

        self.labelResult = Label(self.frame, text="Classification Result Dir:")
        self.labelResult.grid(row=2, column=0, sticky=E)
        self.entryResultDir = Entry(self.frame, textvariable=self.entryResultName)
        self.entryResultDir.grid(row=2, column=1, columnspan=4, sticky=W + E)
        self.btnFolder = Button(self.frame, text="Save", command=self.LoadDir)
        self.btnFolder.grid(row=2, column=5, sticky=W)

        # main panel for labeling
        self.labelFolder = Label(self.frame, text="Image in the folder:")
        self.labelFolder.grid(row=3, column=0, sticky=S)
        lists = StringVar(value=self.imageList)
        self.imageListbox = Listbox(self.frame, listvariable=lists, height=5)
        self.imageListbox.grid(row=4, column=0, rowspan=5, sticky=(N, S, E, W))
        self.imageNameLabel = Label(self.frame, textvariable=self.imageNameMsg)
        self.imageNameLabel.grid(row=4, column=3, columnspan=3, sticky=W)

        # self.imageListbox.bind('<<ListboxSelect>>', self.DisplaySelectedImageName)
        self.imageListbox.bind('<Double-1>', self.ListboxSelected)

        # canvas
        self.canvas = Canvas(self.frame, background='white')
        self.canvas.grid(row=5, column=3, rowspan=2, columnspan=3)

        '''
        self.btnAdd = Button(self.frame, text = "Add", command = self.DebugIncreaseList)
        self.btnAdd.grid(row = 11, column = 3, sticky=(N,S,E,W))

        self.btnUpdate = Button(self.frame, text = "Update", command = self.UpdateListboxImage)
        self.btnUpdate.grid(row = 11, column = 4, sticky=(N,S,E,W))

        self.btnUpdate = Button(self.frame, text = "ImportFolder", command = self.ImportFolder)
        self.btnUpdate.grid(row = 11, column = 5, sticky=(N,S,E,W))

        self.btnClose = Button(self.frame, text = "Load", command = master.quit)
        self.btnClose.grid(row = 11, column = 6, sticky=(N,S,E,W))
        '''
        self.ctrPanel = Frame(self.frame)
        self.ctrPanel.grid(row=7, column=2, columnspan=5, sticky=W + E)
        self.prevBtn = Button(self.ctrPanel, text='<< Prev', width=10, command=self.prevImage)
        self.prevBtn.pack(side=LEFT, padx=5, pady=3)
        self.nextBtn = Button(self.ctrPanel, text='Next >>', width=10, command=self.nextImage)
        self.nextBtn.pack(side=LEFT, padx=5, pady=3)
        self.progLabel = Label(self.ctrPanel, text="Progress:     /    ")
        self.progLabel.pack(side=LEFT, padx=5)
        self.tmpLabel = Label(self.ctrPanel, text="Go to Image No.")
        self.tmpLabel.pack(side=LEFT, padx=5)
        self.idxEntry = Entry(self.ctrPanel, width=5)
        self.idxEntry.pack(side=LEFT)
        self.goBtn = Button(self.ctrPanel, text='Go', command=self.gotoImage)
        self.goBtn.pack(side=LEFT)

        self.classifyPanel = Frame(self.frame)
        self.classifyPanel.grid(row=9, column=2, columnspan=5, sticky=W + E)
        self.classifyBtn = Button(self.classifyPanel, text='Run Classification', width=25,
                                  command=self.classifySingleImage)
        self.classifyBtn.pack(side=LEFT, padx=5, pady=3)
        self.classifyAllBtn = Button(self.classifyPanel, text='Run Classification on All', width=30,
                                     command=self.classifyAllImages)
        self.classifyAllBtn.pack(side=LEFT, padx=5, pady=3)
        # self.progLabel = Label(self.ctrPanel, text = "Progress:     /    ")
        # self.progLabel.pack(side = LEFT, padx = 5)
        # self.tmpLabel = Label(self.ctrPanel, text = "Go to Image No.")
        # self.tmpLabel.pack(side = LEFT, padx = 5)
        # self.idxEntry = Entry(self.ctrPanel, width = 5)
        # self.idxEntry.pack(side = LEFT)
        # self.goBtn = Button(self.ctrPanel, text = 'Go', command = self.gotoImage)
        # self.goBtn.pack(side = LEFT)

    def prevImage(self, event=None):
        print("previous image")
        if self.curImgNumber > 0:
            self.curImgNumber -= 1
            self.imageListbox.see(self.curImgNumber)
            self.LoadImage(self.imageList[self.curImgNumber])

    def nextImage(self, event=None):
        print("next image")
        if self.curImgNumber < self.totalImgNumber - 1:
            self.curImgNumber += 1
            self.imageListbox.see(self.curImgNumber)
            self.LoadImage(self.imageList[self.curImgNumber])

    def gotoImage(self):
        # print("goto image")
        idx = int(self.idxEntry.get()) - 1
        if 0 <= idx and idx < self.totalImgNumber:
            self.curImgNumber = idx
            self.LoadImage(self.imageList[self.curImgNumber])

    def classifySingleImage(self):
        # print("classify single image")
        messagebox.showinfo("Results", "I don't know")

    def classifyAllImages(self):
        # print("classify all images")
        messagebox.showinfo("Results", "I don't know")

    def ImportFolder(self):
        self.imageFolderDir = listdir(self.defaultImgDir)
        self.imageList = []
        for image in self.imageFolderDir:
            # print(image)
            self.imageList.append(image)
        self.total = len(self.imageList)

    def DebugIncreaseList(self):
        print("increase")
        # print(self.imageList)

    def ListboxSelected(self, event=None):
        idxs = self.imageListbox.curselection()
        if len(idxs) == 1:
            idx = int(idxs[0])
            self.curImgNumber = idx
            # print("select index %d" % (idx))
            self.imageListbox.see(idx)
            name = self.imageList[idx]
            self.LoadImage(name)

    def LoadImage(self, name):

        if name != "":
            self.progLabel.config(text="%04d/%04d" % (self.curImgNumber + 1, self.totalImgNumber))
            self.imageNameMsg.set("Image Loaded: %s" % (name))

            if self.ImgDir != "":
                filename = self.ImgDir + name
            else:
                filename = self.defaultImgDir + name

            self.img = ImageTk.PhotoImage(Image.open(filename))
            # Canvas_Image = self.canvas.create_image(100,150,image = self.img)

            self.canvas.config(width=max(self.img.width(), 300), height=max(self.img.height(), 300))
            self.canvas.create_image(0, 0, image=self.img, anchor=NW)
            # self.canvas.config(text = "%04d/%04d" %(2, 5))

    '''
    def loadImage(self):

        idxs = self.imageListbox.curselection()
        if len(idxs) == 1:
            idx = int(idxs[0])
            self.imageListbox.see(idx)
            name = self.imageList[idx]
            self.imageNameMsg.set("Image Loaded: %s" % (name))

            if name != "":
                filename = self.initdir + name
                self.img = ImageTk.PhotoImage(Image.open(filename))
                self.canvas.config(width = max(self.img.width(), 400), height = max(self.img.height(), 400))
                self.canvas.create_image(0, 0, image = self.tkimg, anchor=NW)
                self.progLabel.config(text = "%04d/%04d" %(self.cur, self.total))
    '''

    def Classify(self):
        messagebox.showinfo("Results", "I don't know")
        self.updateListboxImage()

    def LoadFromDir(self, dbg=False):
        result = filedialog.askdirectory(title="Select a Folder to import images")
        if result != "":

            self.entryFolderName.set(result)
            print("Loaded from: " + result)
            self.ImgDir = result + "/"
            self.imageFolderDir = [f for f in listdir(self.ImgDir) if re.match(r'.*\.png', f)]
            if len(self.imageFolderDir) == 0:
                messagebox.showinfo("Warning", "No png files in selected folder.")
            self.imageList = []
            for image in self.imageFolderDir:
                print(image)
                self.imageList.append(image)
                self.UpdateListboxImage()
            self.totalImgNumber = len(self.imageList)
        else:
            messagebox.showinfo("Warning", "Folder not loaded")

    def LoadDir(self, dbg=False):
        self.imageFolderDir = listdir(self.initdir)
        self.imageList = []
        for image in self.imageFolderDir:
            print(image)
            self.imageList.append(image)
            self.UpdateListboxImage()
        self.totalImgNumber = len(self.imageList)

    def UpdateListboxImage(self):
        self.imageListbox.delete(0, END)
        for image in self.imageList:
            self.imageListbox.insert(END, image)


if __name__ == '__main__':
    root = Tk()
    gui = GUI(root)
    root.mainloop()