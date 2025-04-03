import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk,ImageDraw
import json
import os

class BoundingBoxEditor:
    def __init__(self, image_path, output_path):
        self.image_path = image_path
        self.output_path = output_path
        self.image = Image.open(image_path)
        self.clone = self.image.copy()
        self.drawing = False
        self.x1, self.y1 = -1, -1
        self.x2, self.y2 = -1, -1
        self.bboxes = []  # save bounding boxes
        # set up the GUI
        self.window = tk.Tk()
        self.window.title("Bounding Box Editor")

        self.canvas = tk.Canvas(self.window, width=self.image.width, height=self.image.height)
        self.canvas.pack()

        self.save_button = tk.Button(self.window, text="Save Bounding Boxes", command=self.save_bboxes)
        self.save_button.pack()

        self.load_image()
        self.window.mainloop()

    def load_image(self):
        # set image to tkinter canvas
        self.image_tk = ImageTk.PhotoImage(self.image)
        self.canvas.create_image(0, 0, anchor="nw", image=self.image_tk)

        # bind mouse events
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

    def on_mouse_down(self, event):
        self.drawing = True
        self.x1, self.y1 = event.x, event.y

    def on_mouse_move(self, event):
        if self.drawing:
            self.x2, self.y2 = event.x, event.y
            self.redraw()

    def on_mouse_up(self, event):
        self.drawing = False
        self.x2, self.y2 = event.x, event.y
        # scale_w, scale_h = self.image.width / 1000, self.image.height / 1000
        # y1, x1, y2, x2 = (int(coord * scale) for coord, scale in zip(item["box_2d"], [scale_h, scale_w, scale_h, scale_w]))
        self.bboxes.append([self.x1, self.y1, self.x2, self.y2])
        self.redraw()

    def redraw(self):
        # reload the original image
        self.image = self.clone.copy()

        # draw all existing bounding boxes
        for bbox in self.bboxes:
            self.draw_bbox(bbox[0], bbox[1], bbox[2], bbox[3])

        # draw the current bounding box
        if self.drawing:
            self.draw_bbox(self.x1, self.y1, self.x2, self.y2)

        # update the canvas
        self.image_tk = ImageTk.PhotoImage(self.image)
        self.canvas.create_image(0, 0, anchor="nw", image=self.image_tk)

    def draw_bbox(self, x1, y1, x2, y2):
        draw = ImageDraw.Draw(self.image)
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

    def save_bboxes(self):
        # save the bounding boxes to a JSON file
        if not self.bboxes:
            messagebox.showerror("Error", "No bounding boxes to save.")
            return
        
        data = {"image_id": os.path.basename(self.image_path), "bboxes": self.bboxes}
        # with open(self.output_path, 'w') as json_file:
        #     json.dump(data, json_file, indent=4)
        # messagebox.showinfo("Success", "Bounding boxes saved successfully!")
        return self.bboxes

# run the GUI
# image_path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
# output_path = filedialog.asksaveasfilename(title="Save Bounding Boxes", defaultextension=".json", filetypes=[("JSON files", "*.json")])

# if image_path and output_path:
#     editor = BoundingBoxEditor(image_path, output_path)