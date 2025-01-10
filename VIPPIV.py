import os.path
import time
import tkinter as tk
from tkinter import Toplevel, filedialog
from tkinter import simpledialog
from tkinter.ttk import Progressbar
import csv
import cv2.aruco
import numpy as np
from PIL import Image, ImageTk
import cv2
from openpiv import pyprocess
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import copy
from mpl_toolkits.axes_grid1 import ImageGrid
import screeninfo
import matplotlib.colors as mcol
from libpysal.weights import lat2W
from esda.moran import Moran
from aicspylibczi import CziFile

# Enable this when compiling for windows
# ctypes.windll.shcore.SetProcessDpiAwareness(2)



class PIV_gui:
    def __init__(self,window):
        self.root = window
        self.root.title('VIPPIV')

        # Get the screen which contains top
        self.current_screen = self.get_monitor_from_coord(window.winfo_x(), window.winfo_y())

        # Get the monitor's size
        self.width = int(self.current_screen.width / 4)

        self.actions_index = 0
        self.frame_index = 0
        self.loaded_file = False
        self.frames = []
        self.actions = {}
        self.object_detector_test = None
        self.piv_canvas = None
        self.toolbar = None
        self.results = []
        self.Vmean = None
        self.Umean = None
        self.quiver_x = None
        self.quiver_y = None
        self.first_run = True
        self.dragging = None
        self.buttons = {}
        self.setting_buttons = {}

        self.file_name_label = tk.Label(window, text="")
        self.file_name_label.grid(row=5, columnspan=5, padx=2, pady=5)

        self.title = tk.Label(window, text="Visual Interactive Program for Particle Image Velocimetry")
        self.title.grid(row=0, column=0)

        self.actions_frame = tk.Frame(self.root, bg="lightgrey", relief=tk.RIDGE, bd=2, width=220, height=200)
        self.actions_frame.grid(row=4, column=2, padx=10, pady=10, sticky="nsew")
        self.actions_frame.grid_propagate(False)
        self.actions_frame.update_idletasks()  # Ensure dimensions are set

        self.settings_frame = tk.Frame(self.root, bg="lightgrey", relief=tk.RIDGE, bd=2, width=220, height=200)
        self.settings_frame.grid(row=4, column=3, padx=10, pady=10, sticky="nsew")
        self.settings_frame.grid_propagate(False)
        self.settings_frame.update_idletasks()

        self.button_dilate = tk.Button(text='add_dilation', width=20, command=self.dilation_window)
        self.button_dilate.grid(row=1, column=1, padx=2, pady=5)

        self.button_erode = tk.Button(text='add_erosion', width=20, command=self.erosion_window)
        self.button_erode.grid(row=1, column=2, padx=2, pady=5)

        self.button_apply_mask = tk.Button(text='apply_mask', width=20, command=self.apply_mask_window)
        self.button_apply_mask.grid(row=1, column=3, padx=2, pady=5)

        self.button_object_detector = tk.Button(text='Add object detector', width=20,command=self.object_detector_window)
        self.button_object_detector.grid(row=2, column=1, padx=2, pady=5)

        self.button_remove_latest_actions = tk.Button(text='remove action', width=20, command=self.remove)
        self.button_remove_latest_actions.grid(row=2, column=3, padx=2, pady=5)

        self.button_perform_piv = tk.Button(text='PIV analysis', width=20, command=self.piv_thread)
        self.button_perform_piv.grid(row=2, column=2, padx=2, pady=5)

        self.button_moran_index = tk.Button(text='Calculate Morans Index', width=20, command=self.Morans_I)
        self.button_moran_index.grid(row=3, column=5, padx=2, pady=5)
        self.button_moran_index.grid_remove()

        self.button_save_matrix = tk.Button(text='Save arrow information', width=20, command=self.save_matrices)
        self.button_save_matrix.grid(row=5, column=5, padx=2, pady=5)
        self.button_save_matrix.grid_remove()

        self.button_regenerate_plot = tk.Button(text='regenerate_plot', width=20, command=self.regenerate_plot_window)
        self.button_regenerate_plot.grid(row=1, column=5, padx=2, pady=5)
        self.button_regenerate_plot.grid_remove()

        self.piv_check = tk.IntVar()
        self.cb = tk.Checkbutton(window, text="Show PIV", variable=self.piv_check, command=self.piv_display)
        self.cb.grid(row=2, column=5, padx=2, pady=5)
        self.cb.grid_remove()

        self.progress = Progressbar(window, orient="horizontal", length=100, mode='determinate')
        self.progress.grid(row=6, column=2, padx=2, pady=5)
        self.progress.grid_remove()

        self.time_left = tk.Label(window, text="Estimated time left= ")
        self.time_left.grid(row=7, column=2, padx=2, pady=5)
        self.time_left.grid_remove()

        self.button_open = tk.Button(text='Open video file', width=20, command=self.read_video)
        self.button_open.grid(row=0, column=2, padx=2, pady=5)

        self.button_left = tk.Button(text='<--', width=8, command=self.left)
        self.button_left.grid(row=4, column=1, padx=2, pady=5)

        self.button_right = tk.Button(text='-->', width=8, command=self.right)
        self.button_right.grid(row=4, column=4, padx=2, pady=5)

        self.square = Image.fromarray(np.zeros((480, self.width)))
        self.imgtk_sq = ImageTk.PhotoImage(image=self.square)

        self.display_left = tk.Label(window)
        self.display_left.grid(row=4, column=0, padx=2, pady=5)
        self.display_left.imgtk = self.imgtk_sq
        self.display_left.configure(image=self.imgtk_sq)

        self.display_right = tk.Label(window)
        self.display_right.grid(row=4, column=5, padx=2, pady=5)
        self.display_right.imgtk = self.imgtk_sq
        self.display_right.configure(image=self.imgtk_sq)

        self.frame_label = tk.Label(window)
        self.frame_label.grid(row=3, column=2)


    def get_monitor_from_coord(self,x, y):
        monitors = screeninfo.get_monitors()

        for m in reversed(monitors):
            if m.x <= x <= m.width + m.x and m.y <= y <= m.height + m.y:
                return m
        return monitors[0]


    def video_stream(self,frame):
        """
        This function takes a video frame and displays this on the left display field
        :param frame: A single frame from some video
        :return:
        """
        img = Image.fromarray(frame)
        max_width = self.width

        pixels_x, pixels_y = tuple([int(max_width / img.size[0] * x) for x in img.size])

        img = img.resize((pixels_x, pixels_y))
        imgtk = ImageTk.PhotoImage(image=img)
        self.display_left.imgtk = imgtk
        self.display_left.configure(image=imgtk)

    def video_output(self, frame):
        """
        This function takes a video frame and displays this on the right output field
        :param frame: A single frame from some video
        :return:
        """
        img = Image.fromarray(frame)
        max_width = self.width

        pixels_x, pixels_y = tuple([int(max_width / img.size[0] * x) for x in img.size])

        img = img.resize((pixels_x, pixels_y))
        imgtk = ImageTk.PhotoImage(image=img)
        self.display_right.imgtk = imgtk
        self.display_right.configure(image=imgtk)

    def read_video(self):
        """
        Ask the user to provide a path to a video.
        Open this video using cv2 and give the first frame to the display function.
        If this function is called while a PIV analysis has been performed in the runtime, reset the GUI to the start
        layout.
        :return:
        """
        self.Vmean = None
        self.Umean = None
        video_name = filedialog.askopenfilename()
        if video_name != '':

            self.frames = []
            if video_name.endswith('.mp4') or video_name.endswith('.avi'):
                cap = cv2.VideoCapture(video_name)
                succes = True
                j = 0
                while succes:
                    succes, image1 = cap.read()
                    if succes:
                        image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
                        self.frames.append(image1)
                        j += 1
            elif video_name.endswith('.czi'):
                czi_stack = CziFile(video_name)
                n_frames = czi_stack.get_dims_shape()[0]['T'][1]
                for i in range(n_frames):
                    frame = np.squeeze(czi_stack.read_image(T=i, C=0, S=0)[0]).astype(np.float32)
                    frame = ((frame - np.min(frame)) / (np.max(frame) - np.min(frame)) * 255).astype(np.uint8)
                    self.frames.append(frame)

            self.loaded_file = True
            if not self.first_run:
                self.display_right.lift()
                self.piv_canvas.get_tk_widget().destroy()
                self.toolbar.destroy()
                self.cb.grid_remove()
                self.button_moran_index.grid_remove()
                self.button_regenerate_plot.grid_remove()
                self.button_save_matrix.grid_remove()

            self.file_name_label.config(text=os.path.basename(video_name))
            self.frame_index = 0
            self.video_stream(self.frames[self.frame_index])

    def left(self):
        """
        This function is called when the button with the left arrow is called and when possible displays the previous
        frame for some video
        :return:
        """

        if self.loaded_file:
            if self.frame_index != 0:
                self.frame_index -= 1

                self.frame_label.config(text="Frame: " + str(self.frame_index))
                frame = self.frames[self.frame_index]
                self.video_stream(frame)
                output_frame = self.apply_preprocessing(frame, self.object_detector_test)

                self.video_output(output_frame)

    def right(self):
        """
        This function is called when the button with the left arrow is called and when possible displays the next frame
        for some video
        :return:
        """

        if self.loaded_file:

            if not self.frame_index + 1 >= len(self.frames):
                self.frame_index += 1

                self.frame_label.config(text="Frame: " + str(self.frame_index))

                frame = self.frames[self.frame_index]
                self.video_stream(frame)
                output_frame = self.apply_preprocessing(frame, self.object_detector_test)

                self.video_output(output_frame)

    def dilation_window(self):
        """
        When the user presses the dilation preprocessing button this function calles a subwindow asking for settings for
        the dilation operation. These settings are then stored and displayed in the center of the GUI
        :return:
        """
        if len(self.frames) != 0:

            def submit():

                if not self.mandatory_value_check(dilate_window,kernel_size_entry_x.get(),kernel_size_entry_y.get()):
                    input_kernel_x = kernel_size_entry_x.get()
                    input_kernel_y = kernel_size_entry_y.get()
                    self.actions["dilate "+ str(self.actions_index)] = [input_kernel_x, input_kernel_y]
                    self.actions_index += 1
                    self.change_actions()
                    dilate_window.destroy()

            dilate_window = Toplevel(self.root)
            x = self.root.winfo_x()
            y = self.root.winfo_y()
            dilate_window.geometry("+%d+%d" % (x, y))
            dilate_window.transient(self.root)
            dilate_window.grab_set()

            label_rows = tk.Label(dilate_window, text="kernel_size")

            default_x = tk.StringVar(dilate_window, value="")
            default_y = tk.StringVar(dilate_window, value="")
            kernel_size_entry_x = tk.Entry(dilate_window, textvariable=default_x)
            kernel_size_entry_y = tk.Entry(dilate_window, textvariable=default_y)

            kernel_size_entry_x.grid(row=0, column=1, padx=2, pady=5)
            kernel_size_entry_y.grid(row=0, column=2, padx=2, pady=5)
            label_rows.grid(row=0, column=0, padx=2, pady=5)
            submit_button = tk.Button(dilate_window, text='Submit', command=submit)
            submit_button.grid()

    def erosion_window(self):
        """
        When the user presses the erosion preprocessing button this function calles a subwindow asking for settings for
        the erosion operation. These settings are then stored and displayed in the center of the GUI
        :return:
        """
        if len(self.frames) != 0:


            def submit():

                if not self.mandatory_value_check(erode_window, kernel_size_entry_x.get(), kernel_size_entry_y.get()):
                    input_kernel_x = kernel_size_entry_x.get()
                    input_kernel_y = kernel_size_entry_y.get()

                    self.actions["erosion "+ str(self.actions_index)] = [input_kernel_x, input_kernel_y]
                    self.actions_index += 1
                    self.change_actions()
                    erode_window.destroy()

            erode_window = Toplevel(self.root)
            x = self.root.winfo_x()
            y = self.root.winfo_y()
            erode_window.geometry("+%d+%d" % (x, y))
            erode_window.transient(self.root)
            erode_window.grab_set()

            label_rows = tk.Label(erode_window, text="kernel_size")

            default_x = tk.StringVar(erode_window, value="")
            default_y = tk.StringVar(erode_window, value="")
            kernel_size_entry_x = tk.Entry(erode_window, textvariable=default_x)
            kernel_size_entry_y = tk.Entry(erode_window, textvariable=default_y)

            kernel_size_entry_x.grid(row=0, column=1, padx=2, pady=5)
            kernel_size_entry_y.grid(row=0, column=2, padx=2, pady=5)

            label_rows.grid(row=0, column=0, padx=2, pady=5)
            submit_button = tk.Button(erode_window, text='Submit', command=submit)
            submit_button.grid()

    def apply_mask_window(self):
        """
        When the user presses the apply mask preprocessing button this function the preprocssing step is stored and
        displayed in the center window.
        :return:
        """
        if len(self.frames) != 0:

            self.actions["apply_mask "+ str(self.actions_index)] = []
            self.actions_index+= 1
            self.change_actions()

    def object_detector_window(self):
        """
        When the user presses the object detector preprocessing button this function calles a subwindow asking for settings for
        the object detection operation. These settings are then stored and displayed in the center of the GUI
        :return:
        """
        if len(self.frames) != 0:

            def submit():

                threshold_value = threshold_value_entry.get()
                self.actions["object_detector "+ str(self.actions_index)] = [threshold_value]
                self.actions_index+= 1
                self.change_actions()
                self.object_detector_test = cv2.createBackgroundSubtractorMOG2(history=2, varThreshold=int(threshold_value),
                                                                          detectShadows=False)
                OD_window.destroy()

            OD_window = Toplevel(self.root)
            x = self.root.winfo_x()
            y = self.root.winfo_y()
            OD_window.geometry("+%d+%d" % (x, y))
            OD_window.transient(self.root)
            OD_window.grab_set()

            label_rows = tk.Label(OD_window, text="var_threshold")

            default = tk.StringVar(OD_window, value="100")
            threshold_value_entry = tk.Entry(OD_window, textvariable=default)

            threshold_value_entry.grid(row=0, column=1, padx=2, pady=5)
            label_rows.grid(row=0, column=0, padx=2, pady=5)
            submit_button = tk.Button(OD_window, text='Submit', command=submit)
            submit_button.grid()


    def change_actions(self):
        """
        This function is called whenever the user add a preprocessing step and stores and displays the steps with a
        hidden unique ID in the central field
        """

        # Remove existing buttons
        for button in self.buttons.values():
            button.destroy()
        self.buttons.clear()

        # Recreate buttons in the current order of self.operations
        for i, (op, params) in enumerate(self.actions.items()):
            button = tk.Button(
                self.actions_frame,
                text=op.split(" ")[0],
                bg="lightblue",
            )
            # Set initial position using `place`
            button.place(x=10, y=i * 40, width=200, height=30)
            # Bind drag-and-drop events
            button.bind("<ButtonPress-1>", self.start_drag)
            button.bind("<B1-Motion>", self.on_drag)
            button.bind("<ButtonRelease-1>", self.end_drag)
            button.bind("<Button-3>", lambda event, op=op: self.open_config(op))

            self.buttons[op] = button


        # Remove existing buttons
        for button in self.setting_buttons.values():
            button.destroy()
        self.setting_buttons.clear()

        # Recreate buttons in the current order of self.operations

        for i, (op, params) in enumerate(self.actions.items()):
            if op.split(" ")[0] == "erosion" or op.split(" ")[0] == "dilate":
                setting_button_name = "kernel size = " + params[0] + ',' + str(params[1])

            elif op.split(" ")[0] == "object_detector":
                setting_button_name = params[0]

            else:
                setting_button_name = "-"

            button = tk.Button(
                self.settings_frame,
                text=setting_button_name,
                bg="lightblue",
                state="disabled",
                disabledforeground="black",
            )
            # # Set initial position using `place`
            button.place(x=10, y=i * 40, width=200, height=30)

            self.setting_buttons[op] = button


    def start_drag(self, event):
        """Start dragging a draggable button."""

        widget = event.widget
        self.dragging = widget
        self.dragging.startX = event.x
        self.dragging.startY = event.y

    def on_drag(self, event):
        """Handle dragging motion."""
        if not self.dragging:
            return

        # Calculate new position relative to the frame
        new_x = self.dragging.winfo_x() + (event.x - self.dragging.startX)
        new_y = self.dragging.winfo_y() + (event.y - self.dragging.startY)

        # Define buffer space around the frame
        buffer = 20  # Allow 20 pixels of extra space outside the frame

        # Keep the button within the extended bounds of the frame
        frame_width = self.actions_frame.winfo_width()
        frame_height = self.actions_frame.winfo_height()
        new_x = max(-buffer, min(frame_width - self.dragging.winfo_width() + buffer, new_x))
        new_y = max(-buffer, min(frame_height - self.dragging.winfo_height() + buffer, new_y))

        # Update the widget's position
        self.dragging.place(x=new_x, y=new_y)

    def end_drag(self, event):
        """End drag and update the order of operations."""
        if not self.dragging:
            return

        # Determine the new order based on button positions
        button_positions = []
        for op, button in self.buttons.items():
            x, y = button.winfo_x(), button.winfo_y()
            button_positions.append((y, op))
        button_positions.sort()  # Sort by vertical position

        # Update self.operations to reflect the new order
        new_operations = {op: self.actions[op] for _, op in button_positions}
        self.actions = new_operations

        # Recreate buttons in the new order
        self.change_actions()
        self.dragging = None

    def open_config(self,op):
        """Open a dialog to configure hyperparameters."""
        if "apply_mask" not in op:

            def submit_two_values():
                if not self.mandatory_value_check(new_value_window,kernel_size_entry_x.get(),kernel_size_entry_y.get()):
                    input_kernel_x = kernel_size_entry_x.get()
                    input_kernel_y = kernel_size_entry_y.get()
                    self.actions[op] = [input_kernel_x, input_kernel_y]
                    self.change_actions()
                    new_value_window.destroy()


            def submit_one_value():
                if not self.mandatory_value_check(new_value_window,threshold_value_entry.get()):
                    input_x = threshold_value_entry.get()
                    self.actions[op] = [input_x]
                    self.change_actions()
                    if self.actions[op]!= "-":
                        self.object_detector_test = cv2.createBackgroundSubtractorMOG2(history=2,
                                                                                       varThreshold=int(input_x),
                                                                                       detectShadows=False)
                    new_value_window.destroy()



            new_value_window = Toplevel(self.root)
            x = self.root.winfo_x()
            y = self.root.winfo_y()
            new_value_window.geometry("+%d+%d" % (x, y))
            new_value_window.transient(self.root)

            label_rows = tk.Label(new_value_window, text="kernel_size")

            if len(self.actions[op]) == 2:
                submit_button = tk.Button(new_value_window, text='Submit', command=submit_two_values)
                default_x = tk.StringVar(new_value_window, value="")
                default_y = tk.StringVar(new_value_window, value="")
                kernel_size_entry_x = tk.Entry(new_value_window, textvariable=default_x)
                kernel_size_entry_y = tk.Entry(new_value_window, textvariable=default_y)
                kernel_size_entry_x.grid(row=0, column=1, padx=2, pady=5)
                kernel_size_entry_y.grid(row=0, column=2, padx=2, pady=5)
                label_rows.grid(row=0, column=0, padx=2, pady=5)
            else:
                label_rows = tk.Label(new_value_window, text="value")
                default = tk.StringVar(new_value_window, value="100")
                threshold_value_entry = tk.Entry(new_value_window, textvariable=default)
                threshold_value_entry.grid(row=0, column=1, padx=2, pady=5)
                label_rows.grid(row=0, column=0, padx=2, pady=5)
                submit_button = tk.Button(new_value_window, text='Submit', command=submit_one_value)

            submit_button.grid()

    def remove(self):
        """
        When this function is called a subwindow is called which allows the user to select the preprocessing steps to
        remove from the pipeline
        :return:
        """
        if len(self.actions.keys()) != 0 and len(self.frames) != 0:
            remove_window = Toplevel(self.root)
            x = self.root.winfo_x()
            y = self.root.winfo_y()
            remove_window.geometry("+%d+%d" % (x, y))
            remove_window.transient(self.root)
            remove_window.grab_set()

            vars = []
            for i in range(len(self.actions.keys())):
                var = tk.IntVar()
                cb = tk.Checkbutton(remove_window, text=list(self.actions.keys())[i].split(" ")[0], variable=var)
                cb.grid(row=i, column=0, padx=10, sticky='w')
                vars.append(var)

            def submit():
                # extract roll numbers for checked checkbuttons
                result = [var.get() for var in vars]
                # actions = [_[j] for j,_ in enumerate(list(actions.keys())) if not result[j]]

                for j, action in enumerate(list(self.actions.keys())):
                    if result[j]:
                        del (self.actions[action])
                self.change_actions()
                remove_window.destroy()

            submit_button = tk.Button(remove_window, text='Submit', command=submit)
            submit_button.grid()

    def apply_preprocessing(self, frame, object_detector):
        """
        This function takes a frame and returns this same frame with the correct preprocessing applied.
        :param frame: A single frame from a video
        :param object_detector: A cv2 objectdetector object used for preprocessing
        :return:
        """
        original_frame = copy.copy(frame)

        settings = []
        for action, value in self.actions.items():
            settings.append(value)

        counter = 0

        for key, value in self.actions.items():

            if key.split(" ")[0] == "erosion":
                frame = self.erosion(frame, int(value[0]), int(value[1]))

            elif key.split(" ")[0] == "dilate":
                frame = self.dilation(frame, int(value[0]), int(value[1]))

            elif key.split(" ")[0] == "apply_mask":
                frame = self.apply_mask(frame, original_frame)

            elif key.split(" ")[0] == "object_detector":
                frame = self.object_detector_func(frame, object_detector)
            counter += 1
        return frame

    def piv_thread(self):
        """
        This function display a subwindow with some settings for the PIV analysis and then calles the PIV function with
        these settings.
        :return:
        """
        self.first_run = False

        def submit():
            if not self.mandatory_value_check(piv_window, window_size_entry.get(), overlap_size_entry.get()):
                window_size = window_size_entry.get()
                overlap_size = overlap_size_entry.get()

                self.perform_piv(int(window_size), int(overlap_size))
                piv_window.destroy()

        piv_window = Toplevel(self.root)
        x = self.root.winfo_x()
        y = self.root.winfo_y()
        piv_window.geometry("+%d+%d" % (x, y))
        piv_window.transient(self.root)
        piv_window.grab_set()

        label_window = tk.Label(piv_window, text="window_size")
        label_overlap = tk.Label(piv_window, text="overlap_size")

        default_win_size = tk.StringVar(piv_window, value="")
        default_overlap_size = tk.StringVar(piv_window, value="")

        window_size_entry = tk.Entry(piv_window, textvariable=default_win_size)
        overlap_size_entry = tk.Entry(piv_window, textvariable=default_overlap_size)

        window_size_entry.grid(row=0, column=1, padx=2, pady=5)
        overlap_size_entry.grid(row=1, column=1, padx=2, pady=5)

        label_window.grid(row=0, column=0, padx=2, pady=5)
        label_overlap.grid(row=1, column=0, padx=2, pady=5)

        submit_button = tk.Button(piv_window, text='Submit', command=submit)
        submit_button.grid()

    def perform_piv(self, win_size, overlap):
        """
        This function performs a full Particle Image Velocimetry (PIV) analysis on the video currently opened in the GUI.
        For each frame n and frame n+1 it calculates PIV with the end result being an average over these results.
        The first 2 frames are skipped as to give the object detectors 2 frames to instatiate.
        This function als calculates and display the plotted results.

        :param win_size: The size of the windows the video is subdivided into
        :param overlap: The percentage of overlap for the search windows 50 percent is commonly used
        :return:
        """


        self.results = []
        var_threshold = 100
        for key, value in self.actions.items():

            if key.split(" ")[0] == "object_detector":
                var_threshold = int(value[0])
                break
        object_detector_1 = cv2.createBackgroundSubtractorMOG2(history=2, varThreshold=var_threshold,
                                                               detectShadows=False)

        self.frame_index = 0

        self.progress.grid()
        self.time_left.grid()
        preprocessed_frames = []
        for i in range(len(self.frames)):
            preprocessed_frames.append(self.apply_preprocessing(self.frames[i],object_detector_1))

        if len(self.frames) != 0:
            total_len = len(self.frames)
            U = []
            V = []
            for i in range(total_len - 1):
                time_start = time.time()
                self.progress['value'] = 100 * ((i + 1) / total_len)

                # image1 = apply_preprocessing(image1, object_detector_1)
                # image2 = apply_preprocessing(image2, object_detector_1)
                if i != 0 or i != 1:
                    image1 = preprocessed_frames[i]
                    image2 = preprocessed_frames[i + 1]
                    image1 = cv2.cvtColor(image1, cv2.COLOR_GRAY2RGB)
                    image2 = cv2.cvtColor(image2, cv2.COLOR_GRAY2RGB)
                    u, v, s2n = pyprocess.extended_search_area_piv(image1.sum(axis=2),
                                                                   image2.sum(axis=2), window_size=win_size,
                                                                   overlap=overlap, dt=1)
                    self.quiver_x, self.quiver_y = pyprocess.get_coordinates(image1.shape[:2], search_area_size=win_size,
                                                     overlap=overlap)

                    U.append(u)
                    V.append(v)
                time_end = time.time()
                timer = int((time_end - time_start) * (total_len - i))
                self.time_left.config(text="Estimated time left = " + str(timer) + " seconds")
                self.root.update()

            U = np.stack(U)
            U = np.nan_to_num(U)
            self.Umean = np.mean(U, axis=0)

            V = np.stack(V)
            V = np.nan_to_num(V)
            self.Vmean = np.mean(V, axis=0)

            fig, ax = plt.subplots()
            grid = ImageGrid(fig, 111,
                             nrows_ncols=(1, 1),
                             axes_pad=0.05,
                             cbar_location="right",
                             cbar_mode="single",
                             cbar_size="5%",
                             cbar_pad=0.05
                             )

            # Umean = np.flip(Umean, [1])
            # Vmean = np.flip(Vmean, [0])
            #
            # x = np.flip(x, [1])
            self.quiver_y = np.flip(self.quiver_y, [0])

            pyth_matrix = np.sqrt(self.Umean ** 2 + self.Vmean ** 2)
            # Vmean[pyth_matrix < 1] = 0
            # Umean[pyth_matrix < 1] = 0
            max_num = np.amax(pyth_matrix)

            cmap = mcol.LinearSegmentedColormap.from_list("", ["blue", 'violet', "red"])

            Q = grid[0].quiver(self.quiver_x, self.quiver_y, self.Umean, self.Vmean*-1, pyth_matrix, scale_units='dots', scale=2, width=.007, clim=[0, 50],
                               cmap=cmap)

            grid[0].axis('off')

            cbar = plt.colorbar(Q, cax=grid.cbar_axes[0])

            cbar.set_ticks(ticks=[0, 50], labels=['0', '50'])

            self.piv_canvas = FigureCanvasTkAgg(fig, master=self.root)
            self.piv_canvas.draw()
            self.piv_canvas.get_tk_widget().grid(row=4, column=5, padx=10, pady=10)
            self.toolbar = NavigationToolbar2Tk(self.piv_canvas, self.root, pack_toolbar=False)
            self.toolbar.update()
            self.toolbar.grid(row=6, column=5, padx=2, pady=5)
            self.cb.select()
            plt.close()
            self.progress.grid_remove()
            self.time_left.grid_remove()
            self.cb.grid()
            self.button_save_matrix.grid()
            self.button_moran_index.grid()
            self.button_regenerate_plot.grid()

            speed_matrix_out = np.flip(pyth_matrix, 1)

            degrees = np.degrees(np.arctan2(self.Umean, self.Vmean))
            degrees[degrees < 0] = 180 + (180 - abs(degrees[degrees < 0]))
            degrees = np.flip(degrees, 1)

            rowby_x = []
            disp = self.Vmean.shape[1] // 2
            for yd in range(disp):
                UV_row = []
                for yindex in range(len(self.Vmean[1]) - yd):
                    a1 = np.array([[U, V] for U, V in zip(self.Umean[:, yindex], self.Vmean[:, yindex])])
                    a2 = np.array([[U, V] for U, V in zip(self.Umean[:, yindex + yd], self.Vmean[:, yindex + yd])])
                    UV_row.append(self.corr_2d(a1, a2, self.Umean, self.Vmean))
                UV_row = np.array(UV_row)
                rowby_x.append(np.mean(UV_row, 0))

            rowby_x = np.transpose(rowby_x)

            w = lat2W(rowby_x.shape[0], rowby_x.shape[1], rook=False)
            mi = Moran(rowby_x, w)

            self.results.append(speed_matrix_out)
            self.results.append(degrees)
            self.results.append(np.flip(self.Umean,1))
            self.results.append(np.flip(self.Vmean,1))

            analysis = [mi.I, mi.p_norm, np.mean(pyth_matrix), np.std(pyth_matrix)]

            self.results.append(analysis)
            self.results.append(rowby_x)


    def regenerate_quiver_plot(self, scale):

        self.piv_canvas.get_tk_widget().destroy()
        self.toolbar.destroy()

        fig, ax = plt.subplots()
        grid = ImageGrid(fig, 111,
                         nrows_ncols=(1, 1),
                         axes_pad=0.05,
                         cbar_location="right",
                         cbar_mode="single",
                         cbar_size="5%",
                         cbar_pad=0.05
                         )

        cmap = mcol.LinearSegmentedColormap.from_list("", ["blue", 'violet', "red"])

        pyth_matrix = np.sqrt(self.Umean ** 2 + self.Vmean ** 2)

        Q = grid[0].quiver(self.quiver_x, self.quiver_y, self.Umean, self.Vmean * -1, pyth_matrix, scale_units='dots', scale=scale, width=.007,
                           clim=[0, 50],
                           cmap=cmap)
        grid[0].axis('off')

        cbar = plt.colorbar(Q, cax=grid.cbar_axes[0])

        cbar.set_ticks(ticks=[0, 50], labels=['0', '50'])

        self.piv_canvas = FigureCanvasTkAgg(fig, master=self.root)
        self.piv_canvas.draw()
        self.piv_canvas.get_tk_widget().grid(row=4, column=5, padx=10, pady=10)
        toolbar = NavigationToolbar2Tk(self.piv_canvas, self.root, pack_toolbar=False)
        toolbar.update()
        toolbar.grid(row=6, column=5, padx=2, pady=5)
        self.cb.select()
        plt.close()


    def regenerate_plot_window(self):
        """
        When the user presses the erosion preprocessing button this function calles a subwindow asking for settings for
        the erosion operation. These settings are then stored and displayed in the center of the GUI
        :return:
        """
        if len(self.frames) != 0:

            def submit():
                    input_scale = scale_entry.get()
                    plot_window.destroy()
                    self.regenerate_quiver_plot(float(input_scale))

            plot_window = Toplevel(self.root)
            x = self.root.winfo_x()
            y = self.root.winfo_y()
            plot_window.geometry("+%d+%d" % (x, y))
            plot_window.transient(self.root)
            plot_window.grab_set()

            label_rows = tk.Label(plot_window, text="quiver scale")

            default_scale = tk.StringVar(plot_window, value="2")
            scale_entry = tk.Entry(plot_window, textvariable=default_scale)

            scale_entry.grid(row=0, column=2, padx=2, pady=5)

            label_rows.grid(row=0, column=0, padx=2, pady=5)
            submit_button = tk.Button(plot_window, text='Submit', command=submit)
            submit_button.grid()


    def save_matrices(self):
        """
        This function is called when the user wants to save the results that make up the vectorfield plot.
        A path to some location is asked from the user after which 5 csv files are generated consisting of
        1. average distance
        2. average directions in degrees where 0 is straight upward and 180 is straight down
        3. The average X directions
        4. The average Y directions
        5. A file containing the associated Morans index, Morans index p value, the average speed and the standard deviation of the speed
        :return:
        """
        loc = filedialog.askdirectory()
        loc_directions = loc + "/arrow_directions.csv"
        loc_distance = loc + "/arrow_distance.csv"
        loc_meanX = loc + "/arrow_meanX.csv"
        loc_meanY = loc + "/arrow_meanY.csv"
        loc_summary = loc + "/summary.csv"

        with open(loc_distance, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(self.results[0])
        with open(loc_directions, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(self.results[1])
        with open(loc_meanX, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(self.results[2])
        with open(loc_meanY, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(self.results[3])
        with open(loc_summary, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["The Morans index value = ", self.results[4][0]])
            writer.writerow(["The Morans index associated P value = ", self.results[4][1]])
            writer.writerow(["The average movement speed = ", self.results[4][2]])
            writer.writerow(["The standard deviation of the movement speed = ", self.results[4][3]])

    def piv_display(self):
        """
        This function toggles between the PIV and the preprocessing output view by using a checkbox

        :return:
        """

        if self.piv_canvas != None and self.piv_check.get():
            # display_right.lower()
            # display_right.grid_remove()
            self.toolbar.grid()
            self.piv_canvas.get_tk_widget().grid()

            self.button_save_matrix.grid()
            self.button_moran_index.grid()
            self.button_regenerate_plot.grid()

        elif self.piv_canvas != None and not self.piv_check.get():
            # display_right.grid()
            self.toolbar.grid_remove()
            self.piv_canvas.get_tk_widget().grid_remove()

            self.button_save_matrix.grid_remove()
            self.button_moran_index.grid_remove()
            self.button_regenerate_plot.grid_remove()

    def Morans_I(self):
        """
        This function displays the Morans index, Morans index p value, average speed and standard deviation of the speed.
        It also allows the user to see the underlying heatmap on which the Morans index is calculated.
        :return:
        """

        def submit():
            morans_window.destroy()

        def show_plot():
            plt.imshow(self.results[5], origin='lower')
            plt.show()

        analysis = self.results[4]

        morans_window = Toplevel(self.root)
        morans_window.transient(self.root)
        morans_window.grab_set()
        moransIndex_label = tk.Label(morans_window, text="Morans index")
        moransPvalue_label = tk.Label(morans_window, text="P value")
        avgSpeed_label = tk.Label(morans_window, text="Average movement speed")
        sdSpeed_label = tk.Label(morans_window, text="STD movement speed")

        moransIndex = tk.Label(morans_window, text=str(round(analysis[0],3)))
        moransPvalue = tk.Label(morans_window, text=str(format(analysis[1],'.3g')))
        avgSpeed = tk.Label(morans_window, text=str(round(analysis[2],3)))
        sdSpeed = tk.Label(morans_window, text=str(round(analysis[3],3)))

        moransIndex_label.grid(row=0, column=0, padx=2, pady=5)
        moransPvalue_label.grid(row=1, column=0, padx=2, pady=5)
        avgSpeed_label.grid(row=2,column=0,padx=2,pady=5)
        sdSpeed_label.grid(row=3,column=0,padx=2,pady=5)


        moransIndex.grid(row=0, column=1, padx=2, pady=5)
        moransPvalue.grid(row=1, column=1, padx=2, pady=5)
        avgSpeed.grid(row=2,column=1,padx=2,pady=5)
        sdSpeed.grid(row=3,column=1,padx=2,pady=5)


        submit_button = tk.Button(morans_window, text='OK', command=submit)
        submit_button.grid()
        show_button = tk.Button(morans_window, text='Show autocorr heatmap', command=show_plot)
        show_button.grid()


    def dilation(self, frame, shape_x, shape_y):
        """
        Perform a dilation operation over some frame with some kernel (shape_x.shape_y)
        :param frame: An input frame
        :param shape_x: some integer for the kernel x value
        :param shape_y: Some integer for the kernel y value
        :return: An output frame with erosion applied
        """
        frame = cv2.dilate(frame, np.ones((shape_x, shape_y), np.uint8), iterations=1)
        return frame


    def erosion(self, frame, shape_x, shape_y):
        """
        Perform a erosion operation over some frame with some kernel (shape_x.shape_y)
        :param frame: An input frame
        :param shape_x: some integer for the kernel x value
        :param shape_y: Some integer for the kernel y value
        :return: An output frame with erosion applied
        """
        frame = cv2.erode(frame, np.ones((shape_x, shape_y), np.uint8), iterations=1)
        return frame


    def apply_mask(self, mask, original_frame):
        """
        This function applies a mask to a frame
        :param mask: Some black and white mask matching the frame size
        :param original_frame: Some frame
        :return: The frame with the applied mask
        """
        frame = cv2.bitwise_or(original_frame, np.ones_like(original_frame), mask=mask)
        return frame


    def object_detector_func(self, frame, object_detector):
        """
        This function applies an object detector to a frame and calculates a mask of moving objects
        :param frame: Some frame from a video
        :param object_detector: An earlier instantiated object detection object
        :return: A mask of the moving objects in a frame
        """
        frame = object_detector.apply(frame)
        return frame


    def corr_2d(self, a, v, Umean, Vmean):
        """
        Calculates the 2 dimensional correlation of matrices a and v.
        If a == v it calculates the autocorrelation
        :param a: Some matrix
        :param v: Some matrix with the same dimensions as a
        :return: a matrix of average correlation values
        """
        U_avg = np.mean(Umean)
        V_avg = np.mean(Vmean)
        UV_var = np.var(Umean+Vmean)
        v_ = np.array([U_avg,V_avg])
        temp = []
        for i in range(len(a) // 2):
            ai = a[i:]
            vi = v[:len(ai)]
            som = np.sum(((ai - v_)/UV_var) * ((vi - v_)), axis=1)
            temp.append(np.mean(som))
        return temp


    def mandatory_value_check(self, top_window,*inputs):

        def submit():
            mandatory_window.destroy()
        empty = False
        for value in inputs:
            if value == "":
                empty = True
        if empty:
            mandatory_window = Toplevel(top_window)

            x = top_window.winfo_x()
            y = top_window.winfo_y()
            mandatory_window.geometry("+%d+%d" % (x, y))
            label = tk.Label(mandatory_window, text="A mandatory input field is empty")
            submit_button = tk.Button(mandatory_window, text='Ok', command=submit)
            label.grid()
            submit_button.grid()

        return empty

if __name__ == '__main__':
    root = tk.Tk()
    root.columnconfigure(0,weight=1)
    app = PIV_gui(root)
    root.mainloop()
