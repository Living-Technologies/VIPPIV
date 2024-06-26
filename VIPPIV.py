import time
import tkinter as tk
from tkinter import Toplevel, filedialog
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

frame_index = 0
loaded_file = False
frames = []
actions = {}
actions_index = 0
object_detector_test = None
piv_canvas = None
toolbar = None
results = []
Vmean = None
Umean = None
quiver_x = None
quiver_y = None
first_run = True


def run_gui():
    """
    This is the main function to instantiate and run the entire gui
    :return:
    """
    window = tk.Tk()
    window.title('VIPPIV')
    global piv_canvas

    def get_monitor_from_coord(x, y):
        monitors = screeninfo.get_monitors()

        for m in reversed(monitors):
            if m.x <= x <= m.width + m.x and m.y <= y <= m.height + m.y:
                return m
        return monitors[0]

    # Get the screen which contains top
    current_screen = get_monitor_from_coord(window.winfo_x(), window.winfo_y())

    # Get the monitor's size
    width = int(current_screen.width / 4)

    def video_stream(frame):
        """
        This function takes a video frame and displays this on the left display field
        :param frame: A single frame from some video
        :return:
        """
        img = Image.fromarray(frame)
        max_width = width

        pixels_x, pixels_y = tuple([int(max_width / img.size[0] * x) for x in img.size])

        img = img.resize((pixels_x, pixels_y))
        imgtk = ImageTk.PhotoImage(image=img)
        display_left.imgtk = imgtk
        display_left.configure(image=imgtk)

    def video_output(frame):
        """
        This function takes a video frame and displays this on the right output field
        :param frame: A single frame from some video
        :return:
        """
        img = Image.fromarray(frame)
        max_width = width

        pixels_x, pixels_y = tuple([int(max_width / img.size[0] * x) for x in img.size])

        img = img.resize((pixels_x, pixels_y))
        imgtk = ImageTk.PhotoImage(image=img)
        display_right.imgtk = imgtk
        display_right.configure(image=imgtk)

    def read_video():
        """
        Ask the user to provide a path to a video.
        Open this video using cv2 and give the first frame to the display function.
        If this function is called while a PIV analysis has been performed in the runtime, reset the GUI to the start
        layout.
        :return:
        """
        global frames
        global frame_index
        global loaded_file
        global piv_canvas
        global Vmean
        global Umean
        global first_run
        Vmean = None
        Umean = None
        video_name = filedialog.askopenfilename()
        if video_name != '':

            frames = []
            if video_name.endswith('.mp4') or video_name.endswith('.avi'):
                cap = cv2.VideoCapture(video_name)
                succes = True
                j = 0
                while succes:
                    succes, image1 = cap.read()
                    if succes:
                        image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
                        frames.append(image1)
                        j += 1
            elif video_name.endswith('.czi'):
                czi_stack = CziFile(video_name)
                n_frames = czi_stack.get_dims_shape()[0]['T'][1]
                for i in range(n_frames):
                    frame = np.squeeze(czi_stack.read_image(T=i, C=0, S=0)[0]).astype(np.float32)
                    frame = ((frame - np.min(frame)) / (np.max(frame) - np.min(frame)) * 255).astype(np.uint8)
                    frames.append(frame)

            loaded_file = True
            if not first_run:
                display_right.lift()
                piv_canvas.get_tk_widget().destroy()
                toolbar.grid_remove()
                cb.grid_remove()
                button_moran_index.grid_remove()
                button_regenerate_plot.grid_remove()

            frame_index = 0
            video_stream(frames[frame_index])

    def left():
        """
        This function is called when the button with the left arrow is called and when possible displays the previous
        frame for some video
        :return:
        """
        global loaded_file
        global frame_index

        if loaded_file:
            if frame_index != 0:
                frame_index -= 1

                frame_label.config(text="Frame: " + str(frame_index))
                frame = frames[frame_index]
                video_stream(frame)
                output_frame = apply_preprocessing(frame, object_detector_test)

                video_output(output_frame)

    def right():
        """
        This function is called when the button with the left arrow is called and when possible displays the next frame
        for some video
        :return:
        """
        global loaded_file
        global frame_index
        global frames
        if loaded_file:

            if not frame_index + 1 >= len(frames):
                frame_index += 1

                frame_label.config(text="Frame: " + str(frame_index))

                frame = frames[frame_index]
                video_stream(frame)
                output_frame = apply_preprocessing(frame, object_detector_test)

                video_output(output_frame)

    def dilation_window():
        """
        When the user presses the dilation preprocessing button this function calles a subwindow asking for settings for
        the dilation operation. These settings are then stored and displayed in the center of the GUI
        :return:
        """
        if len(frames) != 0:
            global actions_index
            global actions

            def submit():
                global actions_index
                global actions
                if not mandatory_value_check(dilate_window,kernel_size_entry_x.get(),kernel_size_entry_y.get()):
                    input_kernel_x = kernel_size_entry_x.get()
                    input_kernel_y = kernel_size_entry_y.get()
                    actions["dilate " + str(actions_index)] = [input_kernel_x, input_kernel_y]
                    actions_index += 1
                    change_actions()
                    dilate_window.destroy()

            dilate_window = Toplevel(window)
            x = window.winfo_x()
            y = window.winfo_y()
            dilate_window.geometry("+%d+%d" % (x, y))
            dilate_window.transient(window)
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

    def erosion_window():
        """
        When the user presses the erosion preprocessing button this function calles a subwindow asking for settings for
        the erosion operation. These settings are then stored and displayed in the center of the GUI
        :return:
        """
        if len(frames) != 0:
            global actions_index
            global actions

            def submit():
                global actions_index
                global actions
                if not mandatory_value_check(erode_window, kernel_size_entry_x.get(), kernel_size_entry_y.get()):
                    input_kernel_x = kernel_size_entry_x.get()
                    input_kernel_y = kernel_size_entry_y.get()

                    actions["erosion " + str(actions_index)] = [input_kernel_x, input_kernel_y]
                    actions_index += 1
                    change_actions()
                    erode_window.destroy()

            erode_window = Toplevel(window)
            x = window.winfo_x()
            y = window.winfo_y()
            erode_window.geometry("+%d+%d" % (x, y))
            erode_window.transient(window)
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

    def apply_mask_window():
        """
        When the user presses the apply mask preprocessing button this function the preprocssing step is stored and
        displayed in the center window.
        :return:
        """
        if len(frames) != 0:
            global actions
            global actions_index

            actions["apply_mask " + str(actions_index)] = []
            actions_index += 1
            change_actions()

    def object_detector_window():
        """
        When the user presses the object detector preprocessing button this function calles a subwindow asking for settings for
        the object detection operation. These settings are then stored and displayed in the center of the GUI
        :return:
        """
        if len(frames) != 0:
            global actions
            global actions_index

            def submit():
                global actions_index
                global actions
                global object_detector_test

                threshold_value = threshold_value_entry.get()
                actions["object_detector " + str(actions_index)] = [threshold_value]
                actions_index += 1
                change_actions()
                object_detector_test = cv2.createBackgroundSubtractorMOG2(history=2, varThreshold=int(threshold_value),
                                                                          detectShadows=False)
                OD_window.destroy()

            OD_window = Toplevel(window)
            x = window.winfo_x()
            y = window.winfo_y()
            OD_window.geometry("+%d+%d" % (x, y))
            OD_window.transient(window)
            OD_window.grab_set()

            label_rows = tk.Label(OD_window, text="var_threshold")

            default = tk.StringVar(OD_window, value="100")
            threshold_value_entry = tk.Entry(OD_window, textvariable=default)

            threshold_value_entry.grid(row=0, column=1, padx=2, pady=5)
            label_rows.grid(row=0, column=0, padx=2, pady=5)
            submit_button = tk.Button(OD_window, text='Submit', command=submit)
            submit_button.grid()

    def change_actions():
        """
        This function is called whenever the user add a preprocessing step and stores and displays the steps with a
        hidden unique ID in the central field

        :return:
        """
        action_field.config(state="normal")

        action_field.delete("1.0", "end")
        settings_field.delete("1.0", "end")

        output_action = ""
        output_settings = ""
        index = 0
        for key, value in actions.items():
            if key.split(" ")[0] == "erosion" or key.split(" ")[0] == "dilate":
                output_action += key.split(" ")[0] + "\n"
                output_settings += "kernel size = " + value[0] + ',' + str(value[1]) + "\n"

            elif key.split(" ")[0] == "object_detector":
                output_action += key.split(" ")[0] + "\n"
                output_settings += "-" + "\n"

            else:
                output_action += key.split(" ")[0] + "\n"
                output_settings += "-" + "\n"
            index += 1

        action_field.insert("1.0", output_action)
        action_field.config(state="disabled")
        settings_field.insert("1.0", output_settings)

    def save_manual_changes():
        """
        If the user makes a manual change and clicks this button the function is called to fully store the made changes
        :return:
        """
        if len(frames) != 0:
            settings = settings_field.get("1.0", "end").split("\n")

            cleanup = lambda x: x.strip().replace(" ", "").split("=")
            settings = [cleanup(x)[1] if cleanup(x)[0] != '-' else cleanup(x)[0] for x in settings if
                        not cleanup(x) == ['']]
            for i, (key, value) in enumerate(actions.items()):
                if key.split(" ")[0] == "erosion" or key.split(" ")[0] == "dilate":
                    kernel = settings[i].split(",")
                    actions[key] = [kernel[0], kernel[1]]

                elif key.split(" ")[0] == "object_detector":
                    actions[key] = settings[i]

    def remove():
        """
        When this function is called a subwindow is called which allows the user to select the preprocessing steps to
        remove from the pipeline
        :return:
        """
        global actions
        if len(actions.keys()) != 0 and len(frames) != 0:
            remove_window = Toplevel(window)
            x = window.winfo_x()
            y = window.winfo_y()
            remove_window.geometry("+%d+%d" % (x, y))
            remove_window.transient(window)
            remove_window.grab_set()

            vars = []
            for i in range(len(actions.keys())):
                var = tk.IntVar()
                cb = tk.Checkbutton(remove_window, text=list(actions.keys())[i].split(" ")[0], variable=var)
                cb.grid(row=i, column=0, padx=10, sticky='w')
                vars.append(var)

            def submit():
                global actions
                # extract roll numbers for checked checkbuttons
                result = [var.get() for var in vars]
                # actions = [_[j] for j,_ in enumerate(list(actions.keys())) if not result[j]]

                for j, action in enumerate(list(actions.keys())):
                    if result[j]:
                        del (actions[action])
                change_actions()
                remove_window.destroy()

            submit_button = tk.Button(remove_window, text='Submit', command=submit)
            submit_button.grid()

    def apply_preprocessing(frame, object_detector):
        """
        This function takes a frame and returns this same frame with the correct preprocessing applied.
        :param frame: A single frame from a video
        :param object_detector: A cv2 objectdetector object used for preprocessing
        :return:
        """
        # frame = frames[frame_index]
        original_frame = copy.copy(frame)
        settings = settings_field.get("1.0", "end").split("\n")

        cleanup = lambda x: x.strip().replace(" ", "").split("=")
        settings = [cleanup(x)[1] if cleanup(x)[0] != '-' else cleanup(x)[0] for x in settings if
                    not cleanup(x) == ['']]

        counter = 0
        for key, value in actions.items():

            if key.split(" ")[0] == "erosion":
                kernel = settings[counter].split(",")
                frame = erosion(frame, int(kernel[0]), int(kernel[1]))

            elif key.split(" ")[0] == "dilate":
                kernel = settings[counter].split(",")
                frame = dilation(frame, int(kernel[0]), int(kernel[1]))

            elif key.split(" ")[0] == "apply_mask":
                frame = apply_mask(frame, original_frame)

            elif key.split(" ")[0] == "object_detector":
                frame = object_detector_func(frame, object_detector)
            counter += 1
        return frame

    def piv_thread():
        """
        This function display a subwindow with some settings for the PIV analysis and then calles the PIV function with
        these settings.
        :return:
        """
        global first_run
        first_run = False

        def submit():
            if not mandatory_value_check(piv_window, window_size_entry.get(), overlap_size_entry.get()):
                window_size = window_size_entry.get()
                overlap_size = overlap_size_entry.get()

                perform_piv(int(window_size), int(overlap_size))
                piv_window.destroy()

        piv_window = Toplevel(window)
        x = window.winfo_x()
        y = window.winfo_y()
        piv_window.geometry("+%d+%d" % (x, y))
        piv_window.transient(window)
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

    def perform_piv(win_size, overlap):
        """
        This function performs a full Particle Image Velocimetry (PIV) analysis on the video currently opened in the GUI.
        For each frame n and frame n+1 it calculates PIV with the end result being an average over these results.
        The first 2 frames are skipped as to give the object detectors 2 frames to instatiate.
        This function als calculates and display the plotted results.

        :param win_size: The size of the windows the video is subdivided into
        :param overlap: The percentage of overlap for the search windows 50 percent is commonly used
        :return:
        """
        global frame_index
        global piv_canvas
        global toolbar
        global results
        global Vmean
        global Umean
        global quiver_x
        global quiver_y
        results = []
        var_threshold = 100
        for key, value in actions.items():

            if key.split(" ")[0] == "object_detector":
                var_threshold = int(value[0])
                break

        object_detector_1 = cv2.createBackgroundSubtractorMOG2(history=2, varThreshold=var_threshold,
                                                               detectShadows=False)

        frame_index = 0

        progress.grid()
        time_left.grid()
        preprocessed_frames = []
        for i in range(len(frames)):
            preprocessed_frames.append(apply_preprocessing(frames[i],object_detector_1))

        if len(frames) != 0:
            total_len = len(frames)
            U = []
            V = []
            for i in range(total_len - 1):
                time_start = time.time()
                progress['value'] = 100 * ((i + 1) / total_len)

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
                    quiver_x, quiver_y = pyprocess.get_coordinates(image1.shape[:2], search_area_size=win_size,
                                                     overlap=overlap)

                    U.append(u)
                    V.append(v)
                time_end = time.time()
                timer = int((time_end - time_start) * (total_len - i))
                time_left.config(text="Estimated time left = " + str(timer) + " seconds")
                # window.update_idletasks()
                window.update()

            U = np.stack(U)
            U = np.nan_to_num(U)
            Umean = np.mean(U, axis=0)

            V = np.stack(V)
            V = np.nan_to_num(V)
            Vmean = np.mean(V, axis=0)

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
            quiver_y = np.flip(quiver_y, [0])

            pyth_matrix = np.sqrt(Umean ** 2 + Vmean ** 2)
            # Vmean[pyth_matrix < 1] = 0
            # Umean[pyth_matrix < 1] = 0
            max_num = np.amax(pyth_matrix)

            cmap = mcol.LinearSegmentedColormap.from_list("", ["blue", 'violet', "red"])

            Q = grid[0].quiver(quiver_x, quiver_y, Umean, Vmean*-1, pyth_matrix, scale_units='dots', scale=2, width=.007, clim=[0, 50],
                               cmap=cmap)

            grid[0].axis('off')

            cbar = plt.colorbar(Q, cax=grid.cbar_axes[0])

            cbar.set_ticks(ticks=[0, 50], labels=['0', '50'])

            piv_canvas = FigureCanvasTkAgg(fig, master=window)
            piv_canvas.draw()
            piv_canvas.get_tk_widget().grid(row=4, column=5, padx=10, pady=10)
            toolbar = NavigationToolbar2Tk(piv_canvas, window, pack_toolbar=False)
            toolbar.update()
            toolbar.grid(row=6, column=5, padx=2, pady=5)
            cb.select()
            plt.close()
            progress.grid_remove()
            time_left.grid_remove()
            cb.grid()
            button_save_matrix.grid()
            button_moran_index.grid()
            button_regenerate_plot.grid()

            speed_matrix_out = np.flip(pyth_matrix, 1)

            degrees = np.degrees(np.arctan2(Umean, Vmean))
            degrees[degrees < 0] = 180 + (180 - abs(degrees[degrees < 0]))
            degrees = np.flip(degrees, 1)

            rowby_x = []
            disp = Vmean.shape[1] // 2
            for yd in range(disp):
                UV_row = []
                for yindex in range(len(Vmean[1]) - yd):
                    a1 = np.array([[U, V] for U, V in zip(Umean[:, yindex], Vmean[:, yindex])])
                    a2 = np.array([[U, V] for U, V in zip(Umean[:, yindex + yd], Vmean[:, yindex + yd])])
                    UV_row.append(corr_2d(a1, a2, Umean, Vmean))
                UV_row = np.array(UV_row)
                rowby_x.append(np.mean(UV_row, 0))

            rowby_x = np.transpose(rowby_x)

            w = lat2W(rowby_x.shape[0], rowby_x.shape[1], rook=False)
            mi = Moran(rowby_x, w)

            results.append(speed_matrix_out)
            results.append(degrees)
            results.append(np.flip(Umean,1))
            results.append(np.flip(Vmean,1))

            analysis = [mi.I, mi.p_norm, np.mean(pyth_matrix), np.std(pyth_matrix)]

            results.append(analysis)
            results.append(rowby_x)


    def regenerate_quiver_plot(scale):
        global quiver_y
        global quiver_x
        global Vmean
        global Umean

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

        pyth_matrix = np.sqrt(Umean ** 2 + Vmean ** 2)

        Q = grid[0].quiver(quiver_x, quiver_y, Umean, Vmean * -1, pyth_matrix, scale_units='dots', scale=scale, width=.007,
                           clim=[0, 50],
                           cmap=cmap)
        grid[0].axis('off')

        cbar = plt.colorbar(Q, cax=grid.cbar_axes[0])

        cbar.set_ticks(ticks=[0, 50], labels=['0', '50'])

        piv_canvas = FigureCanvasTkAgg(fig, master=window)
        piv_canvas.draw()
        piv_canvas.get_tk_widget().grid(row=4, column=5, padx=10, pady=10)
        toolbar = NavigationToolbar2Tk(piv_canvas, window, pack_toolbar=False)
        toolbar.update()
        toolbar.grid(row=6, column=5, padx=2, pady=5)
        cb.select()
        plt.close()

    def regenerate_plot_window():
        """
        When the user presses the erosion preprocessing button this function calles a subwindow asking for settings for
        the erosion operation. These settings are then stored and displayed in the center of the GUI
        :return:
        """
        if len(frames) != 0:
            global actions_index
            global actions

            def submit():
                    input_scale = scale_entry.get()
                    plot_window.destroy()
                    regenerate_quiver_plot(float(input_scale))

            plot_window = Toplevel(window)
            x = window.winfo_x()
            y = window.winfo_y()
            plot_window.geometry("+%d+%d" % (x, y))
            plot_window.transient(window)
            plot_window.grab_set()

            label_rows = tk.Label(plot_window, text="quiver scale")

            default_scale = tk.StringVar(plot_window, value="2")
            scale_entry = tk.Entry(plot_window, textvariable=default_scale)

            scale_entry.grid(row=0, column=2, padx=2, pady=5)

            label_rows.grid(row=0, column=0, padx=2, pady=5)
            submit_button = tk.Button(plot_window, text='Submit', command=submit)
            submit_button.grid()
    def save_matrices():
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
        global results
        loc = filedialog.askdirectory()
        loc_directions = loc + "/arrow_directions.csv"
        loc_distance = loc + "/arrow_distance.csv"
        loc_meanX = loc + "/arrow_meanX.csv"
        loc_meanY = loc + "/arrow_meanY.csv"
        loc_summary = loc + "/summary.csv"

        with open(loc_distance, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(results[0])
        with open(loc_directions, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(results[1])
        with open(loc_meanX, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(results[2])
        with open(loc_meanY, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(results[3])
        with open(loc_summary, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["The Morans index value = ", results[4][0]])
            writer.writerow(["The Morans index associated P value = ", results[4][1]])
            writer.writerow(["The average movement speed = ", results[4][2]])
            writer.writerow(["The standard deviation of the movement speed = ", results[4][3]])

    def piv_display():
        """
        This function toggles between the PIV and the preprocessing output view by using a checkbox

        :return:
        """

        global piv_canvas
        global toolbar
        if piv_canvas != None and piv_check.get():

            display_right.lower()
            toolbar.grid()
            button_save_matrix.grid()
            button_moran_index.grid()
            button_regenerate_plot.grid()

        elif piv_canvas != None and not piv_check.get():
            display_right.lift()
            toolbar.grid_remove()
            button_save_matrix.grid_remove()
            button_moran_index.grid_remove()
            button_regenerate_plot.grid_remove()

    def Morans_I():
        """
        This function displays the Morans index, Morans index p value, average speed and standard deviation of the speed.
        It also allows the user to see the underlying heatmap on which the Morans index is calculated.
        :return:
        """

        global Vmean
        global Umean
        global results

        def submit():
            morans_window.destroy()

        def show_plot():
            plt.imshow(results[5], origin='lower')
            plt.show()

        analysis = results[4]

        morans_window = Toplevel(window)
        morans_window.transient(window)
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

    progress = Progressbar(window, orient="horizontal",
                           length=100, mode='determinate')
    time_left = tk.Label(window, text="Estimated time left= ")

    action_field = tk.Text(width=20, state="disabled")
    settings_field = tk.Text(width=25)
    display_left = tk.Label(window)
    title = tk.Label(window, text="Visual Interactive Program for Particle Image Velocimetry")

    # square = Image.fromarray(np.zeros((480, int(width/7))))
    square = Image.fromarray(np.zeros((480, width)))

    imgtk_sq = ImageTk.PhotoImage(image=square)
    display_left.imgtk = imgtk_sq
    display_left.configure(image=imgtk_sq)

    display_right = tk.Label(window)

    # square = Image.fromarray(np.zeros((480, int(width/7))))
    imgtk_sq = ImageTk.PhotoImage(image=square)
    display_right.imgtk = imgtk_sq
    display_right.configure(image=imgtk_sq)

    button_open = tk.Button(text='Open video file', width=20, command=read_video)
    button_dilate = tk.Button(text='add_dilation', width=20, command=dilation_window)
    button_erode = tk.Button(text='add_erosion', width=20, command=erosion_window)
    button_apply_mask = tk.Button(text='apply_mask', width=20, command=apply_mask_window)
    button_object_detector = tk.Button(text='Add object detector', width=20, command=object_detector_window)
    button_remove_latest_actions = tk.Button(text='remove action', width=20, command=remove)
    button_perform_piv = tk.Button(text='PIV analysis', width=20, command=piv_thread)
    button_save_manual = tk.Button(text='Save manual changes', width=20, command=save_manual_changes)
    button_save_matrix = tk.Button(text='Save arrow information', width=20, command=save_matrices)
    button_moran_index = tk.Button(text='Calculate Morans Index', width=20, command=Morans_I)
    button_regenerate_plot = tk.Button(text='regenerate_plot',width=20,command=regenerate_plot_window)
    piv_check = tk.IntVar()

    cb = tk.Checkbutton(window, text="Show PIV", variable=piv_check, command=piv_display)

    frame_label = tk.Label(window)

    button_left = tk.Button(text='<--', width=8, command=left)
    button_right = tk.Button(text='-->', width=8, command=right)

    title.grid(row=0, column=0)
    action_field.grid(row=4, column=2, padx=5, pady=5)
    settings_field.grid(row=4, column=3, padx=5, pady=5)

    button_dilate.grid(row=1, column=1, padx=2, pady=5)
    button_erode.grid(row=1, column=2, padx=2, pady=5)
    button_apply_mask.grid(row=1, column=3, padx=2, pady=5)
    button_object_detector.grid(row=2, column=1, padx=2, pady=5)
    button_remove_latest_actions.grid(row=2, column=3, padx=2, pady=5)
    button_perform_piv.grid(row=2, column=2, padx=2, pady=5)
    button_save_manual.grid(row=3, column=3, padx=2, pady=5)
    button_save_matrix.grid(row=5, column=5, padx=2, pady=5)
    button_moran_index.grid(row=3, column=5, padx=2, pady=5)
    button_moran_index.grid_remove()
    button_save_matrix.grid_remove()
    button_regenerate_plot.grid(row=1,column=5, padx=2, pady=5)
    button_regenerate_plot.grid_remove()
    cb.grid(row=2, column=5, padx=2, pady=5)
    cb.grid_remove()

    progress.grid(row=6, column=2, padx=2, pady=5)
    progress.grid_remove()
    time_left.grid(row=7, column=2, padx=2, pady=5)
    time_left.grid_remove()

    button_open.grid(row=0, column=2, padx=2, pady=5)
    button_left.grid(row=4, column=1, padx=2, pady=5)
    button_right.grid(row=4, column=4, padx=2, pady=5)

    display_left.grid(row=4, column=0, padx=2, pady=5)
    display_right.grid(row=4, column=5, padx=2, pady=5)

    frame_label.grid(row=5, column=2)

    window.mainloop()


def dilation(frame, shape_x, shape_y):
    """
    Perform a dilation operation over some frame with some kernel (shape_x.shape_y)
    :param frame: An input frame
    :param shape_x: some integer for the kernel x value
    :param shape_y: Some integer for the kernel y value
    :return: An output frame with erosion applied
    """
    frame = cv2.dilate(frame, np.ones((shape_x, shape_y), np.uint8), iterations=1)
    return frame


def erosion(frame, shape_x, shape_y):
    """
    Perform a erosion operation over some frame with some kernel (shape_x.shape_y)
    :param frame: An input frame
    :param shape_x: some integer for the kernel x value
    :param shape_y: Some integer for the kernel y value
    :return: An output frame with erosion applied
    """
    frame = cv2.erode(frame, np.ones((shape_x, shape_y), np.uint8), iterations=1)
    return frame


def apply_mask(mask, original_frame):
    """
    This function applies a mask to a frame
    :param mask: Some black and white mask matching the frame size
    :param original_frame: Some frame
    :return: The frame with the applied mask
    """
    frame = cv2.bitwise_or(original_frame, np.ones_like(original_frame), mask=mask)
    return frame


def object_detector_func(frame, object_detector):
    """
    This function applies an object detector to a frame and calculates a mask of moving objects
    :param frame: Some frame from a video
    :param object_detector: An earlier instantiated object detection object
    :return: A mask of the moving objects in a frame
    """
    frame = object_detector.apply(frame)
    return frame


def corr_2d(a, v, Umean, Vmean):
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


def mandatory_value_check(top_window,*inputs):

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
    run_gui()
