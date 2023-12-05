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
from  matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import copy
from mpl_toolkits.axes_grid1 import ImageGrid
import ctypes
import screeninfo
import matplotlib.colors as mcol
import matplotlib.cm as cm
from libpysal.weights import lat2W
from esda.moran import Moran

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
first_run = True

def run_gui():
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
    width = int(current_screen.width/4)
    def video_stream(frame):

        img = Image.fromarray(frame)
        max_width = width

        pixels_x, pixels_y = tuple([int(max_width / img.size[0] * x) for x in img.size])

        img = img.resize((pixels_x, pixels_y))
        imgtk = ImageTk.PhotoImage(image=img)
        display_left.imgtk = imgtk
        display_left.configure(image=imgtk)

    def video_output(frame):

        img = Image.fromarray(frame)
        max_width = width

        pixels_x, pixels_y = tuple([int(max_width / img.size[0] * x) for x in img.size])

        img = img.resize((pixels_x, pixels_y))
        imgtk = ImageTk.PhotoImage(image=img)
        display_right.imgtk = imgtk
        display_right.configure(image=imgtk)

    def read_video():
        #video_name = "videos/Videos for Bram/20230626_BARI_exp30_HNEC0153_day 18_elke tile 30sec.czi #01.avi"  # This is your video file path
        global frames
        global frame_index
        global loaded_file
        global piv_canvas
        global Vmean
        global Umean
        global first_run
        Vmean = None
        Umean = None

        frames = []
        video_name = filedialog.askopenfilename()
        cap = cv2.VideoCapture(video_name)
        succes = True
        j = 0
        while succes:
            succes, image1 = cap.read()
            if succes:
                image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
                frames.append(image1)
                j += 1

        loaded_file = True
        if not first_run:
            display_right.lift()
            piv_canvas.get_tk_widget().destroy()
            toolbar.grid_remove()
            cb.grid_remove()
            button_moran_index.grid_remove()

        frame_index = 0
        video_stream(frames[frame_index])

    def left():
        global loaded_file
        global frame_index

        if loaded_file:
            if frame_index != 0:
                frame_index -= 1

                frame_label.config(text="Frame: " + str(frame_index))
                frame = frames[frame_index]
                video_stream(frame)
                output_frame = apply_preprocessing(frame,object_detector_test)

                video_output(output_frame)

    def right():
        global loaded_file
        global frame_index
        global frames
        if loaded_file:

            if not frame_index+1 >= len(frames):
                frame_index += 1

                frame_label.config(text="Frame: " + str(frame_index))

                frame = frames[frame_index]
                video_stream(frame)
                output_frame = apply_preprocessing(frame,object_detector_test)

                video_output(output_frame)

    def dilation_window():
        if len(frames) != 0:
            global actions_index
            global actions

            def submit():
                global actions_index
                global actions
                input_kernel_x = kernel_size_entry_x.get()
                input_kernel_y = kernel_size_entry_y.get()
                actions["dilate " + str(actions_index)] = [input_kernel_x, input_kernel_y]
                actions_index += 1
                change_actions()
                dilate_window.destroy()

            dilate_window = Toplevel(window)
            dilate_window.transient(window)
            dilate_window.grab_set()

            label_rows = tk.Label(dilate_window, text="kernel_size")

            default_x = tk.StringVar(dilate_window, value="15")
            default_y = tk.StringVar(dilate_window, value="15")
            kernel_size_entry_x = tk.Entry(dilate_window, textvariable=default_x)
            kernel_size_entry_y = tk.Entry(dilate_window, textvariable=default_y)

            kernel_size_entry_x.grid(row=0, column=1, padx=2, pady=5)
            kernel_size_entry_y.grid(row=0, column=2, padx=2, pady=5)
            label_rows.grid(row=0, column=0, padx=2, pady=5)
            submit_button = tk.Button(dilate_window, text='Submit', command=submit)
            submit_button.grid()

    def erosion_window():
        if len(frames) != 0:
            global actions_index
            global actions

            def submit():
                global actions_index
                global actions
                input_kernel_x = kernel_size_entry_x.get()
                input_kernel_y = kernel_size_entry_y.get()

                actions["erosion " + str(actions_index)] = [input_kernel_x, input_kernel_y]
                actions_index += 1
                change_actions()
                erode_window.destroy()

            erode_window = Toplevel(window)
            erode_window.transient(window)
            erode_window.grab_set()

            label_rows = tk.Label(erode_window, text="kernel_size")

            default_x = tk.StringVar(erode_window, value="15")
            default_y = tk.StringVar(erode_window, value="15")
            kernel_size_entry_x = tk.Entry(erode_window, textvariable=default_x)
            kernel_size_entry_y = tk.Entry(erode_window, textvariable=default_y)


            kernel_size_entry_x.grid(row=0, column=1, padx=2, pady=5)
            kernel_size_entry_y.grid(row=0, column=2, padx=2, pady=5)

            label_rows.grid(row=0, column=0, padx=2, pady=5)
            submit_button = tk.Button(erode_window, text='Submit', command=submit)
            submit_button.grid()

    def apply_mask_window():
        if len(frames) != 0:
            global actions
            global actions_index

            actions["apply_mask " + str(actions_index)] = []
            actions_index += 1
            change_actions()

    def object_detector_window():

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
                object_detector_test = cv2.createBackgroundSubtractorMOG2(history=2, varThreshold=int(threshold_value),detectShadows=False)
                OD_window.destroy()

            OD_window = Toplevel(window)
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
        action_field.config(state="normal")

        action_field.delete("1.0", "end")
        settings_field.delete("1.0", "end")

        output_action = ""
        output_settings = ""
        index=0
        for key, value in actions.items():
            if key.split(" ")[0] == "erosion" or key.split(" ")[0] == "dilate":
                output_action += key.split(" ")[0] + "\n"
                output_settings += "kernel size = " + value[0] +',' +str(value[1])+ "\n"

            elif key.split(" ")[0] == "object_detector":
                output_action += key.split(" ")[0] + "\n"
                output_settings += "-" + "\n"

            else:
                output_action += key.split(" ")[0] + "\n"
                output_settings += "-" + "\n"
            index+=1

        action_field.insert("1.0", output_action)
        action_field.config(state="disabled")
        settings_field.insert("1.0",output_settings)

    def save_manual_changes():
        if len(frames) != 0:
            settings = settings_field.get("1.0","end").split("\n")

            cleanup = lambda x : x.strip().replace(" ","").split("=")
            settings = [cleanup(x)[1] if cleanup(x)[0] != '-' else cleanup(x)[0] for x in settings if not cleanup(x) == ['']]
            for i, (key,value) in enumerate(actions.items()):
                if key.split(" ")[0] == "erosion" or key.split(" ")[0] == "dilate":
                    kernel = settings[i].split(",")
                    actions[key] = [kernel[0], kernel[1]]

                elif key.split(" ")[0] == "object_detector":
                    actions[key] = settings[i]

    def remove():
        global actions
        if len(actions.keys()) != 0 and len(frames) != 0:
            remove_window = Toplevel(window)
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

    def apply_preprocessing(frame,object_detector):
        #frame = frames[frame_index]
        original_frame = copy.copy(frame)
        settings = settings_field.get("1.0","end").split("\n")

        cleanup = lambda x : x.strip().replace(" ","").split("=")
        settings = [cleanup(x)[1] if cleanup(x)[0] != '-' else cleanup(x)[0] for x in settings if not cleanup(x) == ['']]

        counter = 0
        for key, value in actions.items():

            if key.split(" ")[0] == "erosion":
                kernel = settings[counter].split(",")
                frame = erosion(frame, int(kernel[0]),int(kernel[1]))

            elif key.split(" ")[0] == "dilate":
                kernel = settings[counter].split(",")
                frame = dilation(frame, int(kernel[0]),int(kernel[1]))

            elif key.split(" ")[0] == "apply_mask":
                frame = apply_mask(frame, original_frame)

            elif key.split(" ")[0] == "object_detector":
                frame = object_detector_func(frame,object_detector)
            counter += 1
        return frame

    def piv_thread():
        global first_run
        first_run = False
        def submit():
            window_size = window_size_entry.get()
            overlap_size = overlap_size_entry.get()

            perform_piv(int(window_size),int(overlap_size))
            piv_window.destroy()

        piv_window = Toplevel(window)
        piv_window.transient(window)
        piv_window.grab_set()

        label_window = tk.Label(piv_window, text="window_size")
        label_overlap = tk.Label(piv_window, text="overlap_size")

        default_win_size = tk.StringVar(piv_window, value="150")
        default_overlap_size = tk.StringVar(piv_window, value="75")

        window_size_entry = tk.Entry(piv_window, textvariable=default_win_size)
        overlap_size_entry = tk.Entry(piv_window, textvariable=default_overlap_size)

        window_size_entry.grid(row=0, column=1, padx=2, pady=5)
        overlap_size_entry.grid(row=1, column=1, padx=2, pady=5)

        label_window.grid(row=0, column=0, padx=2, pady=5)
        label_overlap.grid(row=1, column=0, padx=2, pady=5)

        submit_button = tk.Button(piv_window, text='Submit', command=submit)
        submit_button.grid()


    def perform_piv(win_size, overlap):
        global frame_index
        global piv_canvas
        global toolbar
        global results
        global Vmean
        global Umean
        results = []
        var_threshold = 100
        for key, value in actions.items():

            if key.split(" ")[0] == "object_detector":
                var_threshold = int(value[0])
                break


        object_detector_1 = cv2.createBackgroundSubtractorMOG2(history=2, varThreshold=var_threshold,detectShadows=False)
        object_detector_2 = cv2.createBackgroundSubtractorMOG2(history=2, varThreshold=var_threshold,detectShadows=False)

        frame_index = 0

        progress.grid()
        time_left.grid()

        if len(frames) != 0:
            total_len = len(frames)
            U = []
            V = []
            for i in range(total_len-1):
                time_start = time.time()
                progress['value'] = 100 * ((i+1) / total_len)
                if i != 0 or i != 1:
                    image1 = frames[i]
                    image2 = frames[i+1]

                    image1 = apply_preprocessing(image1,object_detector_1)
                    image2 = apply_preprocessing(image2,object_detector_2)

                    image1 = cv2.cvtColor(image1, cv2.COLOR_GRAY2RGB)
                    image2 = cv2.cvtColor(image2, cv2.COLOR_GRAY2RGB)
                    u, v, s2n = pyprocess.extended_search_area_piv(image1.sum(axis=2),
                                                                   image2.sum(axis=2), window_size=win_size,
                                                                   overlap=overlap, dt=1);
                    x, y = pyprocess.get_coordinates(image1.shape[:2], search_area_size=win_size,
                                                     overlap=overlap)

                    U.append(u)
                    V.append(v)
                time_end = time.time()
                timer = int((time_end-time_start) * (total_len- i))
                time_left.config(text="Estimated time left = "+str(timer) + " seconds")
                #window.update_idletasks()
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

            Umean = np.flip(Umean, [1])
            Vmean = np.flip(Vmean,[0])

            x = np.flip(x,[1])
            y = np.flip(y,[0])

            pyth_matrix = np.sqrt(Umean ** 2 + Vmean ** 2)
            max_num = np.amax(pyth_matrix)

            cmap = mcol.LinearSegmentedColormap.from_list("", ["blue", 'violet', "red"])

            Q = grid[0].quiver(x, y, Umean, Vmean, pyth_matrix,scale_units = 'dots',scale=2, width=.007,clim=[0,50],cmap=cmap)

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

            pyth_matrix_out = np.flip(pyth_matrix,1)
            results.append(pyth_matrix_out)
            degrees = np.degrees(np.arctan2(Umean,Vmean))
            degrees[degrees < 0] = 180 + (180 - abs(degrees[degrees < 0]))
            degrees = np.flip(degrees, 1)
            results.append(degrees)
            results.append(Umean)
            results.append(Vmean)


    def save_matrices():
        global results
        loc = filedialog.askdirectory()
        loc_directions = loc + "/arrow_directions.csv"
        loc_distance = loc + "/arrow_distance.csv"
        loc_meanX = loc + "/arrow_meanX.csv"
        loc_meanY = loc + "/arrow_meanY.csv"


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


    def piv_display():
        global piv_canvas
        global toolbar
        if piv_canvas != None and piv_check.get():

            display_right.lower()
            toolbar.grid()
            button_save_matrix.grid()
            button_moran_index.grid()

        elif piv_canvas != None and not piv_check.get():
            display_right.lift()
            toolbar.grid_remove()
            button_save_matrix.grid_remove()
            button_moran_index.grid_remove()


    def Morans_I():
        global Vmean
        global Umean

        def submit():
            morans_window.destroy()

        def show_plot():
            plt.imshow(rowby_x, origin='lower')
            plt.show()

        rowby_x = []
        disp = Vmean.shape[1] // 2
        for yd in range(disp):
            UV_row = []
            for yindex in range(len(Vmean[1]) - yd):
                a1 = np.array([[U, V] for U, V in zip(Umean[:, yindex], Vmean[:, yindex])])
                a2 = np.array([[U, V] for U, V in zip(Umean[:, yindex + yd], Vmean[:, yindex + yd])])
                UV_row.append(corr_2d(a1, a2))
            UV_row = np.array(UV_row)
            rowby_x.append(np.mean(UV_row, 0))
            
        rowby_x = np.transpose(rowby_x)

        w = lat2W(rowby_x.shape[0], rowby_x.shape[1], rook=False)
        mi = Moran(rowby_x, w)

        morans_window = Toplevel(window)
        morans_window.transient(window)
        morans_window.grab_set()
        moransIndex_label = tk.Label(morans_window, text="Morans index")
        moransPvalue_label = tk.Label(morans_window,text="P value")
        moransIndex = tk.Label(morans_window, text=str(mi.I))
        moransPvalue = tk.Label(morans_window,text=str(mi.p_norm))
        moransIndex.grid(row=0, column=1, padx=2, pady=5)
        moransPvalue.grid(row=1, column=1, padx=2, pady=5)
        moransIndex_label.grid(row=0, column=0, padx=2, pady=5)
        moransPvalue_label.grid(row=1, column=0, padx=2, pady=5)

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

    #square = Image.fromarray(np.zeros((480, int(width/7))))
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
    button_save_manual = tk.Button(text='Save manual changes', width=20,command=save_manual_changes)
    button_save_matrix = tk.Button(text='Save arrow information', width=20,command=save_matrices)
    button_moran_index = tk.Button(text='Calculate Morans Index', width=20,command=Morans_I)
    piv_check = tk.IntVar()

    cb = tk.Checkbutton(window, text="Show PIV",variable=piv_check, command=piv_display)

    frame_label = tk.Label(window)

    button_left = tk.Button(text='<--', width=8, command=left)
    button_right = tk.Button(text='-->', width=8, command=right)


    title.grid(row=0,column=0)
    action_field.grid(row=4, column=2, padx=5, pady=5)
    settings_field.grid(row=4,column=3,padx=5,pady=5)

    button_dilate.grid(row=1, column=1, padx=2, pady=5)
    button_erode.grid(row=1, column=2, padx=2, pady=5)
    button_apply_mask.grid(row=1, column=3, padx=2, pady=5)
    button_object_detector.grid(row=2, column=1, padx=2, pady=5)
    button_remove_latest_actions.grid(row=2, column=3, padx=2, pady=5)
    button_perform_piv.grid(row=2, column=2, padx=2, pady=5)
    button_save_manual.grid(row=3,column=3,padx=2,pady=5)
    button_save_matrix.grid(row=5,column=5,padx=2,pady=5)
    button_moran_index.grid(row=3,column=5,padx=2,pady=5)
    button_moran_index.grid_remove()
    button_save_matrix.grid_remove()
    cb.grid(row=2,column=5,padx=2,pady=5)
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


def dilation(frame,shape_x,shape_y):

    frame = cv2.dilate(frame, np.ones((shape_x, shape_y), np.uint8), iterations=1)
    return frame


def erosion(frame,shape_x,shape_y):
    frame = cv2.erode(frame, np.ones((shape_x, shape_y), np.uint8), iterations=1)
    return frame


def apply_mask(mask, original_frame):
    frame = cv2.bitwise_or(original_frame,np.ones_like(original_frame),mask=mask)
    return frame


def object_detector_func(frame,object_detector):
    frame = object_detector.apply(frame)
    return frame


def corr_2d(a, v):
    temp = []
    for i in range(len(a) // 2):
        ai = a[i:]
        vi = v[:len(ai)]
        som = np.sum((ai - np.mean(ai, 0)) * (vi - np.mean(vi, 0)), axis=1)
        temp.append(np.mean(som))

    return temp

if __name__ == '__main__':
    run_gui()
