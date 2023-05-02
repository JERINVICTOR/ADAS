import tkinter as tk
import subprocess

import psutil
import time


drows_running= False
process1 = None

lane_runnnig=False
process2=None

collision_running=False
process3=None

roadSign_running=False
process4=None

ANN_lane_running = False
process5 = None



def toggle_button_2_3():
    toggle_button_2()
    toggle_button_3()

def toggle_button_1():
    global drows_running, process1

    if button1.config('text')[-1] == 'OFF':
        button1.config(text='ON')
        if not drows_running:
            conda_env = "drowsy"
            script_path = "drows.py"
            process1 = subprocess.Popen(["conda", "run", "-n", conda_env, "python", script_path])
            print("drowsy running")
            
                      
            drows_running= True
    else:
        button1.config(text='OFF')
        if drows_running:
            for processDrowsy in psutil.process_iter(['pid', 'name']):
                if processDrowsy.info['name'] == 'drowsy_python':
                    processDrowsy.terminate()           
            process1.terminate()
            drows_running= False

def toggle_button_2():
    global lane_runnnig, process2

    if button2.config('text')[-1] == 'OFF':
        button2.config(text='ON')
        if not lane_runnnig:
            conda_env = "edgemodel"
            script_path = "laneDetection.py"
            process2 = subprocess.Popen(["conda", "run", "-n", conda_env, "python", script_path])
            print("LaneDetection running")
            lane_runnnig= True
            time.sleep(5)
    else:
        button2.config(text='OFF')
        if lane_runnnig:
            for processLane in psutil.process_iter(['pid', 'name']):
                if processLane.info['name'] == 'LaneDetection_python':
                    processLane.terminate()
            process2.terminate()
            lane_runnnig= False

def toggle_button_3():
    global collision_running, process3

    if button3.config('text')[-1] == 'OFF':
        button3.config(text='ON')
        if not collision_running:
            conda_env = "distance"
            script_path = "distance/DistanceEstimation.py"
            process3 = subprocess.Popen(["conda", "run", "-n", conda_env, "python", script_path])
            print("Collision Alert running")

            collision_running= True
    else:
        button3.config(text='OFF')
        if collision_running:
            for processDist in psutil.process_iter(['pid', 'name']):
                if processDist.info['name'] == 'Collision_alert_python':
                    processDist.terminate()
            process3.terminate()
            collision_running= False

def toggle_button_4():
    global roadSign_running, process4

    if button4.config('text')[-1] == 'OFF':
        button4.config(text='ON')
        if not roadSign_running:
            #yolo_script = "conda run -n yoloSign yolo task=detect mode=predict model='yoloSign.pt' conf=0.80 source=2 show=True"
            #process4 = subprocess.Popen(yolo_script, shell = True)
            conda_env = "sign10"
            script_path = "TrafficSign/main.py"
            process4 = subprocess.Popen(["conda", "run", "-n", conda_env, "python", script_path])
            print("sign detection  running")            
            #process4 = subprocess.Popen(['python', ''])
            roadSign_running= True
    else:
        button4.config(text='OFF')
        if roadSign_running:
            for processYolo in psutil.process_iter(['pid', 'name']):
                if processYolo.info['name'] == 'SignDetection_python':
                    processYolo.terminate()
            process4.terminate()
            roadSign_running= False

def toggle_button_5():
    global ANN_lane_running, process5

    if button5.config('text')[-1] == 'OFF':
        button5.config(text='ON')
        if not ANN_lane_running:
            conda_env = "lane_environment"
            script_path = "nnlanedetection/draw_detected_lanes.py"
            process5 = subprocess.Popen(["conda", "run", "-n", conda_env, "python", script_path])
            print("ANN Lane detection running")            
            #process5 = subprocess.Popen(['python', ''])
            ANN_lane_running= True
    else:
        button5.config(text='OFF')
        if ANN_lane_running:
            for processANN in psutil.process_iter(['pid', 'name']):
                if processANN.info['name'] == 'ANN_lane_python':
                    processANN.terminate()
            process5.terminate()
            ANN_lane_running= False




root = tk.Tk()
root.title('Toggle Buttons')
root.geometry('1920x1080')


label1 = tk.Label(root, text="Drowsiness Detection")
label1.pack(side='left', padx=10, pady=5)

button1 = tk.Button(root, text='OFF', command=toggle_button_1, width=10)
button1.pack(side='left', padx=10, pady=10)

#label2 = tk.Label(root, text="Lane detection")
#label2.pack(side='left', padx=20, pady=5)

button2 = tk.Button(root, text='OFF', command=toggle_button_2, width=10)
#button2.pack(side='left', padx=20, pady=10)

label3 = tk.Label(root, text="Lane & Collision Warning")
label3.pack(side='left', padx=30, pady=5)

button3 = tk.Button(root, text='OFF', command=toggle_button_2_3, width=10)
button3.pack(side='left', padx=30, pady=10)

label4 = tk.Label(root, text="Road Sign detection")
label4.pack(side='left', padx=40, pady=5)

button4 = tk.Button(root, text='OFF', command=toggle_button_4, width=10)
button4.pack(side='left', padx=40, pady=10)

label5 = tk.Label(root, text="ANN Lane Detection")
label5.pack(side='left', padx=40, pady=5)

button5 = tk.Button(root, text='OFF', command=toggle_button_5, width=10)
button5.pack(side='left', padx=40, pady=10)

root.mainloop()
