from multiprocessing.synchronize import Event
from multiprocessing import  Process, Queue
from DataFeelCenter import DataFeelCenter, token
import queue as pyqueue
import time

def Worker(stop_evt:Event, q_cmd: Queue):
    dfc = DataFeelCenter(numOfDots=4)  
    
    while not stop_evt.is_set():
        # try:
            while not q_cmd.empty():
                # print current time stamp ms
                # print(f"Current time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}.{int(time.time() * 1000) % 1000:03d}")
                cmd = q_cmd.get_nowait()
                if not cmd:
                    return
                method, args = cmd

                getattr(dfc, method)(*args)

        # except Exception as e:
        #     print(f"worker Err! {e}")
        #     stop_evt.set()
        #     break

def Commander(stop_evt: Event, q_pres:Queue, q_vib:Queue, q_therm:Queue, q_cmd:Queue):
    start = False
    startCool = False
    last_trigger = time.perf_counter()
    lastCool_trigger = time.perf_counter()
    time_priority = [0,0,0,0]
    total_priority = [0,0,0,0]
    last_state = []
    TIMEPRIOIR_SCALE = 2
    while not stop_evt.is_set():
        print(q_pres.qsize(), q_vib.qsize(), q_therm.qsize())
        t = q_pres.get()
        temp, pres1, pres2 = t.split(";")
        p = pres1.split(",")
        p1 = pres2.split(",")
        vib = q_vib.get()
        therm = q_therm.get()

        # ------Thermal Feedback------------------
        tone_smooth = therm

        TONE_THRESHOLD = 0.24
        DURATION_THRESHOLD = 0.15

        thermVal = None
        thermDiff = None


        if tone_smooth >= TONE_THRESHOLD:
            # print("PreHeating...")
            if start == False:
                # print("Heating Count")
                last_trigger = time.perf_counter()
                start = True
            elif time.perf_counter() - last_trigger >= DURATION_THRESHOLD:
                print("Heating...")
                thermDiff = 4
        elif tone_smooth < TONE_THRESHOLD:
            # print("PreCooling...")
            if startCool == False:
                lastCool_trigger = time.perf_counter()
                startCool = True
            elif time.perf_counter() - lastCool_trigger >= DURATION_THRESHOLD*10:
                start = False
                startCool = False
                print("Cooling...")
                thermDiff = 0


        # print(q_pres.empty(), q_vib.empty(), q_therm.empty())
        # print(q_pres.qsize(), q_vib.qsize(), q_therm.qsize())

        # print(f"pres {t1 - t0:.3f}s, vib {t2 - t1:.3f}s, therm {t3 - t2:.3f}s")

        led = [[0,0,0]]*8

        for i, val in enumerate(p):
            if int(val)>60:
                # led[i] = [random.randint(0,255), random.randint(0,255), random.randint(0,255)]
                # led[i] = [int(val), int(val), int(val)]
                # map int(val) from 0-1023 to 0-255
                # led[i] = [int(val) // 4] * 3
                led[i] = [255, 0, 0]

        led1 = [[0,0,0]]*8

        for i, val in enumerate(p1):
            if int(val)>60:
                # led[i] = [random.randint(0,255), random.randint(0,255), random.randint(0,255)]
                # led[i] = [int(val), int(val), int(val)]
                # map int(val) from 0-1023 to 0-255
                # led[i] = [int(val) // 4] * 3
                led1[i] = [255, 0, 0]


        # print(vib, therm)
        t0 = token(superDotID = 0, vibFrequency=70, vibIntensity=0, therIntensity=None, therDiff=0, ledList=led)
        t1 = token(superDotID = 1, vibFrequency=70, vibIntensity=0, therIntensity=None, therDiff=0, ledList=led1)
        t2 = token(superDotID = 2, vibFrequency=70, vibIntensity=vib, therIntensity=thermVal, therDiff=thermDiff, ledList=None)
        t3 = token(superDotID = 3, vibFrequency=70, vibIntensity=vib, therIntensity=thermVal, therDiff=thermDiff, ledList=None)

        # print(t0)
        try:
            # print(f"qcmd: {q_cmd.qsize()}")
            if len(last_state) != 0 :
                '''t0'''
                if t0.therDiff != last_state[0].therDiff: total_priority[0] += 100
                if t0.vibIntensity != None and last_state[0].vibIntensity != None: 
                    total_priority[0] += 5*abs(t0.vibIntensity - last_state[0].vibIntensity) 
                if t0.ledList is not None:
                    total_priority[0] += (int)((sum(abs(x - y) for row1, row2 in zip(t0.ledList, last_state[0].ledList) for x, y in zip(row1, row2)))/255)
                '''t1'''
                if t1.therDiff != last_state[1].therDiff: total_priority[1] += 100
                if t1.vibIntensity != None and last_state[1].vibIntensity != None: 
                    total_priority[1] += 5*abs(t1.vibIntensity - last_state[1].vibIntensity) 
                if t1.ledList is not None:
                    total_priority[1] += (int)((sum(abs(x - y) for row1, row2 in zip(t1.ledList, last_state[1].ledList) for x, y in zip(row1, row2)))/255)
                '''t2'''
                if t2.therDiff != last_state[2].therDiff: total_priority[2] += 100
                if t2.vibIntensity != None and last_state[2].vibIntensity != None: 
                    total_priority[2] += 5*abs(t2.vibIntensity - last_state[2].vibIntensity) 
                if t2.ledList is not None:
                    total_priority[2] += (int)((sum(abs(x - y) for row1, row2 in zip(t2.ledList, last_state[2].ledList) for x, y in zip(row1, row2)))/255)
                '''t3'''
                if t3.therDiff != last_state[3].therDiff: total_priority[3] += 100
                if t3.vibIntensity != None and last_state[3].vibIntensity != None: 
                    total_priority[3] += 5*abs(t3.vibIntensity - last_state[3].vibIntensity) 
                if t3.ledList is not None:
                    total_priority[3] += (int)((sum(abs(x - y) for row1, row2 in zip(t3.ledList, last_state[3].ledList) for x, y in zip(row1, row2)))/255)
            

            if max(total_priority) == total_priority[0]:
                q_cmd.put_nowait(("useToken", (t0, )))
                # print(t0)
                time_priority[0] = 0
                time_priority[1] += TIMEPRIOIR_SCALE
                time_priority[2] += TIMEPRIOIR_SCALE
                time_priority[3] += TIMEPRIOIR_SCALE
            elif max(total_priority) == total_priority[1]:
                q_cmd.put_nowait(("useToken", (t1, )))
                # print(t1)
                time_priority[0] += TIMEPRIOIR_SCALE
                time_priority[1] = 0
                time_priority[2] += TIMEPRIOIR_SCALE
                time_priority[3] += TIMEPRIOIR_SCALE
            elif max(total_priority) == total_priority[2]:
                q_cmd.put_nowait(("useToken", (t2, )))
                # print(t2)
                time_priority[0] += TIMEPRIOIR_SCALE
                time_priority[1] += TIMEPRIOIR_SCALE
                time_priority[2] = 0
                time_priority[3] += TIMEPRIOIR_SCALE
            elif max(total_priority) == total_priority[3]:
                q_cmd.put_nowait(("useToken", (t3, )))
                # print(t3)
                time_priority[0] += TIMEPRIOIR_SCALE
                time_priority[1] += TIMEPRIOIR_SCALE
                time_priority[2] += TIMEPRIOIR_SCALE
                time_priority[3] = 0
            
            last_state.clear()
            last_state.append(t0)
            last_state.append(t1)
            last_state.append(t2)
            last_state.append(t3)
            total_priority = time_priority

        except pyqueue.Full:
            print("q_cmd full!!!")