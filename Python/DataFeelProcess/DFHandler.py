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
    
    '''Cool Down'''
    t0 = token(superDotID = 2, vibFrequency=20, vibIntensity=None, therIntensity=None, therDiff=-3, ledList=None)
    t1 = token(superDotID = 3, vibFrequency=20, vibIntensity=None, therIntensity=None, therDiff=-3, ledList=None)
    cmd = ("useToken", (t0, ))
    cmd2 = ("useToken", (t1, ))
    method, args = cmd
    getattr(dfc, method)(*args)
    method, args = cmd2
    getattr(dfc, method)(*args)

def Commander(stop_evt: Event, q_pres:Queue, q_vib:Queue, q_therm:Queue, q_cmd:Queue):
    start = False
    startCool = False
    last_trigger = time.perf_counter()
    lastCool_trigger = time.perf_counter()
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
            # print("PreHeating...")
            if start == False:
                # print("Heating Count")
                # print("Heating Count")
                last_trigger = time.perf_counter()
                start = True
            elif time.perf_counter() - last_trigger >= DURATION_THRESHOLD:
                print("Heating...")
                thermDiff = 4
        elif tone_smooth < TONE_THRESHOLD:
            # print("PreCooling...")
            # print("PreCooling...")
            if startCool == False:
                lastCool_trigger = time.perf_counter()
                startCool = True
            elif time.perf_counter() - lastCool_trigger >= DURATION_THRESHOLD*3:
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
        for i, val in enumerate(p1):
            if int(val)>60:
                # led[i] = [random.randint(0,255), random.randint(0,255), random.randint(0,255)]
                # led[i] = [int(val), int(val), int(val)]
                # map int(val) from 0-1023 to 0-255
                # led[i] = [int(val) // 4] * 3
                led1[i] = [255, 0, 0]


        # print(vib, therm)
        '''HeadPhone'''
        t0 = token(superDotID = 2, vibFrequency=10, vibIntensity=vib, therIntensity=thermVal, therDiff=thermDiff, ledList=None)
        t1 = token(superDotID = 3, vibFrequency=10, vibIntensity=vib, therIntensity=thermVal, therDiff=thermDiff, ledList=None)
        '''Hand'''
        t2 = token(superDotID = 0, vibFrequency=None, vibIntensity=None, therIntensity=None, therDiff=None, ledList=led)
        t3 = token(superDotID = 1, vibFrequency=None, vibIntensity=None, therIntensity=None, therDiff=None, ledList=led1)
        # print(t0)
        try:
            q_cmd.put_nowait(("useToken", (t0, )))
            q_cmd.put_nowait(("useToken", (t1, )))
            q_cmd.put_nowait(("useToken", (t2, )))
            q_cmd.put_nowait(("useToken", (t3, )))
            q_cmd.put_nowait(("useToken", (t2, )))
            q_cmd.put_nowait(("useToken", (t3, )))
            # print(time.perf_counter()-last)

        except pyqueue.Full:
            print("q_cmd full!!!")
    