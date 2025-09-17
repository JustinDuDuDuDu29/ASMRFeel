from multiprocessing.synchronize import Event
from multiprocessing import  Process, Queue
from DataFeelCenter import DataFeelCenter, token
import queue as pyqueue
import time
from Config import Config

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
    t0 = token(superDotID = 2, vibFrequency=20, vibIntensity=None, heatup=False, ledList=None)
    t1 = token(superDotID = 3, vibFrequency=20, vibIntensity=None, heatup=False, ledList=None)
    cmd = ("useToken", (t0, ))
    cmd2 = ("useToken", (t1, ))
    method, args = cmd
    getattr(dfc, method)(*args)
    method, args = cmd2
    getattr(dfc, method)(*args)

def Commander(stop_evt: Event, q_pres:Queue, q_vib:Queue, q_therm:Queue, q_cmd:Queue, q_unity:Queue):
    start = False
    startCool = False
    last_trigger = time.perf_counter()
    lastCool_trigger = time.perf_counter()
    redValue = 0
    dredValue = 0
    lastled1 = [[0,0,0]]*8
    lastled2 = [[0,0,0]]*8
    buffer = []
    start_time = time.time()
    while not stop_evt.is_set():
        # print(q_pres.qsize(), q_vib.qsize(), q_therm.qsize())
        t = q_pres.get()
        temp, pres1, pres2 = t.split(";")
        p = pres1.split(",")
        p1 = pres2.split(",")
        vib = q_vib.get()
        therm = q_therm.get()

        # ------Thermal Feedback------------------
        tone_smooth = therm

        TONE_THRESHOLD = 0.24
        DURATION_THRESHOLD = 0.5
        vibFreq = 0
        vibFreq1 = 0

        # if vib < 0.3:
        #     vib = 0

        if vib > 0:
            vibFreq = 10
            vibFreq1 = 10

        thermVal = None
        heatup = False
        # thermDiff = None


        if tone_smooth >= TONE_THRESHOLD:
            # print("PreHeating...")
            # print("PreHeating...")
            if start == False:
                # print("Heating Count")
                # print("Heating Count")
                last_trigger = time.perf_counter()
                start = True
            elif time.perf_counter() - last_trigger >= DURATION_THRESHOLD:
                print("Heating")
                heatup = True
        elif tone_smooth < TONE_THRESHOLD:
            # print("PreCooling...")
            # print("PreCooling...")
            if startCool == False:
                lastCool_trigger = time.perf_counter()
                startCool = True
            elif time.perf_counter() - lastCool_trigger >= DURATION_THRESHOLD*3:
                start = False
                startCool = False
                print("Cooling")
                heatup = False


        # print(q_pres.empty(), q_vib.empty(), q_therm.empty())
        # print(q_pres.qsize(), q_vib.qsize(), q_therm.qsize())

        # print(f"pres {t1 - t0:.3f}s, vib {t2 - t1:.3f}s, therm {t3 - t2:.3f}s")

        '''Hand'''
        t2 = token(superDotID = 0, vibFrequency=0, vibIntensity=0, heatup=False, ledList=[[0,0,0]]*8)
        t3 = token(superDotID = 1, vibFrequency=0, vibIntensity=0, heatup=False, ledList=[[0,0,0]]*8)

        for i, val in enumerate(p):
            if int(val)>60:
                # led[i] = [random.randint(0,255), random.randint(0,255), random.randint(0,255)]
                # led[i] = [int(val), int(val), int(val)]
                # map int(val) from 0-1023 to 0-255
                # led[i] = [int(val) // 4] * 3

                if t2.vibIntensity is not None:
                    t2.vibFrequency = 10
                    t2.vibIntensity= max(t2.vibIntensity, int(val) / 512.0)
                else: 
                    t2.vibFrequency = 10
                if t2.ledList is None:
                    t2.ledList = [[0,0,0]] * 8
                t2.ledList[i] = [int(val) // 4] * 3

                # t2.ledList[i] = [255, 0, 0]

        for i, val in enumerate(p1):
            if int(val)>60:
                # led[i] = [random.randint(0,255), random.randint(0,255), random.randint(0,255)]
                # led[i] = [int(val), int(val), int(val)]
                # map int(val) from 0-1023 to 0-255

                if t3.vibIntensity is not None:
                    t3.vibFrequency = 10
                    t3.vibIntensity= max(t3.vibIntensity, int(val) / 512.0)
                else: 
                    t2.vibFrequency = 10
                if t3.ledList is None:
                    t3.ledList = [[0,0,0]] * 8
                t3.ledList[i] = [int(val) // 4] * 3
                
                # t3.ledList[i] = [255, 0, 0]


        # print(vib, therm)
        '''HeadPhone'''

        t0 = token(superDotID = 2, vibFrequency=vibFreq, vibIntensity=vib, heatup=heatup, ledList=[[0,0,0]]*8)
        t1 = token(superDotID = 3, vibFrequency=vibFreq1, vibIntensity=vib, heatup=heatup, ledList=[[0,0,0]]*8)

        try:
            if heatup:
                redValue += Config.AUDIO_CHUNK_MS/2000.0
                if redValue >= 1.0: redValue = 1.0
            else:
                redValue -= Config.AUDIO_CHUNK_MS/2000.0
                if redValue <= 0.0: redValue = 0.0
            buffer.append(redValue)
            print(redValue)
        except pyqueue.Empty:
            pass

        if buffer and (time.time() - start_time) > Config.AUDIO_PLAYBACK_DELAY_S:
            dredValue = buffer.pop(0)
        
        

        if t0.ledList is None:
            t0.ledList = [[0,0,0]] * 8

        if vib > 0.2:
            for i in range(8):
                t0.ledList[i] = [int(vib*255), int((1-dredValue)*vib*255), int((1-dredValue)*vib*255)]
                # if i == 0: print(t0.ledList[0])
        else:
            t0.ledList = lastled1
        
        lastled1 = t0.ledList

        if t1.ledList is None:
            t1.ledList = [[0,0,0]] * 8
        
        if vib > 0.2:
            for i in range(8):
                t1.ledList[i] = [int(vib*255), int((1-dredValue)*vib*255), int((1-dredValue)*vib*255)]
        else:
            t1.ledList = lastled2

        lastled2 = t1.ledList

        # print(t2.ledList)

        try:
            # print(q_cmd.qsize())
            q_cmd.put_nowait(("useToken", (t0, )))
            q_cmd.put_nowait(("useToken", (t1, )))
            q_cmd.put_nowait(("useToken", (t2, )))
            q_cmd.put_nowait(("useToken", (t3, )))
            q_unity.put_nowait((t0, t1, t2, t3))
            
            # print(time.perf_counter()-last)

        except pyqueue.Full:
            print("q_cmd full!!!")
    
