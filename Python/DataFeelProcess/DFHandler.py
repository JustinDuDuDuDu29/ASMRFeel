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
    t0 = token(superDotID = 2, vibFrequency=20, vibIntensity=None, heatup=False, ledList=None)
    t1 = token(superDotID = 3, vibFrequency=20, vibIntensity=None, heatup=False, ledList=None)
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
    lastP = [0,0,0,0,0,0,0,0]
    pLastIsHit = False 
    p1LastIsHit = False 
    lastP1 = [0,0,0,0,0,0,0,0]
    while not stop_evt.is_set():
        # print(q_pres.qsize(), q_vib.qsize(), q_therm.qsize())
        t = q_pres.get()
        temp, pres1, pres2 = t.split(";")
        p = pres1.split(",")
        p1 = pres2.split(",")
        vib = q_vib.get()
        therm = q_therm.get()

        # ------Thermal Feedback------------------
        toneLeft = therm[0]
        toneRight = therm[1]

        TONE_THRESHOLD = 0.24
        DURATION_THRESHOLD = 0.5
        vibFreqLeft = 0
        vibFreqRight = 0

        # if vib < 0.3:
        #     vib = 0

        if vib[0] > 0:
            vibFreqLeft = 100
        if vib[1] > 0:
            vibFreqRight = 100


        heatUpLeft = False
        heatUpRight = False

        if toneLeft >= TONE_THRESHOLD:
            if start == False:
                last_trigger = time.perf_counter()
                start = True
            elif time.perf_counter() - last_trigger >= DURATION_THRESHOLD:
                print("Heating Left")
                heatUpLeft = True
        elif toneLeft < TONE_THRESHOLD:
            if startCool == False:
                lastCool_trigger = time.perf_counter()
                startCool = True
            elif time.perf_counter() - lastCool_trigger >= DURATION_THRESHOLD*3:
                start = False
                startCool = False
                print("Cooling Left")
                heatUpLeft = False



        if toneRight >= TONE_THRESHOLD:
            if start == False:
                last_trigger = time.perf_counter()
                start = True
            elif time.perf_counter() - last_trigger >= DURATION_THRESHOLD:
                print("Heating Right")
                heatUpRight = True
        elif toneRight < TONE_THRESHOLD:
            if startCool == False:
                lastCool_trigger = time.perf_counter()
                startCool = True
            elif time.perf_counter() - lastCool_trigger >= DURATION_THRESHOLD*3:
                start = False
                startCool = False
                print("Cooling Right")
                heatUpRight = False

        # print(q_pres.empty(), q_vib.empty(), q_therm.empty())
        # print(q_pres.qsize(), q_vib.qsize(), q_therm.qsize())

        # print(f"pres {t1 - t0:.3f}s, vib {t2 - t1:.3f}s, therm {t3 - t2:.3f}s")

        '''Hand'''
        t2 = token(superDotID = 0, vibFrequency=0, vibIntensity=0, heatup=False, ledList=[[0,0,0]]*8)
        t3 = token(superDotID = 1, vibFrequency=0, vibIntensity=0, heatup=False, ledList=[[0,0,0]]*8)

        numsT2 = 0

        for i, val in enumerate(p):
            if float(val)>25:
                # led[i] = [random.randint(0,255), random.randint(0,255), random.randint(0,255)]
                # led[i] = [int(val), int(val), int(val)]
                # map int(val) from 0-1023 to 0-255
                # led[i] = [int(val) // 4] * 3
                numsT2 += 1
                if((float(val) - float(lastP[i])) > 200) and numsT2 > 1 :
                    pLastIsHit = True
                    print("Hitting")
                    t2.vibIntensity = 1
                    t2.vibFrequency = 100
                    t2.heatup = True
                    t2.ledList = [[255, 0, 0]]*8
                    break

                if t2.vibIntensity is not None:
                    t2.vibFrequency = 10
                    t2.vibIntensity= max(t2.vibIntensity, int(val) / 1023.0)
                else: 
                    t2.vibFrequency = 10
                if t2.ledList is None:
                    t2.ledList = [[0,0,0]] * 8
                t2.ledList[i] = [int(val) // 4] * 3

                # t2.ledList[i] = [255, 0, 0]
        else: 
            if numsT2 > 4 and not pLastIsHit:
                print(">4")
                t2.vibIntensity = .2
                t2.vibFrequency = 100
                t2.heatup = True
                t2.ledList = [[0, 255, 0]]*8
            pLastIsHit = False
            

        lastP = p 


        numsT3 = 0
        for i, val in enumerate(p1):
            if int(val)>60:
                # led[i] = [random.randint(0,255), random.randint(0,255), random.randint(0,255)]
                # led[i] = [int(val), int(val), int(val)]
                # map int(val) from 0-1023 to 0-255
                numsT3 += 1

                if((float(val) - float(lastP1[i])) > 200) and numsT3 > 1 :
                    p1LastIsHit = True
                    print("Hitting")
                    t3.vibIntensity = 1
                    t3.vibFrequency = 100
                    t3.heatup = True
                    t3.ledList = [[255, 0, 0]]*8
                    break
                if t3.vibIntensity is not None:
                    t3.vibFrequency = 10
                    t3.vibIntensity= max(t3.vibIntensity, int(val) / 1023.0)
                else: 
                    t2.vibFrequency = 10
                if t3.ledList is None:
                    t3.ledList = [[0,0,0]] * 8
                t3.ledList[i] = [int(val) // 4] * 3
                
        else: 
            if numsT3 > 4 and not p1LastIsHit:
                print(">4")
                t3.vibIntensity = .2
                t3.vibFrequency = 100
                t3.heatup = True
                t3.ledList = [[0, 255, 0]]*8
            p1LastIsHit = False
                # t3.ledList[i] = [255, 0, 0]
            lastP1 = p1

        # print(vib, therm)
        '''HeadPhone'''

        t0 = token(superDotID = 2, vibFrequency=vibFreqLeft, vibIntensity=vib[0], heatup=heatUpLeft, ledList=[[255,0,0]]*8)
        t1 = token(superDotID = 3, vibFrequency=vibFreqRight, vibIntensity=vib[1], heatup=heatUpRight, ledList=None)
        # print(t0)
        # print(t0)
        try:
            # print(q_cmd.qsize())
            q_cmd.put_nowait(("useToken", (t0, )))
            q_cmd.put_nowait(("useToken", (t1, )))
            q_cmd.put_nowait(("useToken", (t2, )))
            q_cmd.put_nowait(("useToken", (t3, )))
            # print(time.perf_counter()-last)

        except pyqueue.Full:
            print("q_cmd full!!!")
    
