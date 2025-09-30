from multiprocessing.synchronize import Event
from multiprocessing import  Process, Queue
from DataFeelCenter import DataFeelCenter, token
import queue as pyqueue
import time
from Config import Config
def swap_elements(lst, *pairs):
    for pair in pairs:
        idx1, idx2 = pair
        lst[idx1], lst[idx2] = lst[idx2], lst[idx1]
    return lst
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
    lastP = [0,0,0,0,0,0,0,0]
    pLastIsHit = False 
    p1LastIsHit = False 
    lastP1 = [0,0,0,0,0,0,0,0]
    redValue = 0
    dredValue = 0
    right_redValue = 0
    right_dredValue = 0
    lastled1 = [[0,0,0]]*8
    lastled2 = [[0,0,0]]*8
    buffer = []
    right_buffer = []
    start_time = time.time()

    '''Smoothing Parameters'''
    prev_left_vib  = 0.0
    prev_right_vib = 0.0

    # 上升（攻擊）與下降（釋放）的平滑係數（0~1，越大變化越快）
    ATTACK_ALPHA  = 0.7   # 例如 0.25~0.5 之間自行微調
    RELEASE_ALPHA = 0.2   # 例如 0.05~0.2 之間自行微調

    def smooth_step(prev: float, target: float, attack: float, release: float) -> float:
        alpha = attack if target > prev else release
        return prev + alpha * (target - prev)
    ''''''
    
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


        if Config.STREAMING:
            TONE_THRESHOLD = 0.25
            DURATION_THRESHOLD = 0.5
        else:
            #record mode
            TONE_THRESHOLD = 0.3
            DURATION_THRESHOLD = 0.25
        
        vibFreqLeft = 0
        vibFreqRight = 0

        '''Vibration Adjustment------------------'''
        ini_vib = (
                vib[0],
                vib[1]
            )
        # vib = (
        #         0 if vib[0] < (5.0*Config.HVIB_OUT_SCALE/20.0) else vib[0],
        #         0 if vib[1] < (5.0*Config.HVIB_OUT_SCALE/20.0) else vib[1]
        #     )
        '''Vibration Adjustment------------------'''

        if vib[0] > 0:
            vibFreqLeft = 125
        if vib[1] > 0:
            vibFreqRight = 125


        heatUpLeft = False
        heatUpRight = False
        dheatUpLeft = False
        dheatUpRight = False

        if toneLeft >= TONE_THRESHOLD:
            if start == False:
                last_trigger = time.perf_counter()
                start = True
            elif time.perf_counter() - last_trigger >= DURATION_THRESHOLD:
                # print("Heating Left")
                heatUpLeft = True
        elif toneLeft < TONE_THRESHOLD:
            if startCool == False:
                lastCool_trigger = time.perf_counter()
                startCool = True
            elif time.perf_counter() - lastCool_trigger >= DURATION_THRESHOLD*3:
                start = False
                startCool = False
                # print("Cooling Left")
                heatUpLeft = False



        if toneRight >= TONE_THRESHOLD:
            if start == False:
                last_trigger = time.perf_counter()
                start = True
            elif time.perf_counter() - last_trigger >= DURATION_THRESHOLD:
                # print("Heating Right")
                heatUpRight = True
        elif toneRight < TONE_THRESHOLD:
            if startCool == False:
                lastCool_trigger = time.perf_counter()
                startCool = True
            elif time.perf_counter() - lastCool_trigger >= DURATION_THRESHOLD*3:
                start = False
                startCool = False
                # print("Cooling Right")
                heatUpRight = False

        # print(q_pres.empty(), q_vib.empty(), q_therm.empty())
        # print(q_pres.qsize(), q_vib.qsize(), q_therm.qsize())

        # print(f"pres {t1 - t0:.3f}s, vib {t2 - t1:.3f}s, therm {t3 - t2:.3f}s")

        '''Hand'''
        t2 = token(superDotID = 0, vibFrequency=0, vibIntensity=0, heatup=False, ledList=[[0,0,0]]*8)
        t3 = token(superDotID = 1, vibFrequency=0, vibIntensity=0, heatup=False, ledList=[[0,0,0]]*8)

        numsT2 = 0
        p1 = swap_elements(p1, (7,3), (6,2), (5,1), (4,0),(0,3),(1,2))
        p = swap_elements(p, (0,3),(1,2))

        for i, val in enumerate(p):
            if float(val)>120:
                # print(i)
                # led[i] = [random.randint(0,255), random.randint(0,255), random.randint(0,255)]
                # led[i] = [int(val), int(val), int(val)]
                # map int(val) from 0-1023 to 0-255
                # led[i] = [int(val) // 4] * 3
                if int(val) >= 110: numsT2 += 1

                # if((float(val) - float(lastP[i])) > 130) and numsT2 > 3 :
                #     pLastIsHit = True
                #     print("Hitting")
                #     t2.vibIntensity = 1*Config.HVIB_OUT_SCALE
                #     t2.vibFrequency = 100
                #     t2.heatup = True
                #     t2.ledList = [[255, 0, 0]]*8
                #     break

                if t2.vibIntensity is not None:
                    t2.vibFrequency = 10
                    t2.vibIntensity= max(t2.vibIntensity, int(val) / 512.0)*Config.HVIB_OUT_SCALE
                else: 
                    t2.vibFrequency = 10
                if t2.ledList is None:
                    t2.ledList = [[0,0,0]] * 8
                t2.ledList[i] = [int(val) // 4] * 3

                # t2.ledList[i] = [255, 0, 0]
        else: 
            if numsT2 > Config.HOLD_COUNT and not pLastIsHit:
                print(">4")
                t2.vibIntensity = .2*Config.HVIB_OUT_SCALE
                t2.vibFrequency = 98
                t2.heatup = True
                t2.ledList = [[0, 255, 0]]*8
            pLastIsHit = False
            

        lastP = p 


        numsT3 = 0
        for i, val in enumerate(p1):
            if int(val)>120:
                # print(i)
                # led[i] = [random.randint(0,255), random.randint(0,255), random.randint(0,255)]
                # led[i] = [int(val), int(val), int(val)]
                # map int(val) from 0-1023 to 0-255
                if int(val) >= 110: numsT3 += 1

                # if((float(val) - float(lastP1[i])) > 130) and numsT3 > 3 :
                #     p1LastIsHit = True
                #     print("Hitting")
                #     t3.vibIntensity = 1*Config.HVIB_OUT_SCALE
                #     t3.vibFrequency = 100
                #     t3.heatup = True
                #     t3.ledList = [[255, 0, 0]]*8
                #     break
                if t3.vibIntensity is not None:
                    t3.vibFrequency = 10
                    t3.vibIntensity= max(t3.vibIntensity, int(val) / 512.0)*Config.HVIB_OUT_SCALE
                else: 
                    t2.vibFrequency = 10
                if t3.ledList is None:
                    t3.ledList = [[0,0,0]] * 8
                t3.ledList[i] = [int(val) // 4] * 3
                
        else: 
            if numsT3 > Config.HOLD_COUNT and not p1LastIsHit:
                print(">4")
                t3.vibIntensity = .2*Config.HVIB_OUT_SCALE
                t3.vibFrequency = 98
                t3.heatup = True
                t3.ledList = [[0, 255, 0]]*8
            p1LastIsHit = False
                # t3.ledList[i] = [255, 0, 0]
            lastP1 = p1

        # print(vib, therm)
        '''HeadPhone'''

        t0 = token(superDotID = 3, vibFrequency=vibFreqLeft, vibIntensity=vib[0], heatup=heatUpLeft, ledList=[[0,0,0]]*8)
        t1 = token(superDotID = 2, vibFrequency=vibFreqRight, vibIntensity=vib[1], heatup=heatUpRight, ledList=[[0,0,0]]*8)
        # print(vib[0], vib[1])
        # print(t0)
        # print(t0)
        try:
            if heatUpLeft:
                redValue += Config.AUDIO_CHUNK_MS/2000.0
                if redValue >= 1.0: redValue = 1.0
            else:
                redValue -= Config.AUDIO_CHUNK_MS/2000.0
                if redValue <= 0.0: redValue = 0.0
            buffer.append((redValue, heatUpLeft))
        except pyqueue.Empty:
            pass


        try:
            if heatUpRight:
                right_redValue += Config.AUDIO_CHUNK_MS/2000.0
                if right_redValue >= 1.0: right_redValue = 1.0
            else:
                right_redValue -= Config.AUDIO_CHUNK_MS/2000.0
                if right_redValue <= 0.0: right_redValue = 0.0
            right_buffer.append((right_redValue, heatUpRight))
        except pyqueue.Empty:
            pass

        
        if buffer and (time.time() - start_time) > Config.AUDIO_PLAYBACK_DELAY_S:
            (dredValue, dheatUpLeft) = buffer.pop(0)
            if t0.ledList is None:
                t0.ledList = [[0,0,0]] * 8

            for i in range(8):
                if dredValue:
                    t0.ledList[i] = [int(dredValue*255), 0, 0]
                else:
                    t0.ledList[i] = [int(vib[0]*255), int(vib[0]*255), int(vib[0]*255)]
                # if i == 0: print(t0.ledList[0])
            
            leftVib = vib[0]

            # if t0.vibIntensity is not None and t0.vibIntensity <= 0.1:
            # if dheatUpLeft:
            if dredValue > 0.15:
                leftVib = 0.2* Config.VIB_OUT_SCALE

            trueVibLeft  = smooth_step(prev_left_vib,  leftVib,  ATTACK_ALPHA, RELEASE_ALPHA)
            t0.vibIntensity  = trueVibLeft
            prev_left_vib = trueVibLeft

        if right_buffer and (time.time() - start_time) > Config.AUDIO_PLAYBACK_DELAY_S:
            (right_dredValue, dheatUpRight) = right_buffer.pop(0)
            if t1.ledList is None:
                t1.ledList = [[0,0,0]] * 8
            
            for i in range(8):
                if right_dredValue:
                    t1.ledList[i] = [int(right_dredValue*255), 0, 0]
                else:
                    t1.ledList[i] = [int(vib[1]*255), int(vib[1]*255), int(vib[1]*255)]

            rightvib = vib[1]

            # if t1.vibIntensity is not None and t1.vibIntensity <= 0.1:
            # if dheatUpRight:
            if right_dredValue > 0.15:
                rightvib = 0.2* Config.VIB_OUT_SCALE

            trueVibRight  = smooth_step(prev_right_vib,  rightvib,  ATTACK_ALPHA, RELEASE_ALPHA)
            t1.vibIntensity  = trueVibRight
            prev_right_vib = trueVibRight


        # print(t2.ledList)

        try:
            # print(q_cmd.qsize())
            q_cmd.put_nowait(("useToken", (t0, True)))
            q_cmd.put_nowait(("useToken", (t1, True)))
            q_cmd.put_nowait(("useToken", (t2, )))
            q_cmd.put_nowait(("useToken", (t3, )))
            q_unity.put_nowait((t0, t1, t2, t3))
            
            # print(time.perf_counter()-last)

        except pyqueue.Full:
            print("q_cmd full!!!")
    
