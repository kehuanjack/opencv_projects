import cv2  
import numpy as np

cap = cv2.VideoCapture(0) 
fps = cap.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter('out.mp4',cv2.VideoWriter_fourcc('M','J','P','G'),fps,(int(cap.get(3)),int(cap.get(4))))

# def empty(a):
#     global h_min
#     global h_max
#     global s_min
#     global s_max
#     global v_min
#     global v_max
#     h_min = cv2.getTrackbarPos("Hue Min", "TrackBars")
#     h_max = cv2.getTrackbarPos("Hue Max", "TrackBars")
#     s_min = cv2.getTrackbarPos("Sat Min", "TrackBars")
#     s_max = cv2.getTrackbarPos("Sat Max", "TrackBars")
#     v_min = cv2.getTrackbarPos("Val Min", "TrackBars")
#     v_max = cv2.getTrackbarPos("Val Max", "TrackBars")
#     print(h_min, h_max, s_min, s_max, v_min, v_max)

# cv2.namedWindow("TrackBars")
# cv2.resizeWindow("TrackBars", 640, 240)
# cv2.createTrackbar("Hue Min", "TrackBars", 0, 179, empty)
# cv2.createTrackbar("Hue Max", "TrackBars", 100, 179, empty)
# cv2.createTrackbar("Sat Min", "TrackBars", 110, 255, empty)
# cv2.createTrackbar("Sat Max", "TrackBars", 240, 255, empty)
# cv2.createTrackbar("Val Min", "TrackBars", 153, 255, empty)
# cv2.createTrackbar("Val Max", "TrackBars", 255, 255, empty)

# empty(0)

kalman = cv2.KalmanFilter(4,2)
#设置测量矩阵
kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32)
#设置转移矩阵
kalman.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],np.float32)
#设置过程噪声协方差矩阵
kalman.processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],np.float32)*0.03

last_measurement = current_measurement = np.array((2,1),np.float32)
last_predicition = current_prediction = np.zeros((2,1),np.float32)

while (cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        # 截获乒乓球颜色
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h_min, h_max, s_min, s_max, v_min, v_max = [11,22,144,255,191,255] # 寻找临界值时，请注释掉这行
        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])

        #乒乓球掩膜
        mask = cv2.inRange(hsv, lower, upper) # 低于lower的值，高于upper的值变为0，而在之间的值变为255
        res = cv2.bitwise_and(frame, frame, mask=mask)
        # cv2.imshow("res",res) # 显示掩膜效果

        #图像处理
        gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((10, 10), np.uint8)
        ret, binary1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        dst = cv2.morphologyEx(binary1, cv2.MORPH_CLOSE, kernel,iterations=2)
        
        #获取轮廓
        contours1, hierarchy = cv2.findContours(dst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
        #画轮廓图，打印圆心坐标
        n = len(contours1)
        if n != 0:
            for i in range(n):
                (x, y), radius = cv2.minEnclosingCircle(contours1[i])
                center = (int(x), int(y))
                radius = int(radius)
            # cv2.circle(frame, center, radius, (255, 0, 0), 2)  # 轮廓中像素坐标
            # cv2.circle(frame, center, 2, (255, 0, 0), -1)
            # name = "(" + str(center[0]) + "," + str(center[1]) + ")"
            # cv2.putText(frame, name, (int(center[0]) + 3, int(center[1]) - 3), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,0,0), 2)
            # print(center)

            last_measurement = current_measurement
            last_prediction = current_prediction
            #传递当前测量坐标值
            current_measurement = np.array([[np.float32(x)],[np.float32(y)]])
            #用来修正卡尔曼滤波的预测结果
            kalman.correct(current_measurement)
            # 调用kalman这个类的predict方法得到状态的预测值矩阵，用来估算目标位置
            current_prediction = kalman.predict()
            
            #上一次测量值
            lmx,lmy = last_measurement[0],last_measurement[1]
            #当前测量值
            cmx,cmy = current_measurement[0],current_measurement[1]
            #上一次预测值
            lpx,lpy = last_prediction[0],last_prediction[1]
            #当前预测值
            cpx,cpy = current_prediction[0],current_prediction[1]

            cv2.circle(frame, (int(cmx),int(cmy)), radius, (255, 0, 0), 2)
            # cv2.circle(frame, (int(cmx),int(cmy)), 2, (255, 0, 0), -1)
            cv2.circle(frame, (int(cpx),int(cpy)), radius, (0, 0, 255), 2)
            # cv2.circle(frame, (int(cpx),int(cpy)), 2, (0, 0, 255), -1)
            cv2.arrowedLine(frame, (int(cmx),int(cmy)), (int(cpx),int(cpy)), (0, 0, 0), 2)
            # #绘制测量值轨迹（绿色）
            # cv2.line(frame,(int(lmx),int(lmy)),(int(cmx),int(cmy)),(255,0,0))
            # #绘制预测值轨迹（红色）
            # cv2.line(frame,(int(lpx),int(lpy)),(int(cpx),int(cpy)),(0,0,255))

    
        cv2.imshow('frame', frame)
        out.write(frame)
        c = cv2.waitKey(int(fps))
        if c == 27:
            break
    else:break

cap.release()
out.release()
cv2.destroyAllWindows()