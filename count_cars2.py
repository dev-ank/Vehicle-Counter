import cv2

def count_vehicles_only(vid_path,line_1_start,line_1_end,line2_start,line_2_end): 
    
    '''
    This function takes the video path and the starting and ending cooridinates of two lines
    as arguments and counts the number of vehicles crossing the line.
    
    '''
    
    model = cv2.createBackgroundSubtractorMOG2()  #Using Background Subtraction technique to subtract background from foreground
    vid=cv2.VideoCapture(vid_path)
    
    vehicle_count=0
    
    while vid.isOpened():
        
        ret,curr_frame=vid.read()                 #reading each frame of video
        subtracted_frame=model.apply(curr_frame)  #background subtraction 
        
        cv2.line(curr_frame,line_1_start,line_1_end,(0,0,255),2)
        cv2.line(curr_frame,line_2_start,line_2_end,(0,0,255),2)
        
        conts,_=cv2.findContours(subtracted_frame, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)  #finding all the countours in 1 frame
        
        for i in conts:
            x,y,w,h=cv2.boundingRect(i)
            
            if (w>57) and (h>62) and (y>250) and (x>400) and (x<1300):    #condition check for including only required contours
                cv2.rectangle(curr_frame, (x,y), (x+w,y+h), (0,255,0),3)  
                x_mid=int((x+(x+w))/2)                                    #finding mid points 
                y_mid=int((y+(y+h))/2)
                cv2.circle(curr_frame,(x_mid,y_mid),4,(0,0,255),4)        #creating circle using mid points
                
                if y_mid>line_2_start[1] and y_mid<line_1_start[1]:       #if circle points crosses line , increment count variable
                    vehicle_count+=1
                        
                
        cv2.putText(curr_frame,f'Total Count: {vehicle_count}',(700,60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0),3)
        cv2.imshow('Vehicle Counter Application',curr_frame)
                
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cv2.destroyAllWindows()
    vid.release()
    
    
if __name__=="__main__":
    
    line_1_start=(400,500)
    line_1_end=(1500,500)
    line_2_start=(400,475)
    line_2_end=(1500,475)
    
    path='/Dataset/day2.mp4'
    count_vehicles_only(path,line_1_start,line_1_end,line_2_start,line_2_end)