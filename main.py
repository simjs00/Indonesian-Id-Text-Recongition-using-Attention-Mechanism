import cv2
cv2.setUseOptimized(True)
#import model as detector_model
import numpy as np
import detector.ocr2 as hist
#from detector.ensorFlowDeepAutoencoder.code.aeimport autoencoder_test as detector_model
#import detector.TensorFlowDeepAutoencoder.code.run as detector_model
import detector.cnn.test_model as detector_model
from recognizer.src import launcher as recognizer_model
#import PIL.Image as PIL
from PIL import Image
import sys
import time
from datetime import timedelta
col_texts =[]
image_inf =[]
def swap( A,B, x, y ):
  tmp = A[x]
  tmp1 = B[x]


  A[x] = A[y]
  A[y] = tmp


  B[x] = B[y]
  B[y] = tmp1

def partition(myList,B, start, end):
    pivot = myList[start]
    left = start+1
    right = end
    done = False
    while not done:
        while left <= right and myList[left] <= pivot:
            left = left + 1
        while myList[right] >= pivot and right >=left:
            right = right -1
        if right < left:
            done= True
        else:
            # swap places
            temp=myList[left]
            myList[left]=myList[right]
            myList[right]=temp

            temp1=B[left]
            B[left]=B[right]
            B[right]=temp1
    		


    # swap start with myList[right]
    temp=myList[start]
    myList[start]=myList[right]
    myList[right]=temp



    temp1=B[start]
    B[start]=B[right]
    B[right]=temp1

    return right
def quicksort(A,B, start, end):
    if start < end:
        # partition the list
        pivot = partition(A,B, start, end)
        # sort both halves
        quicksort(A,B, start, pivot-1)
        quicksort(A,B, pivot+1, end)
    return A,B

def text_ext2(img,k,j,pil_img) :


	h_or = int(round(img.shape[0]))
	w_or = int(round(img.shape[1]))
	
	#Create MSER object
	#mser = cv2.MSER_create()

	#Your image path i-e receipt path
	#img = cv2.imread('/home/rafiullah/PycharmProjects/python-ocr-master/receipts/73.jpg')

	#Convert to gray scale

	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	ret,gray = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
	

	c_img = cv2.dilate(gray,None,iterations =2)
	#c_img = hist.lrsm(gray,w_or,h_or,6,1)



	#cv2.imwrite('text2/img'+str(k)+".png", c_img)
	#cv2.imwrite('text2/img_new'+str(k)+".png", new)
	freq,histx =hist.compute_hist(img=c_img,prj=1,h=h_or,w=w_or)
	stack_start,stack_end =hist.segment2 ("test",None, img, freq)
	#print(stack_start)
	#print(stack_end)
	labels=[]
	i=0
	while(i<len(stack_end)) :



		roi=img[0:0+h_or,stack_start[i]:stack_end[i]+1]
		image_inf[k-1].append(roi)	
		#start_time = time.monotonic()
		#start_time = time.time()
		#label =recognizer_model.test(roi)
		#print(time.time()-start_time)
		#end_time = time.monotonic()
		#print(timedelta(seconds=end_time - start_time))
		#labels.append(label)
		#sys.stdout.write(label+" ")
		#print(label+" ")
		#print("\n")
		cv2.imwrite( "./text2/test_"+str(k)+"_"+str(j)+"_"+str(i)+".png", roi)

		#cv2.imwrite( "./text2/test_"+str(k)+"_"+str(i)+".png", roi)
		i+=1
	col_texts.append(labels)
    
def text_ext(new,image,k,pil_img) :

            image_inf.append([])
            x_cord=[]
            w_cord=[]

            h_or = int(round(image.shape[0]))
            w_or = int(round(image.shape[1]))
            #image = cv2.imread(fname)
            #image = cv2.bilateralFilter(image,9,75,75)
            #cv2.imshow("Original",image)
            #cv2.waitKey(1)
            g_img = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            #g_img = cv2.equalizeHist(g_img)

            #cv2.imshow("Gray",g_img)
            #cv2.waitKey(1)
            # Options MORPH_ELLIPSE, RECT, CROSS
            #morph_kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
            morph_kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
            mg_img = cv2.morphologyEx(g_img,cv2.MORPH_GRADIENT,morph_kernel)
            #cv2.imshow("Morph",mg_img)
            #cv2.waitKey(1)          
            _, bw_img = cv2.threshold(mg_img, 0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

            #cv2.imshow("BW",bw_img)
            #cv2.waitKey(1)
            connect_kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(8,1))
            c_img = cv2.morphologyEx(bw_img, cv2.MORPH_GRADIENT, connect_kernel)
            c_img =cv2.dilate(c_img,None,iterations =1)
            #c_img = hist.lrsm(c_img,w_or,h_or,3,1)

            #cv2.imshow("Connected",c_img)	
            
            #cv2.waitKey(10000000)
            mask=np.zeros(bw_img.shape,np.uint8)
            mask2 = mask.copy()
            _, cnts, _ = cv2.findContours(c_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE, offset=(0,0))

            # Testing Code to find show contour
            #c2=cnts[49]
            #rect=cv2.boundingRect(c2)
            #maskROI2 = cv2.bitwise_and(mask2,rect)
            #maskROI2 = cv2.drawContours(mask2,c2,-1,(255,255,255),thickness=-1)
            #cv2.imshow("Test",maskROI2)
            #cv2.waitKey(1)

            clone= image.copy()
            i = -1
            for j,c in enumerate(cnts):
                    rect = x,y,w,h = cv2.boundingRect(c)
                    maskROI = cv2.bitwise_and(mask, rect)
                    #maskROI = (0,0,0)
                    maskROI = cv2.drawContours(maskROI,[c],-1,(255,255,255),thickness=-1)
                    r = float(cv2.countNonZero(maskROI)) / (w*h)
                    i=i+1
                    #cv2.rectangle(clone,(x,y),(x+w,y+h),(0,255,0),1)
                    #if r > 0.60 and (11  < h and 11 < w):
                    if(w>15) :
                    	if(h>5):
                           x_cord.append(x)
                           w_cord.append(w)
		                    #cv2.imshow("Final",roi)
		                    #t =strftime("%H_%M_%S", gmtime())
		                   
                    #if(i==2) :
                    #window(roi=roi,x1=x,y1=y,w=w,h=h,i=i,ra=2,j=k)
                    #cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
                    #cv2.waitKey(0)
                    #cv2.putText(image,str(i),(x+w,y+h),cv2.FONT_HERSHEY_PLAIN, 0.7, (0,0,255))
                    
                    #print(i,r)
           
            #cv2.imshow("Candidates",clone)
            #cv2.waitKey(1)
            #cv2.imshow("Final",image)
            #cv2.waitKey(0)
            #s_height, s_width, s_channels = scaled.shape
            #psize = "_" + str(s_width) + "x" + str(s_height)
            #fname_new = os.path.join(target_folder,fname+psize+fext)
            #cv2.imwrite(fname_new,scaled)
            #print "==> " + fname_new + " is created.\n"
            #shutil.copy2(source_folder, target_folder) #copies csv to new folder

            print(k)
            labels =[]
            x_cord,w_cord=quicksort(x_cord,w_cord, 0, len(x_cord)-1)
            y = 0
            h = image.shape[0]
            i=0
            while(i<len(x_cord)) :
                x= x_cord[i]
                w= w_cord[i]
                roi=image[y:y+h,x:x+w]
                pil_img = pil_img[y:y+h,x:x+w]
                h_roi = roi.shape[0]
                w_roi = roi.shape[1]
                new,response,pil_img,peak = window(pil_img,w_roi, h_roi , 1,8,h_roi , roi)

                # cv2.imshow('image_new_'+str(i),response)
                # cv2.imshow('image_'+str(i),roi)
                # cv2.waitKey(100000)
                if(peak>2) :
                	text_ext2(roi,k,i,pil_img) 
                #cv2.imwrite( "./text2/test_"+str(k)+"_"+str(i)+".png", roi)
                #label =recognizer_model.test(roi)
                #label=""
                #print(label)
                #sys.stdout.write(label+" ")
                #labels.append(label)
               
                i+=1	

            col_texts.append(labels)
            #cv2.destroyAllWindows()

def make_file(name) :
    sp = name.split('/')[-1]
    sp = sp.split('.')[0]
    path = sp+'.txt'
    #print(col_texts)
    with open(path, 'w') as fword:
        for i, dt in enumerate(col_texts)  :
            string =""

            for j, lbl in enumerate(dt)  :			
                string +=lbl
                if(j<len(dt)-1) :
                    string+=" "
                elif(j==len(dt)-1) :
                    string+="\n"
            fword.write(string)


def window(pil_img,w, h , stride,size_w, size_h, ori) :
	response_score = np.zeros(w)
	response_map = np.zeros((h,w))
	response_map2 = np.zeros((h,w))
	img = cv2.cvtColor(ori,cv2.COLOR_BGR2GRAY)
	i=0

    

	while(i<h) :
		sys.stdout.write('.')
		j=0

		while(j<w) :

			max_i =i+size_h-1
			if(max_i > h) :
				break
				max_i = h
			max_j =j+size_w-1
			if(max_j > w) :
				break
				max_j = w

			roi = img[i:max_i,j:max_j]
			score =0

			if(len(roi)!=0 and np.mean(roi)!=0) :
				if(len(roi[0])>1 ) :
					label,score = detector_model.predict(roi)
					
					if(label=='-') :
						score *=-1

			score *=1
			response_score[j] = score
			# z=0
			# while(z<=j+stride) :
			# 	response_score[i][z] = score	
			# 	z+=1	

			# k=i

			# while(k< h) :
			# 	l=j
			# 	while(l< max_j) : 

			# 		if(score >0) :
			# 			response_map[k][l]=255
			# 		elif(score <0 and response_map[k][l]!=255 ) :
			# 			response_map[k][l]=0

			# 		l+=1

			# 	k+=1
			

			# max_i= 0
			# if(score>0) :
			# 	max_i=int(round(score*(h-1)))
			# if(score >0) :
			# 	l=h-1
			# 	while(l>=max_i) : 

			# 				response_map[l][j]=255

			# 				l-=1

					


			j+=stride



		break
		i+=1
	


	l,r,response_score=hist.nms(response_score,w)
	#if(np.mean(response_score)==0 or hist.compute_peak(response_score)==1 ) :
	# print('mean :',np.mean(response_score))
	# print('peak :',hist.compute_peak(response_score) ) 
	
	# j=0

	# while(j<w) :
	# 		max_i= 0
	# 		if(response_score[j]>0) :
	# 			max_i=int(round(response_score[j]*(h-1)))
	# 			l=h-1
	# 			while(l>=max_i) : 		
	# 				response_map2[l][j]=255
	# 				l-=1
	# 		j+=1

	# cv2.imshow('response2',response_map2)

	#cv2.imshow("ori",ori)
	#cv2.imshow("new",response_map)
	
	#img,response_map,pil_img = hist.vertical2(response_map,ori,pil_img)
	#cv2.imshow("img",img)
	#cv2.imshow("new2",response_map)
	#cv2.waitKey(10000000)
	return img,response_map,pil_img,hist.compute_peak(response_score)

	#response_map = cv2.normalize(response_score,  response_score, 0, 255, cv2.NORM_MINMAX)

        






def receive_input(path) :
	start_time = time.time()
	img_ori = cv2.imread(path)
	pil_img_ori =  np.array(Image.open(path))

	#cv2.imshow('img',img_ori)
    #img =cv2.cvtColor(img_ori,cv2.COLOR_BGR2GRAY)
	
	h = int(round(img_ori.shape[0]))
	w = int(round(img_ori.shape[1]))
    #img_ori = cv2.UMat(img_ori)
	# img=cv2.resize(img,(w,h))

	# response = window(w, h , 1,10,10 , img)
	
	# y_proj_ori,histy =hist.compute_hist(img=response,prj=0,h=h,w=w)
	# stack_start1,stack_end1 = hist.segment3("ori", img, y_proj_ori, w)



	stack_start,stack_end,freq= hist.horizontal(path)

	i=1
	while(i<len(stack_end)) :
		response=[]
		roi = img_ori[stack_end[i-1]-4:stack_start[i]+4, 0:w]
		pil_img = pil_img_ori[stack_end[i-1]-4:stack_start[i]+4, 0:w]

		roi,pil_img = hist.vertical(roi,pil_img)

		h_roi = int(round(roi.shape[0]))
		w_roi = int(round(roi.shape[1]))

		#new,response,pil_img = window(pil_img,w_roi, h_roi , 3,20,h_roi , roi)
		
		#text_ext(response,new,i,pil_img) 
		text_ext(response,roi,i,pil_img)
		#cv2.imshow("roi",roi)
		#text_ext(response,i)
		#cv2.imshow("response",response)
		#cv2.waitKey(100000)
		if(i==len(stack_end)-1) :
			Li = hist.search_last(freq, stack_end[i],0) -4
			min_h = (stack_end[i]+13)
			roi = img_ori[Li:min_h, 0:w]
			pil_img=pil_img_ori[Li:min_h, 0:w]
			roi,pil_img = hist.vertical(roi,pil_img)
			h_roi = int(round(roi.shape[0]))
			w_roi = int(round(roi.shape[1]))

			#new,response,pil_img = window(pil_img,w_roi, h_roi , 3,20,h_roi , roi)
			#text_ext(response,new,i+1,pil_img)
			#text_ext(response,new,i+1,pil_img) 
			text_ext(response,roi,i+1,pil_img)
			#cv2.imshow("response",response)
			#cv2.waitKey(100000)

		i+=1
	
	#recognizer_model.test(image_inf)
	print(time.time()-start_time)

import shutil
import os as os
from PIL import Image, ImageTk 
if(os.path.exists('./text2')) :
	shutil.rmtree('./text2')
	os.makedirs('./text2')
else :
	os.makedirs('./text2')
receive_input('sample/ktp_ori4.png')
#make_file()


# im = cv2.imread('./ktp_ori4.png')
# im =cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
# im = Image.fromarray(im)
# im = im.convert('L')
# im = np.asarray(im, dtype=np.uint8)
# im = im[np.newaxis, :]

# img = Image.open('./ktp_ori4.png')
# img = img.convert('L')
# img = np.asarray(img, dtype=np.uint8)
# img = img[np.newaxis, :]
# print(img)
# print(im)
# # img=Image.open('./ktp_ori4.png')
# # img=img.resize((500, 200),Image.ANTIALIAS)


# import tkinter as tk 
# root = tk.Tk()
# tkimage = ImageTk.PhotoImage(img)
# tk.Label(root, image=tkimage).pack()

# root.mainloop()


#cv2.waitKey(1000000)
