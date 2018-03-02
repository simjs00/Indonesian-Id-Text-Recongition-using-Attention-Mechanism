import test_model as model
import cv2
import numpy as np
import glob











def window1(w, h , stride,size, img,k) :


	i=0


	while(i<h) :
		print(i)
		j=0
		while(j<w) :
			max_i =i+size+1
			if(max_i > h) :
				max_i = h
			max_j =j+size+1
			if(max_j > w) :
				max_j = w

			roi = img[i:max_i,j:max_j]
			path = "./window/"+str(k)+"_"+str(i)+"_"+str(max_i)+"_"+str(j)+"_"+str(max_j)+".png"
			cv2.imwrite(path,roi)


			j+=stride
		i+=stride


	#cv2.imshow('response_map',response_map)
	#cv2.waitKey(10000000)
        





def window(w, h , stride,size, img) :
	response_score = np.zeros((h,w))
	response_map = np.zeros((h,w))

	i=0


	while(i<h) :
		print(i)
		j=0
		while(j<w) :



			max_i =i+size+1
			if(max_i > h) :
				max_i = h
			max_j =j+size+1
			if(max_j > w) :
				max_j = w

			roi = img[i:max_i,j:max_j]
			score =0
			if(len(roi[0]) >1) :
				label,score = model.predict_detector(roi)
				if(score<0.99)	:
					print(i,j,label, score)
				if(label=='-') :
					score *=-1



			#path = "./window/"+str(i)+"_"+str(max_i)+"_"+str(j)+"_"+str(max_j)+".png"
			#cv2.imwrite(path,roi)


			#print(score)
			k=i
			l=j
			while(k< max_i) :
				while(l< max_j) : 

					response_score[k][l]=score
					if(score >0) :
						response_map[k][l]=255
					elif(score <0 ) :
						response_map[k][l]=0

					l+=1

				k+=1

			j+=1
		i+=stride

	from time import gmtime, strftime
	path = str(strftime("%Y-%m-%d_%H:%M:%S", gmtime()))+".png"
	


	cv2.imwrite(path,response_map)
	# cv2.imshow('response_map',response_map)
	# response_map = cv2.normalize(response_score,  response_score, 0, 255, cv2.NORM_MINMAX)
	# cv2.imshow('response_map1',response_map)
	# cv2.waitKey(10000000)
        



import shutil
import os
if os.path.isdir('./window/'):
	shutil.rmtree('./window/')
os.mkdir("./window")

img = cv2.imread('ktp_ori2.png')

img= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

h = int(img.shape[0]) 
w = int(img.shape[1]) 
print(w,h)
#img = cv2.resize(img,(w,h))

window(w, h , 1,10, img)


# images=glob.glob('./Scrapped_n_scaled/*png')
# for j, img in enumerate (images) :
# 	img_colour = cv2.imread(img)
# 	img= cv2.cvtColor(img_colour,cv2.COLOR_BGR2GRAY)

# 	face_cascade = cv2.CascadeClassifier('haar/haarcascade_frontalface_alt.xml')
# 	faces = face_cascade.detectMultiScale(img, 1.3, 5)
# 	for (x,y,w,h) in faces:
# 		roi = img[y:y+h, x:x+w]
# 		window1(w,h,10,10,roi,j)

