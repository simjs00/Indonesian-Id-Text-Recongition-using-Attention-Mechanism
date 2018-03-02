import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import glob
#import text_detector as dtc








def compute_hist(img=[],prj=0,h=0,w=0) :
		

	x=0
	y=0
	freq=[]
	if(prj==0) :
		x=h
		y=w
	else :
		x=w
		y=h
	freq = np.zeros((x))
	for x1 in range (x) :
			for y1 in range(y) :
				if(prj==0) :
					if(img[x1][y1] ==255) :
						freq[x1]+=1
				elif (prj==1)   :
					if(img[y1][x1] ==0) :
						freq[x1]+=1
				elif (prj==2)   :
					if(img[y1][x1] ==255) :
						freq[x1]+=1


	if(prj==0) :
			i=0
			ymax =np.max(freq)
			ymin =freq[0]
			ymean = np.mean(freq)

			while(ymin <=0) :
				ymin = freq[i]
				i+=1
			i=0
			while(i<len(freq)) :
				if( freq[i]< ymin  and freq[i] != 0) :

					ymin = freq[i]
				i+=1
			i=0


			h,w = img.shape[:2]
			while(i<len(freq)) :
				if(freq[i] < (ymean) ):

					freq[i] = 0
				i+=1



			hist = np.zeros((h,w))
			for x1 in range (x) :
					for y1 in range( int( round(freq[x1]))) :
								hist[x1][y1]=255
	elif (prj==1) :
		hist = np.ones((h,w))
		for x1 in range (x) :
					y1=h-1
					k=0
					while(k<= round(freq[x1])) :
						hist[y1][x1]=0
						y1-=1
						k+=1
	elif (prj==2) :
		hist = np.zeros((h,w))
		for x1 in range (x) :
					y1=h-1
					k=0
					while(k<= round(freq[x1])) :
						hist[y1][x1]=255
						y1-=1
						k+=1



	return freq,hist


def search_last(freq, start,h) :
	i=start
	while(freq[i] == h) :
		if(i== len(freq)-1 ) :
			break
		else :
			i+=1
	return i

def search_last1(freq, start,h) :
	i=start
	while(freq[i] != h) :
		if(i==0 ) :
			break
		else :
			i-=1
	return i





def threshold(freq , mean, h, freq1) :
	i=0
	while(i<len(freq)) :
		if(freq1[i]<mean) :
			freq[i]=h
		i+=1
	return freq



def segment(name,mean, img, freq) :
	stack = []
	h,w = img.shape[:2]
	min_val = 0
	max_val =0
	stack.append(0)
	i=1
	flag =0
	while(i<len(freq)) :
		if(freq[i]==h) :
				dist = min_val-i
				if(i-stack[len(stack)-1] >1) :
					print(i)
					stack.append(i)
				else :
					stack[len(stack)-1]=i


				cv2.rectangle(img,(min_val,0),(min_val+dist,h),(0,255,0),2)
				min_val = i

		i+=1
	print(len(stack))
	i=0
	while(i<len(stack)-1) :
		cv2.rectangle(img,(stack[i],0),(stack[i+1],h),(0,255,0),2)
		i+=1
	return img

def compute_mean(prj) :
	mean =0
	i=0
	while(i<len(prj)) :
		mean +=prj[i]
		i+=1

	mean = mean/len(prj)
	return mean
def pre(pth, j) :
	# Load the image
	kernel = np.ones((3,3), np.uint8)
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	img = cv2.imread(pth)
	h_or = img.shape[0]
	w_or = img.shape[1]

	w_or*=5
	h_or*=5
	img = cv2.resize(img,(w_or,h_or))
	img1 = cv2.resize(img,(w_or,h_or))

	# convert to grayscale


	gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


	gray = clahe.apply(gray)
	gray = cv2.bilateralFilter(gray,10,97,75*2)
	#gray = cv2.bilateralFilter(gray,10,97,75)

	gray=cv2.morphologyEx(gray,cv2.MORPH_OPEN,kernel )

	#gray = cv2.bilateralFilter(gray,10,97,75)


	# smooth the image to avoid noises



	# Apply adaptive threshold
	thresh = cv2.adaptiveThreshold(gray,255,1,1,11,2)




	#thresh = cv2.adaptiveThreshold(thresh,255,1,1,11,2)
	thresh_color = cv2.cvtColor(thresh,cv2.COLOR_GRAY2BGR)

	# apply some dilation and erosion to join the gaps


	thresh = cv2.erode(thresh,None,iterations = 1)
	#thresh = cv2.dilate(thresh,None,iterations =3)
	# Find the contours
	_,contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

	# For each contour, find the bounding rectangle and draw it
	for cnt in contours:
	    x,y,w,h = cv2.boundingRect(cnt)
	    #cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),4)
	    #cv2.rectangle(thresh_color,(x,y),(x+w,y+h),(0,255,0),2)

	x_proj_ori,histx =compute_hist(img=thresh,prj=1,h=h_or,w=w_or)
	y_proj_ori,histy =compute_hist(img=thresh,prj=0,h=h_or,w=w_or)

	x_proj_ori1 =compute_hist(img=thresh,prj=2,h=h_or,w=w_or)

	mean = compute_mean(x_proj_ori1)
	#print(compute_mean(x_proj_ori))
	#print(compute_mean(y_proj_ori))
	# pos = np.arange(len(x_proj))
	# width = 100    # gives histogram aspect to the bar diagram

	# ax = plt.axes()
	# ax.set_xticks(pos + (width / 2))
	# #ax.set_xticklabels(x_proj)
	# plt.bar(pos, x_proj, width, color='b')
	# plt.show()


	# pos = np.arange(len(y_proj))
	# width = 1    # gives histogram aspect to the bar diagram

	# ax = plt.axes()
	# ax.set_xticks(pos + (width / 2))
	# #ax.set_xticklabels(y_proj)
	# plt.bar(pos, y_proj, width, color='b')
	# plt.show()

	# Finally show the image
	#x_proj_ori = 	threshold(x_proj_ori , mean, h_or, x_proj_ori1)
	img2 = img1
	stack_start1,stack_end1 = segment2("ori",mean, img1, x_proj_ori)
	i=1

	while(i<len(stack_end1)) :
		kernel = np.ones((4,4), np.uint8)
		roi=[]
		roi1=[]

		roi=img[0:h_or-1,stack_end1[i-1]:stack_start1[i]]
		h_roi = int(round(roi.shape[0]))
		w_roi = int(round(roi.shape[1]))

		# roi = cv2.morphologyEx(roi,cv2.MORPH_OPEN,kernel )
		# h,w = roi.shape[:2]
		# x_proj,histx =compute_hist(img=roi,prj=1,h=h,w=w)
		# y_proj,histy =compute_hist(img=roi,prj=0,h=h,w=w)
		# r = img2[0:h_or-1,stack_end1[i-1]:stack_start1[i]]
		# segment2(str(i),mean, r, x_proj)
		# if(i==len(stack_end1)-1) :
		# 	Li = search_last(x_proj_ori, stack_end1[i],h_or)

		# 	roi2=thresh[0:h_or-1,stack_end1[i]:Li]
		# 	roi2 = cv2.morphologyEx(roi2,cv2.MORPH_OPEN,kernel )
		# 	h,w = roi2.shape[:2]
		# 	x_proj,histx =compute_hist(img=roi2,prj=1,h=h,w=w)
		# 	y_proj,histy =compute_hist(img=roi2,prj=0,h=h,w=w)

		# 	r = img2[0:h_or-1,stack_end1[i]:Li]
		# 	segment2(str(i),mean, r, x_proj)

		i+=1


	#cv2.imshow('segments2',img2)
	#img11 = segment(mean, img1, x_proj)
	# plt.imshow(img1)
	# plt.show()

	#sys.exit()



def segment2 (name,mean, img, freq) :

	stack_start = []
	stack_end= []
	i=0
	x1=-1
	x2=-1
	h,w = img.shape[:2]
	#print(freq)
	while(i<len(freq)) :
		if(freq[i]!=h) :
			if(x1==-1) :
				x1=i
				
			else :

				x2=i+1
				
			if(i==len(freq)-1 ) :
				if(x1!=-1 and x2!=-1) :
				
					stack_start.append(x1)
					stack_end.append(x2)


		else :
			# if(i==0) :
			# 	x1=i
			# 	x2=i

			if(x1!=-1 and x2!=-1) :
				
				stack_start.append(x1)
				stack_end.append(x2)


			x1=-1
			x2=-1

		i+=1


	# i=0
	# while(i<len(stack_end)) :
	# 	#cv2.rectangle(img,(stack_end[i-1],0),(stack_start[i],h),(0,255,0),2)
	# 	cv2.rectangle(img,(stack_start[i],0),(stack_end[i],h),(0,255,0),2)
	# 	# if(i==len(stack_end)-1) :
	# 	# 	Li = search_last(freq, stack_end[i],h)
	# 	# 	cv2.rectangle(img,(stack_end[i],0),(Li,h),(0,255,0),2)

	# 	i+=1


	return stack_start,stack_end


def compute_peak(responses) :
	i=0
	count=0
	while(i<len(responses)) :
		if(responses[i]>0) :	
			count+=1
		i+=1
	return count

def nms(histx, w) :
	i=1
	w = len(histx)
	histX_new = np.zeros(w)
	l =0 
	r =0
	while(i+1<w) :
		if(histx[i] > histx[i+1]) :
			if(histx[i] >= histx[i-1]) :
				histX_new[i]=histx[i]
				if(l==0) :
					l=i
				else :
					r=i 
		else :
			i=i+1
			while(i+1<w and histx[i]<=histx[i+1] ) :
				i=i+1
			if(i+1<w) :
				histX_new[i]=histx[i] 
				if(l==0) :
					l=i 
				else :
					r=i 

		i=i+2






	# mean_val = mean(histX_new, w)
	# print("mean :"+str(mean_val))
	# if(mean_val>85.0) :
	# 	return 1,l,r

	return l,r,histX_new

def calc_median(dist) :
	dist.sort()
	n=len(dist)/2
	n = int(round(n))+1
	return dist[n]


def segment_word(freq,roi) :
	h = roi.shape[0]
	w = roi.shape[1]

	i=0
	stack_start=[]
	stack_end=[]
	distance=[]
	temp_start=-1
	flag=0
	while(i<len(freq)) :
		if(freq[i]==0 ) :
			if(flag==0):
				temp_start=i
				flag=1
		else :
			print("n",freq[i])
			if(temp_start!=-1) :

				stack_start.append(temp_start)
				stack_end.append(i)

				distance.append((i)-temp_start)
				temp_start=-1
				flag=0
		i+=1
	
	i=0
	print(distance)
	med=calc_median(distance)
	print(med)
	s=[0]
	e=[]
	while(i<len(distance)) :
		if( round(distance[i]/med) >=5 ) :
			e.append(stack_start[i])
			s.append(stack_end[i])
		i+=1
	i=0
	while(i<len(e)) :
		test = roi[0:h,s[i]:e[i]]
		cv2.imshow("test",test)
		cv2.waitKey(100000)
		i+=1
	print(s)
	print(e)


def vertical2(roi,img,pil_img) :


	

	h_or = roi.shape[0]
	w_or = roi.shape[1]

	#cv2.imshow('img1',img)

	x_proj_ori,histx =compute_hist(img=roi,prj=2,h=h_or,w=w_or)
	
	l,r,x_proj_ori=nms(x_proj_ori, w_or)


	i=0
	while(i<len(x_proj_ori)) :
		if(x_proj_ori[i]==0) :
			j=0
			while(j<len(roi)) :
				roi[j][i]==0
				j+=1
		i+=1

	#l,r,x_proj_ori=nms(x_proj_ori, w_or)


	# if(l-1>=0 and r+1<h_or) :
	# 	roi = img[0:h_or , l:r]
	# else :
	if(l-5>=0) :
		img = img[0:h_or, l-5:r+5]
		#roi = roi[0:h_or, l-5:r+5]
		pil_img = pil_img[0:h_or, l-5:r+5]
	else :
		img = img[0:h_or , l:r+5]
		#roi = roi[0:h_or , l:r+5]
		pil_img = pil_img[0:h_or, l:r+5]

	#x_proj_ori,histx =compute_hist(img=roi,prj=2,h=h_or,w=w_or)
	#segment_word(x_proj_ori,roi)
	#cv2.imshow('img',img)
	#cv2.waitKey(1000000)
	return img,roi,pil_img

def vertical(roi,pil_img) :

	gray = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)

	h_or = roi.shape[0]
	w_or = roi.shape[1]


	gray = cv2.bilateralFilter(gray,10,97,75)
	gray = cv2.bilateralFilter(gray,10,97,75)


	thresh = cv2.adaptiveThreshold(gray,255,1,1,11,2)
	#thresh = cv2.dilate(thresh,None,iterations =2)
	x_proj_ori,histx =compute_hist(img=thresh,prj=2,h=h_or,w=w_or)


	l,r,x_proj_ori=nms(x_proj_ori, w_or)
	#l,r,x_proj_ori=nms(x_proj_ori, len(x_proj_ori))


	roi = roi[0:h_or , l-10:r+3]
	pil_img = pil_img[0:h_or , l-10:r+3]
	return roi, pil_img


def segment3 (name, img, freq,w_or) :


	stack_start = []
	stack_end= []
	i=0
	x1=-1
	x2=-1
	
	w = w_or
	#print(freq)
	while(i<len(freq)) :
		if(freq[i]==0) :
			if(x1==-1) :
				x1=i
			else :

				x2=i+1
		else :
			if(i==0) :
				x1=i
				x2=i

			if(x1!=-1 and x2!=-1) :
				stack_start.append(x1)
				stack_end.append(x2)
			x1=-1
			x2=-1

		i+=1


	i=1


	#cv2.imshow('img_ori',img)
	#cv2.waitKey(100000)
	return stack_start,stack_end,freq




def lrsm2(thresh,w_or,h_or, tr,c) :
	#cv2.imshow('before_thresh',thresh)
	i=0
	while(i<h_or) :
		#c = 1
		j=0
		while(j<w_or) :

			if (thresh[i][j]) == 255:
				if (j-c) <= tr:
					thresh[i][c:j] = 255
				c = j
			j+=1

		if (w_or - c) <= tr:
			thresh[i][c:w_or] = 255
		i+=1
	#cv2.imshow('after_thresh',thresh)
	#cv2.waitKey(100000)
	return thresh

def lrsm(thresh,w_or,h_or, tr,c) :
	#cv2.imshow('before_thresh',thresh)
	i=0
	while(i<h_or) :
		#c = 1
		j=0
		while(j<w_or) :

			if (thresh[i][j]) == 0:
				if (j-c) <= tr:
					thresh[i][c:j] = 0

				c = j
			j+=1

		if (w_or - c) <= tr:
			thresh[i][c:w_or] = 0
		i+=1
	#cv2.imshow('after_thresh',thresh)
	#cv2.waitKey(100000)
	return thresh



def horizontal(pth) :
	#kernel = np.ones((3,3), np.uint8)
	#clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	img = cv2.imread(pth)
	h_or = img.shape[0]
	w_or = img.shape[1]

	#w_or*=5
	#h_or*=5
	#img = cv2.resize(img,(w_or,h_or))
	#img1 = cv2.resize(img,(w_or,h_or))
	# convert to grayscale
	img1 = img
	gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

	# face_cascade = cv2.CascadeClassifier('haar/haarcascade_frontalface_alt.xml')
	# faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	# x1=0
	# for (x,y,w,h) in faces: 
	# 	 #cv2.rectangle(img1,(x,y),(x+w,y+h),(255,0,0),2)
	# 	 x1=x
	# if(x1!=0) :
	# 	gray=gray[0:h_or,0:x1+1]
	# 	w_or = x1+1
	


	#gray = clahe.apply(gray)
	gray = cv2.bilateralFilter(gray,10,97,75)
	gray = cv2.bilateralFilter(gray,10,97,75)
	#gray = cv2.bilateralFilter(gray,10,97,75)
	#gray = cv2.bilateralFilter(gray,10,97,75)

	#gray=cv2.morphologyEx(gray,cv2.MORPH_OPEN,kernel )

	#gray = cv2.bilateralFilter(gray,10,97,75)
	thresh = cv2.adaptiveThreshold(gray,255,1,1,11,2)
	#thresh = cv2.erode(thresh,None,iterations = 1)
	# smooth the image to avoid noises
	# Apply adaptive threshold
	#thresh = cv2.adaptiveThreshold(gray,255,1,1,11,2)
	#thresh=lrsm2(thresh,w_or,h_or,6,1)







	# x_proj_ori,histx =compute_hist(img=thresh,prj=1,h=h_or,w=w_or)
	# if(x1!=0) :
	# 	i=search_last1(x_proj_ori, w_or-1,h_or)
	# 	thresh=thresh[0:h_or,0:i+1]
	# 	w_or=i+1

	y_proj_ori,histy =compute_hist(img=thresh,prj=0,h=h_or,w=w_or)

	#mean = compute_mean(y_proj_ori)


	stack_start1,stack_end1,freq = segment3("ori", img1, y_proj_ori, w_or)
	return stack_start1, stack_end1,freq
	# plt.imshow(img1)
	# plt.show()	
	#cv2.imshow('gray',gray)
	# cv2.imshow('histy',histy)
	# #cv2.imshow('histx',histx)
	# cv2.imshow('thresh',thresh)
	# #cv2.imshow('segments',img1)
	# cv2.waitKey(1000000)

#re("./detect.png",0)
#pre("./sof_ori.png", 1) 
#horizontal('./ktp_ori4.png')

# images = glob.glob('./Scrapped_n_scaled/*png')
# for j, img in enumerate (images) :
# 		print(j)
# 		try:3
# 			horizontal(img)
# 		except Exception as e:
# 			print("failed")
# 			pass



#img = cv2.imread("./name.png")

#vertical(img ,1) 