__author__ = 'moonkey'

import os
import numpy as np
from PIL import Image
from collections import Counter
import pickle as cPickle
import random, math
from recognizer.src.data_util.bucketdata import BucketData
import cv2



class DataGen(object):
    GO = 1
    EOS = 2

    def __init__(self,
                 data_root, annotation_fn,
                 evaluate = False,
                 valid_target_len = float('inf'),
                 img_width_range = (12, 320),
                 word_len = 30):
        """
        :param data_root:
        :param annotation_fn:
        :param lexicon_fn:
        :param img_width_range: only needed for training set
        :return:
        """

        img_height = 32
        self.data_root = data_root
        self.size=0
        if os.path.exists(annotation_fn):
            self.annotation_path = annotation_fn
        else:
            self.annotation_path = os.path.join(data_root, annotation_fn)

        if evaluate:
            self.bucket_specs = [(int(math.floor(64 / 4)), int(word_len + 2)), (int(math.floor(108 / 4)), int(word_len + 2)),
                                 (int(math.floor(140 / 4)), int(word_len + 2)), (int(math.floor(256 / 4)), int(word_len + 2)),
                                 (int(math.floor(img_width_range[1] / 4)), int(word_len + 2))]
        else:
            self.bucket_specs = [(int(64 / 4), 9 + 2), (int(108 / 4), 15 + 2),
                             (int(140 / 4), 17 + 2), (int(256 / 4), 20 + 2),
                             (int(math.ceil(img_width_range[1] / 4)), word_len + 2)]

        self.bucket_min_width, self.bucket_max_width = img_width_range
        self.image_height = img_height
        self.valid_target_len = valid_target_len

        self.bucket_data = {i: BucketData()
                            for i in range(self.bucket_max_width + 1)}
        self.data=[]



    def clear(self):
        self.bucket_data = {i: BucketData()
                            for i in range(self.bucket_max_width + 1)}
        return self.bucket_data
    def get_size(self):
        # with open(self.annotation_path, 'r') as ann_file:
        #     return len(ann_file.readlines())

        return self.size

    def gen2(self, batch_size):

        sz=0
        valid_target_len = self.valid_target_len
        with open(self.annotation_path, 'r') as ann_file:
            lines = ann_file.readlines()
            random.shuffle(lines)

            for l in lines:
                img_path, lex = l.strip().split()
                path = os.path.join(self.data_root, img_path)

                if(os.path.isfile(path)) :

                    sz+=1
                    try:
                        img_bw, word = self.read_data(img_path, lex)
                        if valid_target_len < float('inf'):
                            word = word[:valid_target_len + 1]
                        width = img_bw.shape[-1]

                        # TODO:resize if > 320
                        b_idx = min(width, self.bucket_max_width)
                        bs = self.bucket_data[b_idx].append(img_bw, word, os.path.join(self.data_root,img_path))
                        if bs >= batch_size:
                            b = self.bucket_data[b_idx].flush_out(
                                    self.bucket_specs,
                                    valid_target_length=valid_target_len,
                                    go_shift=1)
                            if b is not None:
                                yield b
                            else:
                                assert False, 'no valid bucket of width %d'%width
                    except IOError:
                        pass # ignore error images
                        #with open('error_img.txt', 'a') as ef:
                        #    ef.write(img_path + '\n')

            self.size=sz
        self.clear()
        #return self.bucket_data

    def gen(self, batch_size):
            #random.shuffle(self.data)
            valid_target_len = self.valid_target_len
            for i,l in enumerate(self.data):


                    try:
                            img_bw=l[0]
                            word =l[1]
                            img_path=l[2]
                            if valid_target_len < float('inf'):
                                word = word[:valid_target_len + 1]
                            width = len(img_bw[0])

                            # TODO:resize if > 320
                            b_idx = min(width, self.bucket_max_width)
                            bs = self.bucket_data[b_idx].append(img_bw, word, os.path.join(self.data_root,img_path))
                            if bs >= batch_size:
                                b = self.bucket_data[b_idx].flush_out(
                                        self.bucket_specs,
                                        valid_target_length=valid_target_len,
                                        go_shift=1)

                                #print(b)

                                if b is not None:
                                    yield b
                                else:

                                    assert False, 'no valid bucket of width %d'%width


                    except IOError:
                        pass # ignore error images
                        #with open('error_img.txt', 'a') as ef:
                        #    ef.write(img_path + '\n')

   
            self.clear()
    def prepare_test_data(self,img_list):
            self.data= []
            for img in img_list :
                img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

                img = Image.fromarray(img)






                w, h = img.size

                aspect_ratio = float(w) / float(h)
                if aspect_ratio < float(self.bucket_min_width) / self.image_height:
                    img = img.resize(
                        (self.bucket_min_width, self.image_height),
                        Image.ANTIALIAS)
                elif aspect_ratio > float(
                        self.bucket_max_width) / self.image_height:
                    img = img.resize(
                        (self.bucket_max_width, self.image_height),
                        Image.ANTIALIAS)
                elif h != self.image_height:
                    img = img.resize(
                        (int(aspect_ratio * self.image_height), self.image_height),
                        Image.ANTIALIAS)
                
                img = img.convert('L')
                img = np.asarray(img, dtype=np.uint8)
                img = img[np.newaxis, :]

                word= [1,50,50,50,2]
                img_path ="./test"
                self.data.append([img, word,img_path])

    def prepare_data(self):
          PIK = 'pickle.dat'
          if not os.path.exists(PIK):
                sz=0
                valid_target_len = self.valid_target_len
                with open(self.annotation_path, 'r') as ann_file:
                    lines = ann_file.readlines()
                    random.shuffle(lines)
                    for i,l in enumerate(lines):
                        img_path, lex = l.strip().split()
                        path = os.path.join(self.data_root, img_path)

                        if(os.path.isfile(path)) :

                                if(len(lex)<self.bucket_specs[-1][1] ) :
                                    if(lex=="STATUS_KAWIN") :
                                        lex="Status_Kawin"    
                                    img_bw, word = self.read_data(img_path, lex)
                                    if valid_target_len < float('inf'):
                                        word = word[:valid_target_len + 1]
                                    self.data.append([img_bw, word,img_path])
                                    sz+=1

                                    if(i%10000) :
                                        print("data:",i)

                self.size=sz 

                os.mknod(PIK)
                with open(PIK, "wb") as f:
                    cPickle.dump(self.data, f)
                    print("Create Done")
          else :
                with open(PIK, "rb") as f:
                    self.data= cPickle.load(f)
                for i,dt in enumerate(self.data) :

                    if(len(dt[1])>=31) :
                       self.data.pop(i)
                    
                self.size=len(self.data) 
                print("Load Data ",self.size)


    def read_data(self, img_path, lex):
        #print(img_path,lex)
        #print(len(lex))
        #print(self.bucket_specs[-1][1])
        assert 0 < len(lex) < self.bucket_specs[-1][1]

        # L = R * 299/1000 + G * 587/1000 + B * 114/1000

        with open(os.path.join(self.data_root, img_path), 'rb') as img_file:

            img = Image.open(img_file)



            w, h = img.size

            aspect_ratio = float(w) / float(h)
            if aspect_ratio < float(self.bucket_min_width) / self.image_height:
                img = img.resize(
                    (self.bucket_min_width, self.image_height),
                    Image.ANTIALIAS)
            elif aspect_ratio > float(
                    self.bucket_max_width) / self.image_height:
                img = img.resize(
                    (self.bucket_max_width, self.image_height),
                    Image.ANTIALIAS)
            elif h != self.image_height:
                img = img.resize(
                    (int(aspect_ratio * self.image_height), self.image_height),
                    Image.ANTIALIAS)

            img_bw = img.convert('L')
            img_bw = np.asarray(img_bw, dtype=np.uint8)
            img_bw = img_bw[np.newaxis, :]

        # 'a':97, '0':48

        word = [self.GO]
        for c in lex:
            assert 31 < ord(c) < 127
            #assert 96 < ord(c) < 123 or 47 < ord(c) < 58
            # word.append(
            #     ord(c) - 97 + 13 if ord(c) > 96 else ord(c) - 48 + 3)
            if (ord(c) > 31 and ord(c)<127):
                word.append(
                    ord(c) - 32 + 3  )

        #print(lex)
        word.append(self.EOS)
        #print(word)
        word = np.array(word, dtype=np.int32)
        #print(word)
        
        # word = np.array( [self.GO] +
        # [ord(c) - 97 + 13 if ord(c) > 96 else ord(c) - 48 + 3
        # for c in lex] + [self.EOS], dtype=np.int32)

        return img_bw, word


def test_gen():
    print('testing gen_valid')
    # s_gen = EvalGen('../../data/evaluation_data/svt', 'test.txt')
    # s_gen = EvalGen('../../data/evaluation_data/iiit5k', 'test.txt')
    # s_gen = EvalGen('../../data/evaluation_data/icdar03', 'test.txt')
    s_gen = EvalGen('../../data/evaluation_data/icdar13', 'test.txt')
    count = 0
    for batch in s_gen.gen(1):
        count += 1
        print(str(batch['bucket_id']) + ' ' + str(batch['data'].shape[2:]))
        assert batch['data'].shape[2] == img_height
    print(count)


if __name__ == '__main__':
    test_gen()
