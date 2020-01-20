#!/usr/bin/env python

##
# Massimiliano Patacchiola, Plymouth University 2016
# website: http://mpatacchiola.github.io/
# email: massimiliano.patacchiola@plymouth.ac.uk
# Python code for information retrieval from the Annotated Facial Landmarks in the Wild (AFLW) dataset.
# In this example the faces are isolated and saved in a specified output folder.
# Some information (roll, pitch, yaw) are returned, they can be used to filter the images.
# This code requires OpenCV and Numpy. You can easily bypass the OpenCV calls if you want to use
# a different library. In order to use the code you have to unzip the images and store them in
# the directory "flickr" mantaining the original folders name (0, 2, 3).
#
# The following are the database properties available (last updated version 2012-11-28):
#
# databases: db_id, path, description
# faceellipse: face_id, x, y, ra, rb, theta, annot_type_id, upsidedown
# faceimages: image_id, db_id, file_id, filepath, bw, widht, height
# facemetadata: face_id, sex, occluded, glasses, bw, annot_type_id
# facepose: face_id, roll, pitch, yaw, annot_type_id
# facerect: face_id, x, y, w, h, annot_type_id
# faces: face_id, file_id, db_id
# featurecoords: face_id, feature_id, x, y
# featurecoordtype: feature_id, descr, code, x, y, z

import sqlite3
import cv2
import scipy
import scipy.misc
import os.path
import numpy as np
import pickle
import collections
import matplotlib.pyplot as plt
from collections import defaultdict
import skimage
import skimage.feature
from skimage.transform import integral_image
from skimage.feature import draw_multiblock_lbp

alfw =  '/home/library03/data/aflw/data'

#Change this paths according to your directories
images_path = os.path.join(alfw, "flickr")
storing_path = "./output/"

Face = collections.namedtuple('Face', ('genre', 'roll', 'pitch', 'yaw', 'feats', 'facex', 'facey', 'facew', 'faceh', 'filepath', 'shape', 'hog'))

lmnames = [
    "LeftBrowLeftCorner",
    "LeftBrowCenter",
    "LeftBrowRightCorner",
    "RightBrowLeftCorner",
    "RightBrowCenter",
    "RightBrowRightCorner",
    "LeftEyeLeftCorner",
    "LeftEyeCenter",
    "LeftEyeRightCorner",
    "RightEyeLeftCorner",
    "RightEyeCenter",
    "RightEyeRightCorner",
    "LeftEar",
    "NoseLeft",
    "NoseCenter",
    "NoseRight",
    "RightEar",
    "MouthLeftCorner",
    "MouthCenter",
    "MouthRightCorner",
    "ChinCenter",
    "LeftTemple",
    "RightTemple",
    "LeftNoseNostril",
    "RightNoseNostril",
    "MouthCenterUpperLipOuterEdge",
    "MouthCenterLowerLipOuterEdge",
    "HeadCenter"]


goodmarksnames = [
    "MouthLeftCorner",
    "MouthCenter",
    "MouthRightCorner",
    "LeftEyeLeftCorner",
    "LeftEyeCenter",
    "LeftEyeRightCorner",
    "RightEyeLeftCorner",
    "RightEyeCenter",
    "RightEyeRightCorner",
    "NoseLeft",
    "NoseRight",
    "NoseCenter"
]

idx = { k : i for i,k in enumerate(lmnames) }

size = 128
smallsize = 32

def main():

    #Image counter

    #Open the sqlite database
    conn = sqlite3.connect(os.path.join(alfw, 'aflw.sqlite'))
    c = conn.cursor()

    #Creating the query string for retriving: roll, pitch, yaw and faces position
    #Change it according to what you want to retrieve
    select_string = "faceimages.filepath, faces.face_id, facepose.roll, facepose.pitch, facepose.yaw, facerect.x, facerect.y, facerect.w, facerect.h, facemetadata.sex"
    from_string = "faceimages, faces, facepose, facerect, facemetadata"
    where_string = "faces.face_id = facepose.face_id and faces.file_id = faceimages.file_id and faces.face_id = facerect.face_id and facemetadata.face_id == faces.face_id"
    query_string = "SELECT " + select_string + " FROM " + from_string + " WHERE " + where_string

    select_string = "faces.face_id, featurecoords.feature_id, featurecoords.x, featurecoords.y"
    from_string = "faces, featurecoords"
    where_string = "faces.face_id = featurecoords.face_id"
    query_string1 = "SELECT " + select_string + " FROM " + from_string + " WHERE " + where_string
    
    #It iterates through the rows returned from the query
    fidm = 0
    imgs = []
    imgs2 = []
    imgsSmall = []
    ind = {}
    md = []
    featn = len(lmnames)

    for counter, row in enumerate(c.execute(query_string)):
        #Using our specific query_string, the "row" variable will contain:
        # row[0] = image path
        # row[1] = face id
        # row[2] = roll
        # row[3] = pitch
        # row[4] = yaw
        # row[5] = face coord x
        # row[6] = face coord y
        # row[7] = face width
        # row[8] = face heigh

        #Creating the full path names for input and output
        input_path = images_path + '/' + str(row[0])
        output_path = storing_path + str(row[0])

        #If the file exist then open it       
        if(os.path.isfile(input_path)  == True):
            image = cv2.imread(input_path)
            #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #load the colour version

            #Image dimensions
            image_h, image_w, _ = image.shape
            #Roll, pitch and yaw
            roll   = row[2]
            pitch  = row[3]
            yaw    = row[4]
            #Face rectangle coords
            face_x = row[5]
            face_y = row[6]
            face_w = row[7]
            face_h = row[8]
            genre = row[9]

            overflow = face_x < 0 or face_x + face_w > image_w or face_y < 0 or face_y + face_h > image_h
            if not overflow :
                ind[int(row[1])] = len(md)
                #Error correction
                if(face_x < 0): face_x = 0
                if(face_y < 0): face_y = 0
                if(face_w > image_w): 
                    face_w = image_w
                    face_h = image_w
                if(face_h > image_h): 
                    face_h = image_h
                    face_w = image_h

                #Crop the face from the image
                image_cropped = np.copy(image[face_y:face_y+face_h, face_x:face_x+face_w])
                #Uncomment the lines below if you want to rescale the image to a particular size

                
                imager = scipy.misc.imresize(image_cropped, (size, size), interp='bicubic')
                imager1 = scipy.misc.imresize(imager, (smallsize, smallsize), interp='bicubic')
                imager2 = scipy.misc.imresize(imager1, (size, size), interp='bicubic')
                imgs.append(imager)
                imgs2.append(imager2)
                imgsSmall.append(imager1)
                hog  = skimage.feature.hog(np.mean(imager, axis=2), pixels_per_cell=(8, 8), cells_per_block=(2, 2), transform_sqrt=True)
                if False:
                    fig = plt.figure()
                    fig.add_subplot(1,3,1)
                    plt.imshow(imager[:,:,::-1])
                    fig.add_subplot(1,3,2)
                    plt.imshow(imager1[:,:,::-1])
                    fig.add_subplot(1,3,3)
                    plt.imshow(imager2[:,:,::-1])
                    plt.show()


                
                assert(abs(float(face_h)/face_w - 1.) < 0.001)
                md.append(Face(genre, roll, pitch, yaw,                       
                               np.ones((featn, 2)) * -10, # feats
                               face_x,
                               face_y,
                               face_w,
                               face_h,
                               input_path,
                               image.shape,
                               hog))



                #image_rescaled = cv2.resize(image_cropped, (to_size,to_size), interpolation = cv2.INTER_)
                #Uncomment the line below if you want to use adaptive histogram normalisation
                #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(5,5))
                #image_normalised = clahe.apply(image_rescaled)
                #Save the image
                #change "image_cropped" with the last uncommented variable name above
                #cv2.imwrite(output_path, image_cropped)

                #Printing the information
                if False :
                    print "Counter: " + str(counter)
                    print "iPath:    " + input_path
                    print "oPath:    " + output_path
                    print "Roll:    " + str(roll)
                    print "Pitch:   " + str(pitch)
                    print "Yaw:     " + str(yaw)
                    print "x:       " + str(face_x)
                    print "y:       " + str(face_y)
                    print "w:       " + str(face_w)
                    print "h:       " + str(face_h)
                    print "genre:" + str(row[9])
                    print ""

                if counter % 1000 == 0 :
                    print counter
                if counter > 1000000:
                    break

            #if the file does not exits it return an exception
        else:
            raise ValueError('Error: I cannot find the file specified: ' + str(input_path))
    print fidm
    #Once finished the iteration it closes the database
    print 'len=', len(md)
    a = np.asarray(imgs)
    cf = 0
    pbkey = defaultdict(lambda:0)
    for i, row in enumerate(c.execute(query_string1)):
        fid = row[0]
        featid = int(row[1])
        try :
            ix = ind[fid]
            sc = float(md[ix].facew) / size
            featx = float(row[2] - md[ix].facex) / sc
            featy = float(row[3] - md[ix].facey) / sc
            md[ix].feats[featid-1,0] = featx
            md[ix].feats[featid-1,1] = featy
            cf += 1
            if False :
                print '\nix=', ix, ' fid= ' , fid, ' featx=', featx, ' featy=', featy
                print ' facex ' , md[ix].facex, '\n'
                print ' facey ' , md[ix].facey, '\n'
                print ' facew ' , md[ix].facew, '\n'
                print ' faceh ' , md[ix].faceh, '\n'
                print ' featid = ', featid, '\n'
                print ' shape = ', md[ix].shape, '\n'
                print ' rox ', row[2], ',', row[3]
                print np.sum(md[ix].feats > 0)
            if False and np.sum(md[ix].feats > 0) >= 40 :
                fig = plt.figure()
                fig.add_subplot(1,2,1)
                plt.imshow(a[ix, :,:,::-1])
                for label, x, y in zip(lmnames,
                                       md[ix].feats[:,0],
                                       md[ix].feats[:,1]) :
                    plt.annotate(
                        label, 
                        xy = (x, y),
                        xytext = (-20, 20),
                        textcoords = 'offset points', ha = 'right', va = 'bottom',
                        bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
                        arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))


                if True :
                    fig.add_subplot(1,2,2)
                    image = cv2.imread(md[ix].filepath)
                    plt.imshow(image[:,:,::-1])

                    #plt.plot(md[ix].feats[:,0], md[ix].feats[:,1], 'o')
                    for label, x, y in zip([ lmnames[featid-1] ],
                                           [row[2]],
                                           [row[3]]) :
                        plt.annotate(
                            label, 
                            xy = (x, y),
                            xytext = (-20, 20),
                            textcoords = 'offset points', ha = 'right', va = 'bottom',
                            bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
                            arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
                    plt.show()

            
        except ZeroDivisionError :
            print 'fid ', fid , ' ix ', ix, ' face_w = 0'
        except KeyError :
            print 'pb key fid ', fid
            pbkey[fid] += 1
        if i > 1000000 :
            break
    print cf
    print 'len(d) ', len(pbkey)
    print 'len(ind) ', len(ind)
    c.close()
    #print a.shape
    np.save(alfw + '/aflw_imgs.npy', a)
    np.save(alfw + '/aflw_imgs2.npy', np.asarray(imgs2))
    np.save(alfw + '/aflw_imgsSmall.npy', np.asarray(imgsSmall))
    #print md
    pickle.dump(md,  open(alfw + "/md.pckl", "wb" ))


def load(n = 99999) :
    ims = np.load(alfw + '/aflw_imgs.npy')
    ims2 = np.load(alfw + '/aflw_imgs2.npy')
    imsSmall = np.load(alfw + '/aflw_imgsSmall.npy')
    #print md
    md = pickle.load(open(alfw + "/md.pckl", "rb" ))
    return ims[0:n, :, :].astype(np.float32), md[0:n], ims2[0:n, :, : ].astype(np.float32), imsSmall[0:n, :, : ].astype(np.float32)

def draw(im, p, ff = None) :
    plt.imshow(im) #[:, :, ::-1])
    gg = p[:,0] >= 0
    p = p[gg]
    #print md[ix].feats
    plt.scatter(x=p[:,0], y=p[:,1], c='r', s=40)
    nm = goodmarksnames
    for i in range(0, p.shape[0]) :
        label = str(i).zfill(2) #nm[i]
        y,x = p[i,:]
        plt.annotate(
            label, #''.join([c for c in label if c <= 'Z']), 
            xy = (y,x),
            xytext = (-2, 2),
            textcoords = 'offset points',
            ha = 'right', va = 'bottom',
            bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
            arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))


    
    #plt.plot(md[ix].feats[:,0], md[ix].feats[:,1], 'o')
    if ff is not None:
        plt.savefig(ff)
    else :
        plt.show()
    plt.close()
    
def show(im, md, ix) :
    plt.imshow(im[ix, :, :, ::-1])
    #print md[ix].feats
    plt.scatter(x=md[ix].feats[:,0], y=md[ix].feats[:,1], c='r', s=40)
    #plt.plot(md[ix].feats[:,0], md[ix].feats[:,1], 'o')
    plt.show()

def show1(im, md) :
    plt.imshow(im[:, :, ::-1])
    plt.scatter(x=md[:,0], y=md[:,1], c='r', s=40)
    #plt.plot(md[ix].feats[:,0], md[ix].feats[:,1], 'o')
    plt.show()

def split(im, feats, hog, mmm, im2, md, small) :
    n = im.shape[0] * 9 / 10
    return im[0:n,:,:,:], feats[0:n, :, :], hog[0:n, :], mmm[0:n, :, :], im[n:,:,:,:], feats[n:, :, :], hog[n:,:], mmm[n:,:,:], im2[0:n, : :], im2[n:, :, :], md[0:n], md[n:],  small[0:n, : :], small[n:, :, :]

def prepare(n = 99999, onlyvisible = False) :
    print 'loading..'
    im, md, im2, small = load(n)
    print 'im.shape: ', im.shape
    feats = np.asarray([f.feats for f in md])
    hogs = np.asarray([f.hog for f in md])
    print 'feats.shape:', feats.shape
    print 'hogs.shape:', hogs.shape
    idx = { k : i for i, k in enumerate(lmnames)}
    lmidx = np.asarray([ l in goodmarksnames for l in lmnames])
    goodmarks = np.asarray([ feats[:,idx[mn],:] >= 0 for mn in goodmarksnames])
    'goodmarks ', goodmarks.shape
    mask = np.logical_not(np.sum(np.logical_not(goodmarks), axis=(0, 2))) # since not(not(a) or not(b)) = a and b
    #print 'assert ', np.sum(mask - np.prod(goodmarks,  axis=(0, 2)))
    #print 'mask ', mask.shape
    print np.sum(mask), np.prod(mask.shape)
    mmm = goodmarks.transpose((1,0,2))
    #print 'mmm ', mmm.shape
    #print np.sum(mmm)
    if onlyvisible:
        md = [ e for i, e in enumerate(md) if mask[i] == True]
        rfeats = feats[mask, :, :][:,lmidx,:]
        rhogs = hogs[mask, :]
        rim = im[mask, :, :]
        rim2 = im2[mask, :, :]
        rimsmall = small[mask, :, :]
        mmm = mmm[mask,:,:]        
    else :
        rfeats = feats[:, :, :][:,lmidx,:]
        rhogs = hogs
        rim = im
        rim2 = im2
        rimsmall = small
        mmm = mmm
    #print 'rim.shape: ', rim.shape
    #print 'rfeats.shape:', rfeats.shape
    #print 'rhogs.shape:', rhogs.shape
    imtr, ytrain, Xtrain, mmm, imtst, ytest, Xtest, mmmt, im2tr, im2tst, mdtr, mdtst, smalltr, smalltst = split(rim, rfeats, rhogs, mmm, rim2, md, rimsmall)

    mean, std  = np.mean(Xtrain, axis=0), np.std(Xtrain, axis=0)
    #print Xtrain.shape
    #print mean.shape
    meany, stdy  = np.mean(ytrain, axis=0), np.std(ytrain, axis=0)
    print 'ytrain.shape ', ytrain.shape
    #print 'meany.shape ', meany.shape
    #print 'stdy.shape ', stdy.shape
    meani, stdi  = np.mean(imtr), np.std(imtst)
    #print imtr.shape
    #print meani.shape
    #print 'mmm ', mmm.shape
    def scale(im, y, x) :
        return im / 255, (y - size/2) / (size/2), (x - mean)/std
        return (im - meani) / stdi, (y - meany) / stdy, (x - mean)/std

    i1, y1, x1 = scale(imtr, ytrain, Xtrain)
    #print np.mean(x1, axis=0), np.std(x1, axis=0)
    #print np.mean(y1, axis=0), np.std(y1, axis=0)
    
    def unscale(y) :
        return y * (size/2) + size/2
        return (y * stdy) + meany
    #print np.sum(ytrain - unscale(y1))
        
    return imtr, ytrain, Xtrain, mmm, imtst, ytest, Xtest, mmmt, (mean, std, meany, stdy, meani, stdi), scale, unscale, { 'im2' : (im2tr, im2tst), 'md' : (mdtr, mdtst), 'imsmall' :  (smalltr, smalltst)  } 

if __name__ == "__main__":
    main()
    im, md, _, _ = load()
    print im.shape
    print len(md)
    #show(im, md, 10)
    
    """
    plt.hist(np.asarray([f.roll for f in md]))
    plt.show()
    plt.hist(np.asarray([f.pitch for f in md]))
    plt.show()
    plt.hist(np.asarray([f.yaw for f in md]))
    plt.show()
    """
    
    feats = np.asarray([f.feats for f in md])
    print feats.shape

    lmidx = np.asarray([ l in goodmarksnames for l in lmnames])
    print lmidx
    #goodmarksnames = lmnames
    idx = { k : i for i, k in enumerate(lmnames)}
    goodmarks = np.asarray([ feats[:,idx[mn],:] >= 0 for mn in goodmarksnames])
    mask = np.logical_not(np.sum(np.logical_not(goodmarks), axis=0)) # since not(not(a) or not(b)) = a and b
    #print 'goodmarks:', goodmarks.shape
    #print 'mask:', mask.shape
    #print 'feats:', feats.shape
    rfeats = feats[mask[:,0], :, :][:,lmidx,:]
    #print im.shape
    rim = im[mask[:,0], :, :]
    print rfeats.shape
    print rim.shape
    print 'goodmarks[mask]', goodmarks[:,mask[:,0],:].shape
    #plt.hist(np.asarray([np.sum(f.feats >= 0) / 2 for f in md]), bins = 100)
    #plt.show()
    
    show1(rim[1000, :,:,:], rfeats[1000, :, :])

    hog  = skimage.feature.hog(np.mean(rim[1000, :,:,:], axis=2), pixels_per_cell=(8, 8), cells_per_block=(2, 2), transform_sqrt=True)
    #print 'hog shape ', hog.shape
    lbp =  skimage.feature.local_binary_pattern(np.mean(rim[1000, :,:,:], axis=2), 6, 6)
    #print lbp.shape
    lbp_code =  skimage.feature.multiblock_lbp(integral_image(np.mean(rim[1000, :,:,:], axis=2)), 0, 0, 90, 90)
    #print lbp_code
