# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 10:50:20 2018

@author: aldi242
"""

from shutil import copyfile

fileTest = open("testlist03.txt", "r") 
testFiles = fileTest.readlines() 

fileTrain = open("trainlist03.txt", "r") 
trainFiles = fileTrain.readlines() 

for filename in trainFiles:
    
    fileSplit = filename.split(" ")[0]
    sourceFolder = "UCF-101/"
    destFolder = "Eksperimen/Split3/TrainData/"

    copyfile(sourceFolder + fileSplit, destFolder + fileSplit)

for filename in testFiles:
    
    sourceFolder = "UCF-101/"
    destFolder = "Eksperimen/Split3/TestData/"

    copyfile(sourceFolder + filename[:-2], destFolder + filename[:-2])
    
