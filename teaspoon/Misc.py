# @package teaspoon.Misc
# Some helper code used throughout the package.

import os


## Prints a nice output from the time.time() information.
# @input seconds 
#	Output from time.time() or, more commonly, endTime-startTime.
def printPrettyTime(seconds):
    numHours, numMinutes = divmod(seconds, 60 ** 2)
    numMinutes, numSeconds = divmod(numMinutes, 60)

    output = str(int(numHours)) + ' hrs, ' + str(int(numMinutes)) + ' min, ' + str(int(numSeconds)) + ' secs'
    return output


## Pulls all the filenames in a folder as a list of strings
# @input folderPath
# 	Uses the current folder unless otherwise specified
# @input filetype
# 	Only pulls files with that string at the end of the filename. If you want all files, pass ```filetype = ''```. 
# @return
# 	A list of filenames as strings
#
def getAllFilenames(folderPath=os.getcwd(), filetype='.mat'):
    cwd = folderPath
    filenames = os.listdir(cwd)
    if len(filetype) >0:
        filenames = [f for f in filenames if f[-len(filetype):] == filetype]
    return filenames