# @package teaspoon.Misc
# Some helper code used throughout the package.



## Prints a nice output from the time.time() information.
# @input seconds 
#	Output from time.time() or, more commonly, endTime-startTime.
def printPrettyTime(seconds):
    numHours, numMinutes = divmod(seconds, 60 ** 2)
    numMinutes, numSeconds = divmod(numMinutes, 60)

    output = str(int(numHours)) + ' hrs, ' + str(int(numMinutes)) + ' min, ' + str(int(numSeconds)) + ' secs'
    return output