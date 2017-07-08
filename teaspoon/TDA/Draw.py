'''
Code for drawing various things.

'''



import matplotlib.pyplot as plt


def drawDgm(D,max=None,epsilon = .5):

    if not max:
        max = D.max()+epsilon

    plt.plot([0,max],[0,max])
    
    plt.scatter(D[:,0],D[:,1])
    plt.axis([0,max,0,max])


def drawPtCloud(P):
	plt.scatter(P[:,0],P[:,1])