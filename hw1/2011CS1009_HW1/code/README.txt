scripts named graphplot.m, graphplot1.m, graphplot2.m are used to generate various plot. 
These scripts may in turn call other functions.

graphplot.m:- plot the graph of lambda vs MSE(training/test) for different training set fractions 
graphplot1.m:- plot the graph of fraction vs mean average mean squared testing error and 
		graph of fraction vs lamdba( for which squared error is minimum )
graphplot2.m:- plot the graph of Predicted values vs Actual values of classes for fraction = 0.5 and 
		lambda = 0.01

removeAttr.m:- this scripts check the effect of removing 3 least contributing attributes on mean square error

mylinridgereg.m, mylinridgeregeval.m and meansquarederr.m are the functions implemented as per mentioned in the 
assignment

linridgereg.m:- this function returns the mean square error for various combinations of lambda and fraction splits.