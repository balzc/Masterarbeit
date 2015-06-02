package main;

import mdp.HomeHeatingMDP;

import org.jblas.DoubleMatrix;

import util.FileHandler;
import cov.*;
import gp.GP;

public class Main {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		SquaredExponential c1 = new SquaredExponential();
		Periodic c2 = new Periodic();
		int noData = 1000;
		double[] dataX = new double[noData];
		double[] dataY =  new double[noData];
		for(int i=0; i< dataX.length; i++){
			dataX[i] = i;
		}
		for(int i=0; i< dataY.length; i++){
			dataY[i] = i;
		}
		double[] dataP = {0.5,0.5};
		double[] dataTest = {11,12,13,14,15,16};
		double nl = 0;		
		DoubleMatrix X = new DoubleMatrix(dataX);
		DoubleMatrix Y = new DoubleMatrix(dataY);
		DoubleMatrix P = new DoubleMatrix(dataP);
		DoubleMatrix testIn = new DoubleMatrix(dataTest);
		DoubleMatrix fakeP = new DoubleMatrix(new double[] {1,0.5});
		GP gp = new GP(X, Y, testIn, P, c1, nl);
		DoubleMatrix samples = gp.generateSamples(X, P, nl, c1);
		
//		System.out.println("parmas: ");
//		samples.print();
		gp.setup();
		double noise = 0.5;
		double cumulativeU = 0;
		double currentTemp = 24;
		int stepsize = 10;
		int ibefore = 0;
		double[] xTrain = new double[stepsize];
		for(int i = 0; i < stepsize; i++){
			xTrain[i] = i;
		}
		DoubleMatrix xTrainM = new DoubleMatrix(xTrain);
		for(int i = stepsize; i < 1000; i+= stepsize){
			double[] xTest = new double[stepsize];
			for(int o = 0; o < stepsize; o++){
				xTest[o] = o + ibefore;
			}
			DoubleMatrix xTestM = new DoubleMatrix(xTest);
			DoubleMatrix yTrainM = subVector(0, i, samples);
			GP newGP = new GP(xTrainM,yTrainM,xTestM,P,c1,noise);
			newGP.setup();
			DoubleMatrix predMean = newGP.getPredMean();
			HomeHeatingMDP testmdp = new HomeHeatingMDP(predMean,newGP.getTestCov(),1,stepsize);
			testmdp.work();
			// heat according to policy and update cumulative utility
			for(int o = ibefore; o < ibefore + stepsize; o++){
				System.out.println("pts: " + testmdp.priceToState(predMean.get(o-ibefore)) + " ets: " + testmdp.externalTempToState(predMean.get(o-ibefore)) + " its: " + testmdp.internalTempToState(currentTemp));
				int action = testmdp.getOptPolicy()[o-ibefore][testmdp.priceToState(predMean.get(o-ibefore))][testmdp.internalTempToState(currentTemp)][testmdp.externalTempToState(predMean.get(o-ibefore))];
				currentTemp = testmdp.updateInternalTemperature(currentTemp,samples.get(o) , action);
				System.out.println(currentTemp);
				cumulativeU += testmdp.rewards(currentTemp, action, samples.get(o));
			}
			
			xTrainM = DoubleMatrix.concatVertically(xTrainM,xTestM);
			ibefore = i;
		}
		System.out.println(cumulativeU);
//		DoubleMatrix np = gp.minimize(fakeP, -20, X, samples);
//		np.print();
//
//		gp.getPredMean().print();
//		gp.getTestCov().print();
//		int num = 10;
		
//		testmdp.printOptPolicy();
//		testmdp.printQvals();
//		testmdp.printPrices();
	}
	public static DoubleMatrix subVector(int start, int end, DoubleMatrix vector){
		DoubleMatrix result = DoubleMatrix.zeros(end-start);
		for(int i = start; i < end; i++){
			result.put(i-start, vector.get(i));
		}
		return result;
	}
	
	public static void printMatrix(DoubleMatrix m){
		System.out.print("[");
		for(int i = 0; i< m.rows; i++){
			for(int j = 0; j< m.columns; j++){
				System.out.print(m.get(i,j));
			}
			System.out.print("; ");

		}
		System.out.println("]");
	}
}
