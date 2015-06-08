package main;

import mdp.HomeHeatingMDP;

import org.jblas.DoubleMatrix;

import com.sun.corba.se.impl.orb.ParserTable.TestBadServerIdHandler;

import util.FileHandler;
import cov.*;
import gp.GP;

public class Main {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		SquaredExponential c1 = new SquaredExponential();
		Periodic c2 = new Periodic();
		Matern m = new Matern();
		Multiplicative mult = new Multiplicative(c1, c2);
		Additive a1 = new Additive(mult, m);
		
		int noData = 100;
		double stepsize = 0.010416666666;
		double[] dataX = new double[noData];
		double[] dataY =  new double[noData];
		for(int i=0; i< dataX.length; i++){
			dataX[i] = i*stepsize;
		}
		for(int i=0; i< 30; i++){
			dataY[i] = 10;
		}
		for(int i=30; i< 60; i++){
			dataY[i] = 30;
		}
		for(int i=60; i< 100; i++){
			dataY[i] = 20;
		}
		double[] dataP = {2,1.5,1,1.2,0.2,0.2};
		double[] dataTest = new double[noData];// = {11,12,13,14,15,16};
		for(int i = 0; i < dataTest.length; i++){
			dataTest[i] = i*stepsize + noData*stepsize;
		}
		double nl = 0.001;		
		DoubleMatrix X = new DoubleMatrix(dataX);
		DoubleMatrix Y = new DoubleMatrix(dataY);
		DoubleMatrix P = new DoubleMatrix(dataP);
		DoubleMatrix testIn =  new DoubleMatrix(dataTest);
		DoubleMatrix fakeP = new DoubleMatrix(new double[] {1,0.5});
		CovarianceFunction cf = a1;

		
		GP gp = new GP(X, Y, testIn, P, cf, nl);
		gp.setup();
		DoubleMatrix samples = gp.generateSamples(X, P, nl, cf);
		GP gptemp = new GP(X,Y,testIn,P,new Matern(),nl);
		gptemp.setup();
		DoubleMatrix tempSamples = gptemp.generateSamples(X, P, nl, new Matern());
//		HomeHeatingMDP hhm = new HomeHeatingMDP(samples.add(20), gp.getTrainCov(),tempSamples.add(10),gptemp.getTrainCov(), 5, 20);
//		hhm.work();
//		hhm.printOptPolicy();
//		hhm.printPrices();
//		hhm.printExtTemps();
//		hhm.printIntTemps();
		printMatrix(samples);
		printMatrix(tempSamples);
//		printMatrix(gp.getPredMean());
//		printMatrix(samples);
//		printMatrix(gp.getTrainCov());
//		printMatrix(gp.getTestCov());
//		printMatrix(gp.getTestTrainCov());

//		System.out.println("parmas: ");
//		samples.print();
//		gp.setup();
//		double noise = 0.5;
//		double cumulativeU = 0;
//		double currentTemp = 24;
//		int steps = 10;
//		int ibefore = 0;
//		double[] xTrain = new double[steps];
//		for(int i = 0; i < steps; i++){
//			xTrain[i] = i;
//		}
//		samples.print();
//		System.out.println(samples.mean());
//		DoubleMatrix xTrainM = new DoubleMatrix(xTrain);
//		for(int i = steps; i < 100; i+= steps){
//			double[] xTest = new double[steps];
//			for(int o = 0; o < steps; o++){
//				xTest[o] = o + ibefore;
//			}
//			DoubleMatrix xTestM = new DoubleMatrix(xTest);
//			DoubleMatrix yTrainM = subVector(0, i, samples);
//			GP newGP = new GP(xTrainM,yTrainM,xTestM,P,c1,noise);
//			newGP.setup();
//			DoubleMatrix predMean = newGP.getPredMean();
//			HomeHeatingMDP testmdp = new HomeHeatingMDP(predMean,newGP.getTestCov(),1,steps);
//			testmdp.work();
//			// heat according to policy and update cumulative utility
//			for(int o = ibefore; o < ibefore + stepsize; o++){
//				System.out.println("pts: " + testmdp.priceToState(predMean.get(o-ibefore)) + " ets: " + testmdp.externalTempToState(predMean.get(o-ibefore)) + " its: " + testmdp.internalTempToState(currentTemp));
//				int action = testmdp.getOptPolicy()[o-ibefore][testmdp.priceToState(predMean.get(o-ibefore))][testmdp.internalTempToState(currentTemp)][testmdp.externalTempToState(predMean.get(o-ibefore))];
//				currentTemp = testmdp.updateInternalTemperature(currentTemp,samples.get(o) , action);
//				System.out.println(currentTemp);
//				cumulativeU += testmdp.rewards(currentTemp, action, samples.get(o));
//			}
//			
//			xTrainM = DoubleMatrix.concatVertically(xTrainM,xTestM);
//			ibefore = i;
//		}
//		System.out.println(cumulativeU);
//		DoubleMatrix np = gp.minimize(fakeP, -20, X, samples);
//		np.print();
//
//		gp.getPredMean().print();
//		gp.getTestCov().print();
//		int num = 10;
//		
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
				System.out.print(m.get(i,j)+ " ");
			}
			System.out.print("; ");

		}
		System.out.println("]");
	}
}
