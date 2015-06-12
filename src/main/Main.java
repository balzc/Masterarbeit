package main;

import mdp.HomeHeatingMDP;

import org.jblas.DoubleMatrix;

import com.apple.concurrent.Dispatch.Priority;
import com.sun.corba.se.impl.orb.ParserTable.TestBadServerIdHandler;

import util.FileHandler;
import cov.*;
import gp.GP;

public class Main {

	public static void main(String[] args) {
		test1();
		
		
//		gp.setup(samples);
//		printMatrix(X);
//		printMatrix(testIn);
//		printMatrix(gp.getTestTrainCov());
//		printMatrix(gp.getTestCov());
//		printMatrix(gp.getTrainCov());
//		printMatrix(gp.getPredMean());
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
	
	public static void test1(){
		String fileHandlePrices = "/users/balz/documents/workspace/masterarbeit/data/prices1.csv";
		String fileHandleTemps = "/users/balz/documents/workspace/masterarbeit/data/temps1.csv";

		SquaredExponential c1 = new SquaredExponential();
		Periodic c2 = new Periodic();
		Matern m = new Matern();
		Multiplicative mult = new Multiplicative(c1, c2);
		Additive a1 = new Additive(mult, m);
		
		int noData = 960;
		double stepsize = 0.010416666666;
		double[] dataX = new double[noData];
		double[] dataY =  new double[noData];
		for(int i=0; i< dataX.length; i++){
			dataX[i] = i*stepsize;
		}
		for(int i=0; i< noData; i++){
			dataY[i] = 10;
		}
	
		double[] dataP = {2,1.5,1,5,0.2,0.2};//{2,1.5,1,1.2,0.2,0.2}
		double[] dataTest = new double[noData];// = {11,12,13,14,15,16};
		for(int i = 0; i < dataTest.length/10; i++){
			dataTest[i] = i*stepsize + noData*stepsize;
		}
		double nl = 0.1;		
		DoubleMatrix priceSamples = FileHandler.csvToMatrix(fileHandlePrices).add(20);
		DoubleMatrix tempSamples = FileHandler.csvToMatrix(fileHandleTemps);
		DoubleMatrix P = new DoubleMatrix(dataP);
		CovarianceFunction cf = a1;
		int runs = 1;
		double cumulativeU = 0;
		double currentTemp = 16;
		int steps = 96;
		int ibefore = 0;
		double[] temperatures = new double[runs*steps];
		int[] actions = new int[runs*steps];

		for(int i = 0; i < runs; i++){
			double[] xTrain = new double[steps];
			for(int o = 0; o < steps; o++){
				xTrain[o] = o*stepsize+ibefore*stepsize;
			}
			DoubleMatrix xTrainM = new DoubleMatrix(xTrain);
			double[] xTest = new double[steps];
			for(int o = 0; o < steps; o++){
				xTest[o] = o*stepsize + (ibefore+steps)*stepsize;
			}
			DoubleMatrix xTestM = new DoubleMatrix(xTest);
			DoubleMatrix yTrainMPrices = subVector(ibefore, ibefore+steps, priceSamples);
			DoubleMatrix yTrainMTemps = subVector(ibefore, ibefore+steps, tempSamples);

			GP priceGP = new GP(xTrainM,xTestM,P,cf,nl);
			priceGP.setup(yTrainMPrices);
			GP tempGP = new GP(xTrainM,xTestM,P,cf,nl);
			tempGP.setup(yTrainMTemps);
			DoubleMatrix predMeanPrices = priceGP.getPredMean().add(20);
			DoubleMatrix predMeanTemps = tempGP.getPredMean();

			HomeHeatingMDP testmdp = new HomeHeatingMDP(predMeanPrices,priceGP.getTestCov(),predMeanTemps, tempGP.getTestCov(), 5,steps);

			testmdp.work();
			// heat according to policy and update cumulative utility
			int tmp = ibefore+steps;
			for(int o = tmp; o < ibefore + 2*steps; o++){
				System.out.println(o + " " + tmp + " " + testmdp.priceToState(predMeanPrices.get(o-tmp))+ " " + testmdp.externalTempToState(predMeanTemps.get(o-tmp)) + " "+testmdp.internalTempToState(currentTemp) + " " + currentTemp);
				int action = testmdp.getOptPolicy()[o-tmp][testmdp.priceToState(predMeanPrices.get(o-tmp))][testmdp.internalTempToState(currentTemp)][testmdp.externalTempToState(predMeanTemps.get(o-tmp))];
				currentTemp = testmdp.updateInternalTemperature(currentTemp,priceSamples.get(o) , action);
				cumulativeU += testmdp.rewards(currentTemp, action, priceSamples.get(o));
				temperatures[o-steps] = currentTemp;
				actions[o-steps] = action;
			}
//			testmdp.printOptPolicy();
			ibefore += steps;
		}
		System.out.print("[");
		for(int i = 0; i< temperatures.length; i++){
			System.out.print(temperatures[i] + ", ");

		}
		System.out.println("]");
		System.out.print("[");
		for(int i = 0; i< actions.length; i++){
			System.out.print(actions[i] + ", ");

		}
		System.out.println("]");
		System.out.println(cumulativeU);
		

	}
	
	public static void generateData(){
		SquaredExponential c1 = new SquaredExponential();
		Periodic c2 = new Periodic();
		Matern m = new Matern();
		Multiplicative mult = new Multiplicative(c1, c2);
		Additive a1 = new Additive(mult, m);
		
		int noData =960;
		double stepsize = 0.01041;
		double[] dataX = new double[noData];
		double[] dataY =  new double[noData];
		for(int i=0; i< dataX.length; i++){
			dataX[i] = i*stepsize;
		}
		for(int i=0; i< noData; i++){
			dataY[i] = 10;
		}
	
		double[] dataP = {2,1.5,1,5,0.2,0.2};//{2,1.5,1,1.2,0.2,0.2}
		int noTestData = 192;
		double[] dataTest = new double[noTestData];// = {11,12,13,14,15,16};
		for(int i = 0; i < dataTest.length; i++){
			dataTest[i] = i*stepsize + stepsize*noData;
		}
		double nl = 0.1;		
		DoubleMatrix X = new DoubleMatrix(dataX);
		DoubleMatrix Y = new DoubleMatrix(dataY);
		DoubleMatrix P = new DoubleMatrix(dataP);
		DoubleMatrix testIn =  new DoubleMatrix(dataTest);
		DoubleMatrix fakeP = new DoubleMatrix(new double[] {1,0.5});
		CovarianceFunction cf = a1;
		GP gp = new GP(X.dup(), testIn.dup(), P.dup(), cf, nl);
		DoubleMatrix samples = gp.generateSamples(X.dup(), P.dup(), nl, cf);
		FileHandler.matrixToCsv(samples, "/users/balz/documents/workspace/masterarbeit/data/temps1.csv");
		printMatrix(samples);
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
