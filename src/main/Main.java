package main;

import mdp.EVMDP;
import mdp.HomeHeatingMDP;

import org.jblas.DoubleMatrix;

import com.apple.concurrent.Dispatch.Priority;
import com.sun.corba.se.impl.orb.ParserTable.TestBadServerIdHandler;

import util.FileHandler;
import cov.*;
import gp.GP;

public class Main {

	public static void main(String[] args) {
		testEVMDP();
	}
	
	public static void test1(){
		String fileHandlePrices = "/users/balz/documents/workspace/masterarbeit/data/prices2.csv";
		String fileHandleTemps = "/users/balz/documents/workspace/masterarbeit/data/temps2.csv";

		SquaredExponential c1 = new SquaredExponential();
		Periodic c2 = new Periodic();
		Matern m = new Matern();
		Multiplicative mult = new Multiplicative(c1, c2);
		Additive a1 = new Additive(mult, m);
		
		int noData = 96;
		double stepsize = 1./96.;
		double[] dataX = new double[noData];
		double[] dataY =  new double[noData];
		for(int i=0; i< dataX.length; i++){
			dataX[i] = i*stepsize;
		}
		for(int i=0; i< noData; i++){
			dataY[i] = 10;
		}
	
		double[] dataP = {2,1.5,2,1,1.2,0.2,0.2};//{2,1.5,1,1.2,0.2,0.2}
		double[] dataTest = new double[noData];// = {11,12,13,14,15,16};
		for(int i = 0; i < dataTest.length/10; i++){
			dataTest[i] = i*stepsize + noData*stepsize;
		}
		double nl = 0.05;		
		DoubleMatrix priceSamples = FileHandler.csvToMatrix(fileHandlePrices);
		DoubleMatrix tempSamples = FileHandler.csvToMatrix(fileHandleTemps);
		DoubleMatrix predictedPrices = new DoubleMatrix();
		DoubleMatrix predictedTemps = new DoubleMatrix();
		DoubleMatrix P = new DoubleMatrix(dataP);
		CovarianceFunction cf = a1;
		int runs = 1;
		double cumulativeU = 0;
		double currentTemp = 20;
		int steps = 96;
		int trainSetSize = 0;
		int initialOffset = 0;
		double[] temperatures = new double[runs*steps];
		int[] actions = new int[runs*steps];

		DoubleMatrix priceSimple1 = DoubleMatrix.ones(24).mul(0);
		DoubleMatrix priceSimple2 = DoubleMatrix.ones(24).mul(0);
		DoubleMatrix priceSimple3 = DoubleMatrix.ones(24).mul(40);
		DoubleMatrix priceSimple4 = DoubleMatrix.ones(24).mul(40);
		DoubleMatrix priceSimple = DoubleMatrix.concatVertically(priceSimple1, priceSimple2);
		priceSimple = DoubleMatrix.concatVertically(priceSimple, priceSimple3);
		priceSimple = DoubleMatrix.concatVertically(priceSimple, priceSimple4);

		for(int i = 0; i < runs; i++){
			double[] xTrain = new double[steps*trainSetSize];
			for(int o = 0; o < steps*trainSetSize; o++){
				xTrain[o] = o*stepsize+initialOffset*stepsize;
			}
			DoubleMatrix xTrainM = new DoubleMatrix(xTrain);
			double[] xTest = new double[steps];
			for(int o = 0; o < steps; o++){
				xTest[o] = o*stepsize + (initialOffset+steps)*stepsize*trainSetSize;
			}
			DoubleMatrix xTestM = new DoubleMatrix(xTest);

			
		
//			DoubleMatrix yTrainMPrices = subVector(initialOffset, initialOffset+steps*trainSetSize, priceSamples);
//			DoubleMatrix yTrainMTemps = subVector(initialOffset, initialOffset+steps*trainSetSize, tempSamples);
//
//			GP priceGP = new GP(xTrainM,xTestM,P,cf,nl);
//			priceGP.setup(yTrainMPrices);
//			GP tempGP = new GP(xTrainM,xTestM,P,cf,nl);
//			tempGP.setup(yTrainMTemps);
			DoubleMatrix predMeanPrices = priceSimple;//priceGP.getPredMean().add(20);
			DoubleMatrix predMeanTemps = DoubleMatrix.ones(96).mul(10);//tempGP.getPredMean().add(10);
			DoubleMatrix predVarPrices = DoubleMatrix.ones(96);
			DoubleMatrix predVarTemps = DoubleMatrix.ones(96);
			HomeHeatingMDP testmdp = new HomeHeatingMDP(predMeanPrices,predVarPrices,predMeanTemps, predVarTemps, 5,steps);

			testmdp.work();
			// heat according to policy and update cumulative utility
			int tmp = initialOffset+steps*trainSetSize;
			for(int o = tmp; o < tmp + steps; o++){
				System.out.println(o  + " " + testmdp.priceToState(predMeanPrices.get(o-tmp))+ " " + predMeanTemps.get(o-tmp) + " "+testmdp.internalTempToState(currentTemp) + " " + currentTemp + " " + testmdp.getInternalTemp()[testmdp.internalTempToState(currentTemp)]);
				int action = testmdp.getOptPolicy()[o-tmp][testmdp.priceToState(predMeanPrices.get(o-tmp))][testmdp.internalTempToState(currentTemp)][testmdp.externalTempToState(predMeanTemps.get(o-tmp))];
				currentTemp = testmdp.updateInternalTemperature(currentTemp,predMeanTemps.get(o) , action);
//				if(currentTemp > 24 || currentTemp < 16){
//					System.out.println("strange temps" +testmdp.rewards(currentTemp, 0, priceSamples.get(o)) +" "+ testmdp.rewards(currentTemp, 1, priceSamples.get(o)));
//				}
				cumulativeU += testmdp.rewards(currentTemp, action, priceSamples.get(o));
				temperatures[o-steps*trainSetSize] = currentTemp;
				actions[o-steps*trainSetSize] = action;
			}
//			testmdp.printOptPolicy();
			initialOffset += steps;
			if(i > 0){
				predictedPrices = DoubleMatrix.concatVertically(predictedPrices, predMeanPrices);
				predictedTemps = DoubleMatrix.concatVertically(predictedTemps, predMeanTemps);
			} else {
				predictedPrices = predMeanPrices;
				predictedTemps  = predMeanTemps;
			}

		}
		printMatrix(subVector(steps*trainSetSize, steps*trainSetSize+runs*steps, priceSamples));
		printMatrix(subVector(steps*trainSetSize, steps*trainSetSize+runs*steps, tempSamples));
		printMatrix(predictedPrices);
		printMatrix(predictedTemps);
		System.out.print("[");
		for(int i = 0; i< temperatures.length; i++){
			System.out.print(temperatures[i] + "; ");

		}
		System.out.println("]");
		System.out.print("[");
		for(int i = 0; i< actions.length; i++){
			System.out.print(actions[i] + "; ");

		}
		System.out.println("]");
		System.out.println(cumulativeU);

	}
	public static void testEVMDP(){
		String fileHandlePrices = "/users/balz/documents/workspace/masterarbeit/data/prices2.csv";
		String fileHandleTemps = "/users/balz/documents/workspace/masterarbeit/data/temps2.csv";

		SquaredExponential c1 = new SquaredExponential();
		Periodic c2 = new Periodic();
		Matern m = new Matern();
		Multiplicative mult = new Multiplicative(c1, c2);
		Additive a1 = new Additive(mult, m);
		
		int noData = 96;
		double stepsize = 1./96.;
		double[] dataX = new double[noData];
		double[] dataY =  new double[noData];
		for(int i=0; i< dataX.length; i++){
			dataX[i] = i*stepsize;
		}
		for(int i=0; i< noData; i++){
			dataY[i] = 10;
		}
	
		double[] dataP = {2,1.5,2,1,1.2,0.2,0.2};//{2,1.5,1,1.2,0.2,0.2}
		double[] dataTest = new double[noData];// = {11,12,13,14,15,16};
		for(int i = 0; i < dataTest.length/10; i++){
			dataTest[i] = i*stepsize + noData*stepsize;
		}
		double nl = 0.05;		
		DoubleMatrix priceSamples = FileHandler.csvToMatrix(fileHandlePrices);
		DoubleMatrix predictedPrices = new DoubleMatrix();
		DoubleMatrix P = new DoubleMatrix(dataP);
		CovarianceFunction cf = a1;
		int runs = 1;
		double cumulativeU = 0;
		int currentLoad = 0;
		int steps = 96;
		int trainSetSize = 1;
		int initialOffset = 150;
		double[] loads = new double[runs*steps];
		int[] actions = new int[runs*steps];

		DoubleMatrix priceSimple1 = DoubleMatrix.ones(24).mul(40);
		DoubleMatrix priceSimple2 = DoubleMatrix.ones(24).mul(40);
		DoubleMatrix priceSimple3 = DoubleMatrix.ones(24).mul(0);
		DoubleMatrix priceSimple4 = DoubleMatrix.ones(24).mul(0);
		DoubleMatrix priceSimple = DoubleMatrix.concatVertically(priceSimple1, priceSimple2);
		priceSimple = DoubleMatrix.concatVertically(priceSimple, priceSimple3);
		priceSimple = DoubleMatrix.concatVertically(priceSimple, priceSimple4);

		for(int i = 0; i < runs; i++){
			double[] xTrain = new double[steps*trainSetSize];
			for(int o = 0; o < steps*trainSetSize; o++){
				xTrain[o] = o*stepsize+initialOffset*stepsize;
			}
			DoubleMatrix xTrainM = new DoubleMatrix(xTrain);
			double[] xTest = new double[steps];
			for(int o = 0; o < steps; o++){
				xTest[o] = o*stepsize + (initialOffset+steps)*stepsize*trainSetSize;
			}
			DoubleMatrix xTestM = new DoubleMatrix(xTest);

			
		
			DoubleMatrix yTrainMPrices = subVector(initialOffset, initialOffset+steps*trainSetSize, priceSamples);
			printMatrix(xTrainM);
			printMatrix(xTestM);
			GP priceGP = new GP(xTrainM,xTestM,P,cf,nl);
			priceGP.setup(yTrainMPrices);
			DoubleMatrix predMeanPrices = priceGP.getPredMean().add(20);
			DoubleMatrix predVarPrices = priceGP.getPredVar();
			EVMDP testmdp = new EVMDP(predMeanPrices,predVarPrices, .5,steps);

			testmdp.work();
			// heat according to policy and update cumulative utility
			int tmp = initialOffset+steps*trainSetSize;
			for(int o = tmp; o < tmp + steps; o++){
				System.out.println(o  + " " + testmdp.priceToState(predMeanPrices.get(o-tmp))+ " " +  currentLoad );
				int action = testmdp.getOptPolicy()[o-tmp][testmdp.priceToState(predMeanPrices.get(o-tmp))][currentLoad];
				currentLoad = testmdp.updateLoad(currentLoad, action);

				cumulativeU += testmdp.rewards(currentLoad, action, priceSamples.get(o),o);
				loads[o-steps*trainSetSize-initialOffset] = currentLoad;
				actions[o-steps*trainSetSize-initialOffset] = action;
			}
			initialOffset += steps;
			if(i > 0){
				predictedPrices = DoubleMatrix.concatVertically(predictedPrices, predMeanPrices);
			} else {
				predictedPrices = predMeanPrices;
			}

		}
		printMatrix(subVector(steps*trainSetSize, steps*trainSetSize+runs*steps, priceSamples));
		System.out.println();

		printMatrix(predictedPrices);
		System.out.println();

		System.out.print("[");
		for(int i = 0; i< loads.length; i++){
			System.out.print(loads[i] + "; ");

		}
		System.out.println("]");
		System.out.print("[");
		for(int i = 0; i< actions.length; i++){
			System.out.print(actions[i] + "; ");

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
		
		int noData =96;
		double stepsize = 1./96.;
		double[] dataX = new double[noData];
		double[] dataY =  new double[noData];
		for(int i=0; i< dataX.length; i++){
			dataX[i] = i*stepsize;
		}
		for(int i=0; i< noData; i++){
			dataY[i] = 10;
		}
	
		double[] dataP = {2,1.5,2,1,1.2,0.2,0.2};//{2,1.5,1,1.2,0.2,0.2}
		int noTestData = 50;
		double[] dataTest = new double[noTestData];// = {11,12,13,14,15,16};
		for(int i = 0; i < dataTest.length; i++){
			dataTest[i] = i*stepsize + stepsize*noData;
		}
		double nl = 0.05;		
		DoubleMatrix X = new DoubleMatrix(dataX);
		DoubleMatrix Y = new DoubleMatrix(dataY);
		DoubleMatrix P = new DoubleMatrix(dataP);
		DoubleMatrix testIn = new DoubleMatrix(dataTest);// new DoubleMatrix(dataTest);
		CovarianceFunction cf = a1;
		GP gp = new GP(X.dup(), testIn.dup(), P.dup(), cf, nl);
		DoubleMatrix samples = gp.generateSamples(X.dup(), P.dup(), nl, cf);
		gp.setup(samples);
		printMatrix(gp.getPredMean());
		printMatrix(samples);
		printMatrix(gp.getTrainCov());
		printMatrix(testIn);
		printMatrix(X);
		printMatrix(gp.getPredVar());
//		DoubleMatrix trainOut = subVector(0, noData-noTestData, samples);
//		GP gpnew = new GP(trainIn.dup(), testIn.dup(), P.dup(), cf, nl);
//		gpnew.setup(trainOut);
//		FileHandler.matrixToCsv(samples, "/users/balz/documents/workspace/masterarbeit/data/prices2.csv");
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
