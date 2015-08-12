package main;

import java.io.File;

import learning.BayesInf;
import mdp.EVMDP;
import mdp.HomeHeatingMDP;

import org.jblas.DoubleMatrix;










import org.jblas.util.Random;

import cobyla.Cobyla;

import com.sun.corba.se.spi.ior.MakeImmutable;

import simulation.Simulation;
import util.FileHandler;
import cov.*;
import gp.GP;

public class Main {

	public static void main(String[] args) {
//		interpolate();
//		optimizeParams();
//		makePriceFile();
//		makeNoPeakFile();
		runSim(args);
//		testEVMDP("/users/balz/documents/workspace/masterarbeit/data/prices2.csv", "/users/balz/documents/workspace/masterarbeit/data/out.csv");
//		testBayes();
	}
	
	public static void runSim(String[] args){
		Simulation s = new Simulation();
		//s.work(fhprices, fhout, vminvarInput, qminInput, qmaxInput, tstartInput, tcritInput, mqInput, kwhPerUnitInput)
		s.work(args[0], args[1], args[2], Double.valueOf(args[3]), Double.valueOf(args[4]), Double.valueOf(args[5]),Double.valueOf(args[6]), Double.valueOf(args[7]),Double.valueOf(args[8]),Double.valueOf(args[9]),Double.valueOf(args[10]));
	}
	public static void testSim(){
		Simulation s = new Simulation();
		s.work("/users/balz/documents/workspace/masterarbeit/data/prices2.csv","/users/balz/documents/workspace/masterarbeit/data/out.csv","/users/balz/documents/workspace/masterarbeit/data/comp.csv", 300.,10.,20.,60.,65.,30.,44,5);
	}
	public static void testBayes(){
		DoubleMatrix x = DoubleMatrix.concatVertically((new DoubleMatrix(new double[] {1,1,1,1,1})).transpose(),(new DoubleMatrix(new double[] {1,2,3,4,5})).transpose());
		DoubleMatrix y = new DoubleMatrix(new double[] {3,5,7,9,11});
		double var = 0.05;
		printMatrix(x);
		printMatrix(y);
		BayesInf bi = new BayesInf(x, y, var);
		bi.setup();
		printMatrix(bi.getMean());
	}
	
	
	public static void testEVMDP(String prices, String out){
		String fileHandlePrices = prices;//"/users/balz/documents/workspace/masterarbeit/data/prices2.csv";

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
		double currentLoad = 0;
		int steps = 96;
		int trainSetSize = 1;
		int initialOffset = 96;
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
//			printMatrix(xTrainM);
//			printMatrix(xTestM);
			GP priceGP = new GP(xTrainM,xTestM,P,cf,nl);
			priceGP.setup(yTrainMPrices);
			DoubleMatrix predMeanPrices = priceGP.getPredMean().add(20);
			DoubleMatrix predVarPrices = priceGP.getPredVar();
			EVMDP testmdp = new EVMDP(predMeanPrices,predVarPrices, .5,steps);

			testmdp.setup();
			// heat according to policy and update cumulative utility
			int tmp = initialOffset+steps*trainSetSize;
			for(int o = tmp; o < tmp + steps; o++){
//				System.out.println(o  + " " + testmdp.priceToState(predMeanPrices.get(o-tmp))+ " " +  currentLoad );
				int action = testmdp.getOptPolicy()[o-tmp][testmdp.priceToState(predMeanPrices.get(o-tmp))][testmdp.loadToState(currentLoad)][0];
				cumulativeU += testmdp.rewards(currentLoad, action, priceSamples.get(o)+20,o,0);
				currentLoad = testmdp.updateLoad(currentLoad, action);

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
		printMatrix(subVector(steps*trainSetSize, steps*trainSetSize+runs*steps, priceSamples).add(20));
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
		FileHandler.matrixToCsv(predictedPrices, out);

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
	
	public static void makeNoPeakFile(){
		String fileHandlePrices = "/users/balz/documents/workspace/masterarbeit/data/interpolatedPrices.csv";
		String destination1 = "/users/balz/documents/workspace/masterarbeit/data/noPeakPrices.csv";
		String destination2 = "/users/balz/documents/workspace/masterarbeit/data/peakIndexes.csv";
		String destination3= "/users/balz/documents/workspace/masterarbeit/data/flatPrices.csv";

		DoubleMatrix dm = FileHandler.csvToMatrix(fileHandlePrices);
		DoubleMatrix indexes = DoubleMatrix.zeros(dm.rows);
		DoubleMatrix result = dm.dup();
		for(int i = 0; i < dm.rows; i++){
			if(result.get(i,0) >= 34){
				result.put(i,0,20);
				indexes.put(i,0,1);
			}
		}
		FileHandler.matrixToCsv(indexes, destination2);
		FileHandler.matrixToCsv(DoubleMatrix.ones(dm.rows).mul(20), destination3);

		FileHandler.matrixToCsv(result, destination1);
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
	
	public static void makePriceFile(){
		DoubleMatrix result = new DoubleMatrix(0,1);
		for(int o = 3; o < 6; o++){
			for(int j = 1; j < 13; j++){

				for(int i = 1; i < 32; i++){
					File f = new File("/Users/Balz/Downloads/Outlook/prices/prices-201" + o + "-"+j+"-"+i+".csv");
					if(f.exists() && !f.isDirectory()) { 
						System.out.println("/Users/Balz/Downloads/Outlook/prices/prices-201" + o + "-"+j+"-"+i+".csv");
						DoubleMatrix file = FileHandler.csvToMatrix("/Users/Balz/Downloads/Outlook/prices/prices-201" + o + "-"+j+"-"+i+".csv");
						result = DoubleMatrix.concatVertically(result, file.getColumn(4));
					}
				}
			}
		}
		
		FileHandler.matrixToCsv(result, "/Users/Balz/Downloads/Outlook/allPrices.csv");
	}

	public static void interpolate(){
		DoubleMatrix dm = FileHandler.csvToMatrix("/users/balz/documents/workspace/masterarbeit/data/adjustedPrices.csv");
		DoubleMatrix result = new DoubleMatrix(dm.rows*2, 1);
		for(int i = 0; i < dm.rows*2-2; i += 2){
			result.put((int)(i), dm.get((int)(i/2.),0));
			result.put((int)(i+1), (dm.get((int)(i/2.),0)+dm.get((int)(i/2.+1),0))/2);
		}
		result.put((int)(dm.rows*2-2), dm.get(dm.rows-1,0));
		FileHandler.matrixToCsv(result, "/users/balz/documents/workspace/masterarbeit/data/interpolatedPrices.csv");
	}
	
	
	public static void optimizeParams(){
	/*	1.4016926456748566
		5.939885027240675
		10.258330715290294
		7.814278005934424
		11.71818393993244
		0.8228888605895213
		0.8228888605895213 */
		double[] priceParameters = {1.0368523093853659,5.9031209989290048,1.0368523093853659,.3466674176616187,3.5551018122094575,8.1097474657929007};//{SE1,SE2,P1,OU1,OU2}1.0368523093853659, 5.9031209989290048, .3466674176616187, 3.5551018122094575, 8.1097474657929007, .49489818206999125, 0.049489818206999125};
		SquaredExponential c1 = new SquaredExponential();
		Periodic c2 = new Periodic();
		OrnsteinUhlenbeck m = new OrnsteinUhlenbeck();
		Multiplicative mult = new Multiplicative(c1, c2);
		Additive a1 = new Additive(mult, m);
		CovarianceFunction cf = a1;
		double gpVar = 0;
		DoubleMatrix parameters = new DoubleMatrix(priceParameters);
		int numsteps = 96;
		double stepSize = 1./numsteps;
		System.out.println(stepSize);
		DoubleMatrix predictions = new DoubleMatrix(0,1);
		int numruns = 1;
		int learnSize = 3*numsteps;
		int predictSize = numsteps;
		DoubleMatrix priceData = FileHandler.csvToMatrix("/users/balz/documents/workspace/masterarbeit/data/interpolatedPrices.csv");
		System.out.println("rows " + priceData.rows);
		int learnStart = 0;
//		for(int i = 0; i < numruns; i++){
			double predictStart = learnStart + learnSize;
			double[] xTrain = new double[learnSize];
			for(int o = 0; o < learnSize; o++){
				xTrain[o] = o*stepSize+learnStart*stepSize;
			}
			DoubleMatrix xTrainM = new DoubleMatrix(xTrain);
			double[] xTest = new double[predictSize];
			for(int o = 0; o < predictSize; o++){
				xTest[o] = o*stepSize + predictStart*stepSize;
			}

					printMatrix(xTrainM);
			DoubleMatrix xTestM = new DoubleMatrix(xTest);
					printMatrix(xTestM);

			GP  gp = new GP(xTrainM, xTestM, parameters, cf, gpVar);
			gp.setup(subVector(learnStart,learnStart+learnSize,priceData));
			predictions = DoubleMatrix.concatVertically(predictions, gp.getPredMean());
			learnStart = learnStart  + predictSize;
//		}
//		printMatrix(gp.getTrainCov());
		printMatrix(predictions);
		printMatrix(subVector(learnSize, learnSize + predictSize*numruns, priceData));
		double startRmse = computeRMSE(predictions,subVector(learnSize, learnSize + predictSize*numruns, priceData));
		System.out.println("RMSE at Start: " + startRmse);

		double rhobeg = .5;
		double rhoend = 1.0e-4;
		int iprint = 0;
		int maxfun = 100;
		int numRep = 100;
		int numVar = 6;//cf.getNumParams();
		int numConstr = 2*numVar;
		double upperBound = 10;
		double[] startX = new double[numVar];

		double[][] opt = new double[numVar][numVar];
		double maxLoglikeli = Double.NEGATIVE_INFINITY;
		double[][] res = null;
		double currentRMSE = Double.POSITIVE_INFINITY;
		DoubleMatrix bestVars = new DoubleMatrix();
		for (int r = 0; r<numRep; r++){
			for (int i = 0; i<numVar; i++){
				startX[i] = priceParameters[i];//Math.random()*upperBound;
				
			}
			if(numVar==8){
				startX[6] = Math.random();
			}

			try {
				res =  Cobyla.FindMinimum(gp, numVar, numConstr, startX, rhobeg, rhoend, iprint, maxfun);
				if(res[0][0]>maxLoglikeli){
					maxLoglikeli = res[0][0];
					opt[0][0] = res[0][0];
				}

				
				System.out.println("loglikeli:"+res[0][0]);
				for(int i = 0; i<res[1].length; i++){
					System.out.println(res[1][i]);
				}
				GP newGP = new GP(xTrainM,xTestM, new DoubleMatrix(res[1]),cf,gpVar);
				newGP.setup(subVector(0,learnSize,priceData));
				double rmse = computeRMSE(newGP.getPredMean(),subVector(learnSize, learnSize + predictSize, priceData));
				if(rmse < currentRMSE){
					currentRMSE = rmse;
					bestVars = new DoubleMatrix(res[1]);

				}
				System.out.println(r + " RMSE: " + currentRMSE + " " + rmse + " " + maxLoglikeli + " " +  res[0][0]);

			} catch (Exception e) {
				// TODO: handle exception
			}

		}
		printMatrix(bestVars);

		
	}
	public static double computeRMSE(DoubleMatrix a, DoubleMatrix b){
		double RMSE = 0.;
		DoubleMatrix y = a.sub(b);
		RMSE = (y.mul(y)).sum();

		RMSE = Math.sqrt((RMSE/(double)y.length));
		return RMSE;
	}
}
