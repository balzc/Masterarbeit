package simulation;

import gp.GP;

import java.util.ArrayList;

import learning.BayesInf;
import main.Main;
import mdp.EVMDP;

import org.apache.commons.math3.distribution.NormalDistribution;
import org.jblas.DoubleMatrix;
import org.jblas.util.Random;

import util.FileHandler;
import cov.Additive;
import cov.CovarianceFunction;
import cov.Matern;
import cov.Multiplicative;
import cov.Periodic;
import cov.SquaredExponential;

public class Simulation {
	public void work(){
		/* create User
		 * get User feedback
		 * learn Parameters
		 * make Priceprediction
		 * Solve MDP
		 * Load according to MDP
		 * calculate Utility
		 */
		// user intrinsic parameters
		double vmintrue = 600.;
		double qmintrue = 10.;
		double tstarttrue = 15.;
		double tcrittrue = 18.;
		double mqtrue = 60.;
		double sd = 0.05;
		double var = sd*sd;
		double qmax = 20.;
		double qrequired = 10.;
		double currentLoad = 0;
		double tdepmean = 15;
		double tdeplearned = tstarttrue;
		double tdeplearnedsd = 0.05;
		double returnLoadsd = 2;
		// general simulation parameters
		double priceOffset = 20;
		int counter = 0;
		int numberOfRuns = 10;
		double totalUtility = 0;
		ArrayList<Double> additionalLoadDataQ = new ArrayList<Double>();
		ArrayList<Double> additionalLoadDataV = new ArrayList<Double>();
		double[] loadDataQuestions = {0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9};
		
		// Price Prediction parameters
		String fileHandlePrices = "/users/balz/documents/workspace/masterarbeit/data/prices2.csv";
		double[] priceParameters = {2,1.5,2,1,1.2,0.2,0.2};
		SquaredExponential c1 = new SquaredExponential();
		Periodic c2 = new Periodic();
		Matern m = new Matern();
		Multiplicative mult = new Multiplicative(c1, c2);
		Additive a1 = new Additive(mult, m);
		CovarianceFunction cf = a1;
		double gpVar = 0.05;
		DoubleMatrix parameters = new DoubleMatrix(priceParameters);
		int learnStart = 0;
		int learnEnd = 19;
		int learnSize = 20;
		int predictStart = 20;
		int predictEnd = 39;
		int predictSize = 20;
		int dataPointsPerDay = 20;
		double stepsize = 1./dataPointsPerDay;

		// MDP Parameters
		double deltaPrice = 5;
		int numSteps = 20;
		
		// 
		ArrayList<Double> loads = new ArrayList<Double>();
		ArrayList<Integer> actions = new ArrayList<Integer>();
		ArrayList<Double> totalUtilities = new ArrayList<Double>();
		while(counter < numberOfRuns){

		
			// sample the observed values from normal distributions
			//!!! values need to be consistent
			// qmin, tstart & tcrit actual values
			double vmin = sampleFromNormal(vmintrue, sd);
			while(vmin < 0){
				vmin = sampleFromNormal(vmintrue, sd);
			}
			double qmin = qmintrue;//sampleFromNormal(qmintrue, sd);
			double tstart = tstarttrue;//sampleFromNormal(tstarttrue, sd);
			double tcrit = tcrittrue;//sampleFromNormal(tcrittrue, sd);
			double mq = sampleFromNormal(mqtrue, sd);
			while(mq < 0 ){
				mq = sampleFromNormal(mqtrue, sd);
			}
			double tdep = sampleFromNormal(tdepmean, sd);
			while(!isInRange(tdep, tstart, tcrit-tstart)){
				tdep = sampleFromNormal(tdepmean, sd);
			}
			
			int rndQuestion = Random.nextInt(loadDataQuestions.length-1);
			additionalLoadDataQ.add((qmax-qmin)*loadDataQuestions[rndQuestion]);
			additionalLoadDataV.add(vmin + (qmax-qmin)*loadDataQuestions[rndQuestion]*mq);
			DoubleMatrix xLoad = DoubleMatrix.concatVertically((DoubleMatrix.ones(counter+1)).transpose(),(new DoubleMatrix(additionalLoadDataQ)).transpose());
			// do Bayesian inference to get mq and vmin
			BayesInf bi = new BayesInf(xLoad, new DoubleMatrix(additionalLoadDataV), var);
			bi.setup();
			// bad first few runs
			double mqLearned = bi.getMean().get(1);
			double vminLearned =  bi.getMean().get(0);

			//prepare GP
			double[] xTrain = new double[learnSize];
			for(int o = 0; o < learnSize; o++){
				xTrain[o] = o*stepsize+learnStart*stepsize;
			}
			DoubleMatrix xTrainM = new DoubleMatrix(xTrain);
			double[] xTest = new double[predictSize];
			for(int o = 0; o < predictSize; o++){
				xTest[o] = o*stepsize + predictStart*stepsize;
			}
			DoubleMatrix xTestM = new DoubleMatrix(xTest);
			DoubleMatrix priceSamples = FileHandler.csvToMatrix(fileHandlePrices);
			DoubleMatrix yTrainMPrices = Main.subVector(learnStart, learnSize, priceSamples);
			GP priceGP = new GP(xTrainM,xTestM,parameters,cf,gpVar);
//			Main.printMatrix(xTestM);
//			Main.printMatrix(xTrainM);
//			Main.printMatrix(yTrainMPrices);
			priceGP.setup(yTrainMPrices);
			DoubleMatrix predMeanPrices = priceGP.getPredMean().add(priceOffset);
			DoubleMatrix predVarPrices = priceGP.getPredVar();
			// setup mdp
			EVMDP mdp = new EVMDP(predMeanPrices, predVarPrices, deltaPrice, numSteps, qmax, qrequired, mqLearned, tstart, tcrit, vminLearned,tdeplearned,tdeplearnedsd,currentLoad);
			mdp.work();
			// charge the vehicle according to policy and update total utility
			
			// sample tdep and load till tdep, calculate utility until tdep
			System.out.println("tdep: " + tdep);

			for(int o = 0; o < tdep; o++){
				// pricesSamples instead of predmean
				int action = mdp.getOptPolicy()[o][mdp.priceToState(priceSamples.get(o)+priceOffset)][mdp.loadToState(currentLoad)][0];
				// cost summation, save for every day
				totalUtility += mdp.rewards(currentLoad, action, priceSamples.get(o)+priceOffset,o,0);
				currentLoad = mdp.updateLoad(currentLoad, action);
				loads.add(currentLoad);
				actions.add(action);
			}
			System.out.println("endload: " + currentLoad);
			System.out.println("costs: " + totalUtility);

			totalUtility = totalUtility + mdp.rewards(currentLoad, 0, 0, (int)tdep, 1);
			totalUtilities.add(totalUtility);
			System.out.println("total utility: " + totalUtility);

			totalUtility = 0;
			counter++;
			if(currentLoad > 0){
				double tempload = currentLoad - qmin;
				currentLoad = (int)sampleFromNormal(tempload, returnLoadsd);
				while(currentLoad < 0){
					currentLoad = (int)sampleFromNormal(tempload, returnLoadsd);
				}
			}
			System.out.println("returnload: " + currentLoad);
			// TODO: update tdepmean and sd, update learn and predict interval, introduce treturn
			learnEnd += learnSize;
			predictStart = predictEnd;
			predictEnd += predictSize;
		}
	}
	
//	public GP predictPrices(){
//		String fileHandlePrices = prices;//"/users/balz/documents/workspace/masterarbeit/data/prices2.csv";
//
//		SquaredExponential c1 = new SquaredExponential();
//		Periodic c2 = new Periodic();
//		Matern m = new Matern();
//		Multiplicative mult = new Multiplicative(c1, c2);
//		Additive a1 = new Additive(mult, m);
//		
//		int noData = 96;
//		double stepsize = 1./96.;
//		double[] dataX = new double[noData];
//		double[] dataY =  new double[noData];
//		for(int i=0; i< dataX.length; i++){
//			dataX[i] = i*stepsize;
//		}
//		for(int i=0; i< noData; i++){
//			dataY[i] = 10;
//		}
//	
//		double[] dataP = {2,1.5,2,1,1.2,0.2,0.2};//{2,1.5,1,1.2,0.2,0.2}
//		double[] dataTest = new double[noData];// = {11,12,13,14,15,16};
//		for(int i = 0; i < dataTest.length/10; i++){
//			dataTest[i] = i*stepsize + noData*stepsize;
//		}
//		double nl = 0.05;		
//		DoubleMatrix priceSamples = FileHandler.csvToMatrix(fileHandlePrices);
//		DoubleMatrix predictedPrices = new DoubleMatrix();
//		DoubleMatrix P = new DoubleMatrix(dataP);
//		CovarianceFunction cf = a1;
//		int runs = 1;
//		double cumulativeU = 0;
//		double currentLoad = 0;
//		int steps = 96;
//		int trainSetSize = 1;
//		int initialOffset = 0;
//		double[] loads = new double[runs*steps];
//		int[] actions = new int[runs*steps];
//
//		DoubleMatrix priceSimple1 = DoubleMatrix.ones(24).mul(40);
//		DoubleMatrix priceSimple2 = DoubleMatrix.ones(24).mul(40);
//		DoubleMatrix priceSimple3 = DoubleMatrix.ones(24).mul(0);
//		DoubleMatrix priceSimple4 = DoubleMatrix.ones(24).mul(0);
//		DoubleMatrix priceSimple = DoubleMatrix.concatVertically(priceSimple1, priceSimple2);
//		priceSimple = DoubleMatrix.concatVertically(priceSimple, priceSimple3);
//		priceSimple = DoubleMatrix.concatVertically(priceSimple, priceSimple4);
//
//		for(int i = 0; i < runs; i++){
//			double[] xTrain = new double[steps*trainSetSize];
//			for(int o = 0; o < steps*trainSetSize; o++){
//				xTrain[o] = o*stepsize+initialOffset*stepsize;
//			}
//			DoubleMatrix xTrainM = new DoubleMatrix(xTrain);
//			double[] xTest = new double[steps];
//			for(int o = 0; o < steps; o++){
//				xTest[o] = o*stepsize + (initialOffset+steps)*stepsize*trainSetSize;
//			}
//			DoubleMatrix xTestM = new DoubleMatrix(xTest);
//
//			
//		
//			DoubleMatrix yTrainMPrices = subVector(initialOffset, initialOffset+steps*trainSetSize, priceSamples);
////			printMatrix(xTrainM);
////			printMatrix(xTestM);
//			GP priceGP = new GP(xTrainM,xTestM,P,cf,nl);
//			priceGP.setup(yTrainMPrices);
//			DoubleMatrix predMeanPrices = priceGP.getPredMean().add(20);
//			DoubleMatrix predVarPrices = priceGP.getPredVar();
//			EVMDP mdp = new EVMDP(predMeanPrices,predVarPrices, .5,steps);
//
//			mdp.work();
//			// heat according to policy and update cumulative utility
//			int tmp = initialOffset+steps*trainSetSize;
//			for(int o = tmp; o < tmp + steps; o++){
////				System.out.println(o  + " " + testmdp.priceToState(predMeanPrices.get(o-tmp))+ " " +  currentLoad );
//				int action = mdp.getOptPolicy()[o-tmp][mdp.priceToState(predMeanPrices.get(o-tmp))][mdp.loadToState(currentLoad)];
//				cumulativeU += mdp.rewards(currentLoad, action, priceSamples.get(o)+20,o);
//				currentLoad = mdp.updateLoad(currentLoad, action);
//
//				loads[o-steps*trainSetSize-initialOffset] = currentLoad;
//				actions[o-steps*trainSetSize-initialOffset] = action;
//			}
//			initialOffset += steps;
//			if(i > 0){
//				predictedPrices = DoubleMatrix.concatVertically(predictedPrices, predMeanPrices);
//			} else {
//				predictedPrices = predMeanPrices;
//			}
//
//		}
//		printMatrix(subVector(steps*trainSetSize, steps*trainSetSize+runs*steps, priceSamples).add(20));
//		System.out.println();
//
//		printMatrix(predictedPrices);
//		System.out.println();
//
//		System.out.print("[");
//		for(int i = 0; i< loads.length; i++){
//			System.out.print(loads[i] + "; ");
//
//		}
//		System.out.println("]");
//		System.out.print("[");
//		for(int i = 0; i< actions.length; i++){
//			System.out.print(actions[i] + "; ");
//
//		}
//		System.out.println("]");
//		System.out.println(cumulativeU);
//		FileHandler.matrixToCsv(predictedPrices, out);
//	}
	
	public double sampleFromNormal(double mean, double sd){
		NormalDistribution nd = new NormalDistribution(mean, sd);
		return nd.sample();
	}
	
	public boolean isInRange(double value, double goal, double range){
		if(Math.abs(value-goal)<= range){
			return true;
		}
		else{
			return false;
		}
	}
	
}
