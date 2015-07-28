package simulation;

import gp.GP;

import java.util.ArrayList;

import learning.BayesInf;
import main.Main;
import mdp.EVMDP;

import org.apache.commons.math3.distribution.MultivariateNormalDistribution;
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

	public void work(String fhprices, String fhout, double vminvarInput, double qminInput, double qmaxInput, double tstartInput, double tcritInput, double mqInput){
		/* create User
		 * get User feedback
		 * learn Parameters
		 * make Priceprediction
		 * Solve MDP
		 * Load according to MDP
		 * calculate Utility
		 */
		boolean PROFILING = false;
		long time = (long)0.;

		// user intrinsic parameters
		double vminTrue = vminvarInput;
		double qminTrue = qminInput;
		double qmax = qmaxInput;
		double tstartTrue = tstartInput;
		double tcritTrue = tcritInput;
		double mqTrue = mqInput;
		double bayesInfSD = 5.;
		double vminSD = 50.;
		double mqSD = 50.;
		double tdepTrueSD = 0.05;
		double bayesInfVar = bayesInfSD*bayesInfSD;
		double currentLoad = 0;
		double tdepTrueMean = tstartTrue;
		double tdepLearnedMean = tstartTrue;
		double tdepLearnedSD = tdepTrueSD;
		double returnLoadSD = 2;
		double vmin = sampleFromNormal(vminTrue, vminSD);
		double mq = sampleFromNormal(mqTrue, mqSD);
		double endOfDayOffset = 0;
		// Stopping criterion parameters
		int numberOfSamplesForStoppingCriterion = 0;
		double stoppingCriterionThreshold = 0.5;

		// general simulation parameters
		ArrayList<Double> tdepMeans = new ArrayList<Double>();
		ArrayList<Double> vminList = new ArrayList<Double>();
		ArrayList<Double> mqList = new ArrayList<Double>();
		String fileHandleOut = fhout;
		double priceOffset = 20;
		int counter = 0;
		int numberOfRuns = 10;
		double totalUtility = 0;
		ArrayList<Double> additionalLoadDataQ = new ArrayList<Double>();
		ArrayList<Double> additionalLoadDataV = new ArrayList<Double>();
		double[] loadDataQuestions = {0.5,0.6};//{0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1};
		boolean stopAsking = false;
		// Price Prediction parameters
		String fileHandlePrices = fhprices;//"/users/balz/documents/workspace/masterarbeit/data/prices2.csv";
		DoubleMatrix priceSamples = FileHandler.csvToMatrix(fileHandlePrices);

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
		int learnSize = 288;
		int predictStart = learnStart + learnSize;
		int predictSize = 96;
		int dataPointsPerDay = 96;
		double treturnMean = 0;
		double treturnSD = 0.05;
		double unpluggedDuration = 30;
		double stepSize = 1./dataPointsPerDay;

		// MDP Parameters
		double deltaPrice = 0.5;
		int numSteps = 96;

		// 
		ArrayList<Double> loads = new ArrayList<Double>();
		ArrayList<Double> realPrices = new ArrayList<Double>();
		ArrayList<Double> predictedPrices = new ArrayList<Double>();

		ArrayList<Integer> actions = new ArrayList<Integer>();
		ArrayList<Double> totalUtilities = new ArrayList<Double>();
		DoubleMatrix printOut = new DoubleMatrix(0,4);
		while(counter < numberOfRuns){


			// sample the observed values from normal distributions
			//!!! values need to be consistent
			// qmin, tstart & tcrit actual values
			if(PROFILING){
				time = System.nanoTime();
			}
			double qmin = qminTrue;//sampleFromNormal(qmintrue, sd);
			double tstart = tstartTrue;//sampleFromNormal(tstarttrue, sd);
			double tcrit = tcritTrue;//sampleFromNormal(tcrittrue, sd);
			// if we don't have reached the stopping criterion yet ask the user about mq and vmin
			if(!stopAsking){
				mq = sampleFromNormal(mqTrue, mqSD);
				while(mq < 0 ){
					mq = sampleFromNormal(mqTrue, mqSD);
				}
				mqList.add(mq);
				vmin = sampleFromNormal(vminTrue, vminSD);
				while(vmin < 0){
					vmin = sampleFromNormal(vminTrue, vminSD);
				}
				vminList.add(vmin);
			}
			double tdep = sampleFromNormal(tdepTrueMean, tdepTrueSD);
			while(!isInRange(tdep, tstart, tcrit-tstart)){
				tdep = sampleFromNormal(tdepTrueMean, tdepTrueSD);
			}
			// update the learned mean departure time with new information
			tdepMeans.add(tdep);
			double tdepSum = 0;
			for(Double d: tdepMeans){
				tdepSum += d;
			}
			double tdepSampleMean = tdepSum/tdepMeans.size();
			double tdepVar = tdepLearnedSD*tdepLearnedSD;
			tdepLearnedMean = (tdepSampleMean*tdepVar/(tdepMeans.size()*tdepVar+tdepVar))+(tdepMeans.size()*tdepLearnedMean*tdepVar/(tdepMeans.size()*tdepVar+tdepVar));

			// choose a Question at random and add the additional data
			int rndQuestion = Random.nextInt(loadDataQuestions.length);
			additionalLoadDataQ.add(qmin+(qmax-qmin)*loadDataQuestions[rndQuestion]);
			additionalLoadDataV.add((vmin-qmin*mq)+(qmin + (qmax-qmin)*loadDataQuestions[rndQuestion])*mq);
			DoubleMatrix xLoad = DoubleMatrix.concatVertically((DoubleMatrix.ones(counter+1)).transpose(),(new DoubleMatrix(additionalLoadDataQ)).transpose());
			if(PROFILING){
				System.out.println("Init Data - Time exceeded: "+(System.nanoTime()-time)/Math.pow(10,9)+" s");
			}
			if(PROFILING){
				time = System.nanoTime();
			}
			// do Bayesian inference to get mq and vmin
			BayesInf bi = new BayesInf(xLoad, new DoubleMatrix(additionalLoadDataV), bayesInfVar);
			Main.printMatrix(xLoad);
			Main.printMatrix(new DoubleMatrix(additionalLoadDataV));
			bi.setup();
			if(PROFILING){
				System.out.println("Bayesian Inference - Time exceeded: "+(System.nanoTime()-time)/Math.pow(10,9)+" s");
			}
			double mqLearned = bi.getMean().get(1);
			double vminLearned =  bi.getMean().get(0)+mqLearned*qmin;
			if(PROFILING){
				time = System.nanoTime();
			}
			//prepare GP
			double[] xTrain = new double[learnSize];
			for(int o = 0; o < learnSize; o++){
				xTrain[o] = o*stepSize+learnStart*stepSize;
			}
			DoubleMatrix xTrainM = new DoubleMatrix(xTrain);
			double[] xTest = new double[predictSize];
			for(int o = 0; o < predictSize; o++){
				xTest[o] = o*stepSize + predictStart*stepSize;
			}
			DoubleMatrix xTestM = new DoubleMatrix(xTest);
//			System.out.println("Ahi" + priceSamples.rows);
			DoubleMatrix yTrainMPrices = Main.subVector(learnStart, learnStart+learnSize, priceSamples);
			DoubleMatrix realPriceMatrix = Main.subVector(predictStart, predictStart+predictSize, priceSamples);
			GP priceGP = new GP(xTrainM,xTestM,parameters,cf,gpVar);
			//			Main.printMatrix(xTestM);
			//			Main.printMatrix(xTrainM);
//						Main.printMatrix(yTrainMPrices);
			priceGP.setup(yTrainMPrices);
			if(PROFILING){
				System.out.println("Gaussian Process - Time exceeded: "+(System.nanoTime()-time)/Math.pow(10,9)+" s");
			}
			DoubleMatrix predMeanPrices = priceGP.getPredMean().add(priceOffset);
			DoubleMatrix predVarPrices = priceGP.getPredVar();
			// setup mdp
			if(PROFILING){
				time = System.nanoTime();
			}
			System.out.println("tdep: " + (tdepLearnedMean  + endOfDayOffset) );
			System.out.println("tstart: " + (tstart  + endOfDayOffset) );

			System.out.println("tcrit: " + (tcrit  + endOfDayOffset) );

			EVMDP mdp = new EVMDP(predMeanPrices, predVarPrices, deltaPrice, numSteps, qmax, qmin, mqLearned, tstart + endOfDayOffset, tcrit + endOfDayOffset, vminLearned,tdepLearnedMean  + endOfDayOffset,tdepLearnedSD,currentLoad);
			mdp.work();
			// charge the vehicle according to policy and update total utility
			if(PROFILING){
				System.out.println("mdp calculation - Time exceeded: "+(System.nanoTime()-time)/Math.pow(10,9)+" s");
			}
			// sample tdep and load till tdep, calculate utility until tdep
			if(PROFILING){
				time = System.nanoTime();
			}
			// Check stopping criterion
			if(!stopAsking){
				double expectedUtility = mdp.getqValues()[0][mdp.priceToState(realPriceMatrix.get(0)+priceOffset)][mdp.loadToState(currentLoad)][0];
				double cumulatedDifference = 0;
				double[][] covarianceMatrix = new double[2][2];
				covarianceMatrix[0] = new double[] {bi.getAInv().get(0,0),bi.getAInv().get(0,1)};
				covarianceMatrix[1] = new double[] {bi.getAInv().get(1,0),bi.getAInv().get(1,1)};
				Main.printMatrix(new DoubleMatrix(covarianceMatrix));
				Main.printMatrix(bi.getAInv());
				Main.printMatrix(bi.getMean());
				MultivariateNormalDistribution mvnd = new MultivariateNormalDistribution(new double[] {bi.getMean().get(0),bi.getMean().get(1)},covarianceMatrix);
				for(int i = 0; i < numberOfSamplesForStoppingCriterion; i++){
					double[] sample = mvnd.sample();
					double sampledVmin = sample[0] + qmin * sample[1];
					double sampledMQ = sample[1];
					System.out.println(i + " " + sampledVmin + " vminmq " + sampledMQ);
					EVMDP samplemdp = new EVMDP(predMeanPrices, predVarPrices, deltaPrice, numSteps, qmax, qmin, sampledMQ, tstart  + endOfDayOffset, tcrit  + endOfDayOffset, sampledVmin,tdepLearnedMean + endOfDayOffset,tdepLearnedSD,currentLoad);
					samplemdp.work();
					double expectedSampleUtility = samplemdp.getqValues()[0][mdp.priceToState(realPriceMatrix.get(0)+priceOffset)][mdp.loadToState(currentLoad)][0];
					cumulatedDifference += expectedSampleUtility - expectedUtility;
				}
				System.out.println("stoppinval " + Math.abs(cumulatedDifference/numberOfSamplesForStoppingCriterion));
//				if(Math.abs(cumulatedDifference/numberOfSamplesForStoppingCriterion) < stoppingCriterionThreshold){
//					stopAsking = true;
//				}
			}
			if(PROFILING){
				System.out.println("stopping criterion - Time exceeded: "+(System.nanoTime()-time)/Math.pow(10,9)+" s");
			}
			for(int o = 0; o < tdep + endOfDayOffset; o++){
				if(mdp.priceToState(realPriceMatrix.get(o)+priceOffset) < 0){
					System.out.println("Bad Price" + realPriceMatrix.get(o)+priceOffset);
				}
				int action = mdp.getOptPolicy()[o][mdp.priceToState(realPriceMatrix.get(o)+priceOffset)][mdp.loadToState(currentLoad)][0];
				// cost summation, save for every day
				totalUtility += mdp.rewards(currentLoad, action, realPriceMatrix.get(o)+priceOffset,o,0);
				currentLoad = mdp.updateLoad(currentLoad, action);
				loads.add(currentLoad);
				actions.add(action);
				realPrices.add(realPriceMatrix.get(o));
				totalUtilities.add(-1.);
				predictedPrices.add(predMeanPrices.get(o));
			}
			System.out.println("endload: " + currentLoad);
			System.out.println("costs: " + totalUtility);

			totalUtility = totalUtility + mdp.rewards(currentLoad, 0, 0, (int)tdep, 1);

			System.out.println("total utility: " + totalUtility);

			counter++;
			if(currentLoad > 0){
				double tempload = currentLoad - qmin;
				currentLoad = (int)sampleFromNormal(tempload, returnLoadSD);
				while(currentLoad < 0){
					currentLoad = (int)sampleFromNormal(tempload, returnLoadSD);
				}
			}
//			loads.add(currentLoad);
//			actions.add(-1);
//			realPrices.add(-1.);
//			totalUtilities.add(totalUtility);
//			predictedPrices.add(-1.);

			System.out.println("returnload: " + currentLoad);
			// TODO:
			System.out.println("predicstart: " + predictStart);

			treturnMean = sampleFromNormal(predictStart + tdep + endOfDayOffset + unpluggedDuration, treturnSD);
			learnStart = (int)(treturnMean - learnSize);
			predictStart = (int)treturnMean;
			endOfDayOffset = numSteps-treturnMean%numSteps;
			System.out.println("tret: " + treturnMean);
			System.out.println("endofday offset: " + endOfDayOffset);
			System.out.println();
			totalUtility = 0;



		}
		//		printOut = DoubleMatrix.concatHorizontally(DoubleMatrix.concatHorizontally(DoubleMatrix.concatHorizontally(new DoubleMatrix(loads), new DoubleMatrix(realPrices)),new DoubleMatrix(predictedPrices)),new DoubleMatrix(totalUtilities));
		Main.printMatrix(new DoubleMatrix(loads));
		Main.printMatrix(new DoubleMatrix(predictedPrices));
		Main.printMatrix(new DoubleMatrix(realPrices));

		//		FileHandler.matrixToCsv(printOut, fileHandleOut);

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
