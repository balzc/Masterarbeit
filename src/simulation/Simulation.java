package simulation;

import gp.GP;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;

import learning.BayesInf;
import main.Main;
import mdp.EVMDP;

import org.apache.commons.math3.distribution.MultivariateNormalDistribution;
import org.apache.commons.math3.distribution.NormalDistribution;
import org.jblas.DoubleMatrix;
import org.jblas.util.Random;

import agents.LowPriceLoader;
import agents.PlugAndForgetLoader;
import agents.SortedMinLoader;
import util.FileHandler;
import cov.Additive;
import cov.CovarianceFunction;
import cov.Matern;
import cov.Multiplicative;
import cov.OrnsteinUhlenbeck;
import cov.Periodic;
import cov.SquaredExponential;

public class Simulation {

	/*
	 * @param blablabla
	 */
	public void work(String fhprices, String fhout,String fhcomp, double vminvarInput, double qminInput, double qmaxInput, double tstartInput, double tcritInput, double mqInput, double kwhPerUnitInput,double bisd){
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
		if(!PROFILING){
			time = System.nanoTime();
		}
		// user intrinsic parameters
		// input is the kwh of a car battery, qmax is then the number of timesteps required to fill that battery with kwhPerUnitInput
		double qmax = qmaxInput/kwhPerUnitInput;
		// input is a percentage number, qminTrue is then the number of timesteps required to fill the battery to that percentage
		double qminTrue = qminInput*qmaxInput/(kwhPerUnitInput*100);
		// input is a price per kwh, vmintrue is then the number of units required for qmin times the price per kwh times kwh per unit
		double vminTrue = vminvarInput*qminTrue*kwhPerUnitInput;
		// input relative to time discretization
		double tstartTrue = tstartInput;
		// input relative to tstart, tcritTrue value relative to time discretization, tcritTrue is no offset anymore
		double tcritTrue = tcritInput + tstartInput;
		// input is price per kwh, mqTrue is then utility per loadunit
		double mqTrue = mqInput;
		boolean log = false;
		String runId = "qmax " + qmax + "  qmin " + qminTrue + " vmin " + vminTrue + " mq " + mqTrue + " kwh " + kwhPerUnitInput + " bisd "+ bisd;
		if(log)
		System.out.println("qmax " + qmax + "  qmin " + qminTrue + " vmin " + vminTrue + " mq " + mqTrue + " kwh " + kwhPerUnitInput + " bisd "+ bisd);
		double vminSD = 0.1;// mqsd *qmin
		double mqSD = 0.1;//5
		double tdepTrueSD = 0.0005;//15 min
		double bayesInfVar = bisd;
		double vtrueSD = 0.5;
		
		double tdepTrueMean = tstartTrue;
		double tdepLearnedMean = tstartTrue;
		double tdepLearnedSD = tdepTrueSD;
		double returnLoadSD = 2;
		double vmin = sampleFromNormal(vminTrue, vminSD);
		double mq = sampleFromNormal(mqTrue, mqSD);
		double endOfDayOffset = 0;
		double kwhPerUnit = kwhPerUnitInput;
		double currentLoadMDP = 0;
		double currentLoadLPL = 0;
		double currentLoadPAFL = 0;
		double currentLoadSML = 0;

		// Stopping criterion parameters
		int numberOfSamplesForStoppingCriterion = 1000;
		int numberOfConcurrentThreads = 10;
		double stoppingCriterionThreshold = 0.00001;
		// general simulation parameters
		int numberOfRuns = 1;
		
		ArrayList<Double> tdepMeans = new ArrayList<Double>();
		ArrayList<Double> vminList = new ArrayList<Double>();
		ArrayList<Double> mqList = new ArrayList<Double>();
		String fileHandleOut = fhout;
		double priceOffset = 0;
		int counter = 0;
		double totalUtilityMDP = 0;
		double totalUtilityLPL = 0;
		double totalUtilityPAFL = 0;
		double totalUtilitySML = 0;
		double mqLearned = mqTrue;
		double vminLearned = vminTrue;
		
		ArrayList<Double> additionalLoadDataQ = new ArrayList<Double>();
		ArrayList<Double> additionalLoadDataV = new ArrayList<Double>();
		double[] loadDataQuestions = {0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1};
		boolean notLearning = false;
		// Price Prediction parameters
		String fileHandlePrices = fhprices;//"/users/balz/documents/workspace/masterarbeit/data/prices2.csv";
		DoubleMatrix priceSamples = FileHandler.csvToMatrix(fileHandlePrices);

		double[] priceParameters = {1.0368523093853659,5.9031209989290048,1.0368523093853659,.3466674176616187,3.5551018122094575,8.1097474657929007};
		SquaredExponential c1 = new SquaredExponential();
		Periodic c2 = new Periodic();
		OrnsteinUhlenbeck ou = new OrnsteinUhlenbeck();
		Multiplicative mult = new Multiplicative(c1, c2);
		Additive a1 = new Additive(mult, ou);
		CovarianceFunction cf = a1;
		double gpVar = 0.5;
		DoubleMatrix parameters = new DoubleMatrix(priceParameters);
		int learnStart = 17472+96*9;//17472; // start after half a year
		int learnSize = 96;
		int predictStart = learnStart + learnSize;
		int predictSize = 96;
		int dataPointsPerDay = 96;
		double treturnMean = 0;
		double treturnSD = 0.05;
		double unpluggedDuration = 28;
		double stepSize = 1./dataPointsPerDay;

		// MDP Parameters
		double deltaPrice = 1;
		int numSteps = 96;

		// 
		LowPriceLoader LPL = new LowPriceLoader(mq, vmin, qminTrue, qmax);
		PlugAndForgetLoader PAFL = new PlugAndForgetLoader(qmax);
		ArrayList<Double> loadsMDP = new ArrayList<Double>();
		ArrayList<Double> loadsLPL = new ArrayList<Double>();
		ArrayList<Double> loadsPAFL = new ArrayList<Double>();
		ArrayList<Double> loadsSML = new ArrayList<Double>();

		ArrayList<Double> loadsDailyMDP = new ArrayList<Double>();
		ArrayList<Double> loadsDailyLPL = new ArrayList<Double>();
		ArrayList<Double> loadsDailyPAFL = new ArrayList<Double>();
		ArrayList<Double> loadsDailySML = new ArrayList<Double>();

		ArrayList<Double> realPrices = new ArrayList<Double>();
		ArrayList<Double> predictedPrices = new ArrayList<Double>();

		ArrayList<Double> actionsMDP = new ArrayList<Double>();
		ArrayList<Double> actionsLPL = new ArrayList<Double>();
		ArrayList<Double> actionsPAFL = new ArrayList<Double>();
		ArrayList<Double> actionsSML = new ArrayList<Double>();

		ArrayList<Double> costsDailyMDP = new ArrayList<Double>();
		ArrayList<Double> costsDailyLPL = new ArrayList<Double>();
		ArrayList<Double> costsDailyPAFL = new ArrayList<Double>();
		ArrayList<Double> costsDailySML = new ArrayList<Double>();

		ArrayList<Double> costsPerDeltaTMDP = new ArrayList<Double>();
		ArrayList<Double> costsPerDeltaTLPL = new ArrayList<Double>();
		ArrayList<Double> costsPerDeltaTPAFL = new ArrayList<Double>();
		ArrayList<Double> costsPerDeltaTSML = new ArrayList<Double>();

		ArrayList<Double> totalUtilitiesMDP = new ArrayList<Double>();
		ArrayList<Double> totalUtilitiesLPL = new ArrayList<Double>();
		ArrayList<Double> totalUtilitiesPAFL = new ArrayList<Double>();
		ArrayList<Double> totalUtilitiesSML = new ArrayList<Double>();

		ArrayList<Double> regret = new ArrayList<Double>();
		ArrayList<Double> mqDiffs = new ArrayList<Double>();
		ArrayList<Double> vminDiffs = new ArrayList<Double>();

		ArrayList<Double> timeStepsCounter = new ArrayList<Double>();

		// Start simulation
		while(counter < numberOfRuns){
//			if(counter%100 == 0)
//			System.out.println("Run " + counter + "   " + runId);
			// sample the observed values from normal distributions
			//!!! values need to be consistent
			// qmin, tstart & tcrit actual values
			if(PROFILING){
				time = System.nanoTime();
			}
			double qmin = qminTrue;//sampleFromNormal(qmintrue, sd);
			double tstart = tstartTrue;//sampleFromNormal(tstarttrue, sd);
			double tcrit = tcritTrue;//sampleFromNormal(tcrittrue, sd);
			double q = 0;
			double v = 0;
			// if we don't have reached the stopping criterion yet ask the user about mq and vmin
			if(!notLearning){
				int rndQuestion = Random.nextInt(loadDataQuestions.length);
//				System.out.println(qmax + " " + qmin);
//				System.out.println(vminTrue + " " + mqTrue);
				q = qmin*kwhPerUnit + (qmax-qmin)*kwhPerUnit*loadDataQuestions[rndQuestion];
				double vtrue = vminTrue + (q-qmin*kwhPerUnit)*mqTrue;
				v = sampleFromNormal(vtrue, vtrueSD);

//				mq = sampleFromNormal(mqTrue, mqSD);
//				while(mq < 0 ){
//					mq = sampleFromNormal(mqTrue, mqSD);
//				}
//				mqList.add(mq);
//				vmin = sampleFromNormal(vminTrue, vminSD);
//				while(vmin < 0){
//					vmin = sampleFromNormal(vminTrue, vminSD);
//				}
//				vminList.add(vmin);
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
//			int rndQuestion = Random.nextInt(loadDataQuestions.length);
//			double qvalue = qmin+(qmax-qmin)*loadDataQuestions[rndQuestion];
//			additionalLoadDataQ.add(qvalue);
//			additionalLoadDataV.add((vmin-qmin*mq)+(qmin + (qmax-qmin)*loadDataQuestions[rndQuestion])*mq);
			additionalLoadDataQ.add(q);
			additionalLoadDataV.add(v);
			DoubleMatrix xLoad = DoubleMatrix.concatVertically((DoubleMatrix.ones(counter+1)).transpose(),(new DoubleMatrix(additionalLoadDataQ)).transpose());
			if(PROFILING){
				System.out.println("Init Data - Time exceeded: "+(System.nanoTime()-time)/Math.pow(10,9)+" s");
			}
			if(PROFILING){
				time = System.nanoTime();
			}
			// do Bayesian inference to get mq and vmin  DoubleMatrix x, DoubleMatrix y, DoubleMatrix priorcovar, DoubleMatrix priormean, double var){
			DoubleMatrix priorCovar = DoubleMatrix.zeros(2,2);
			priorCovar.put(0,0, vminSD);
			priorCovar.put(1,1, mqSD);
			DoubleMatrix priorMean = new DoubleMatrix(new double[] {vminTrue, mqTrue});
			BayesInf bi = new BayesInf(xLoad, new DoubleMatrix(additionalLoadDataV),priorCovar,priorMean, bayesInfVar);
//			Main.printMatrix(xLoad);
//			Main.printMatrix(new DoubleMatrix(additionalLoadDataV));
			if(!notLearning){
				bi.setup();
				if(PROFILING){
					System.out.println("Bayesian Inference - Time exceeded: "+(System.nanoTime()-time)/Math.pow(10,9)+" s");
				}

				mqLearned = bi.getMean().get(1);
				vminLearned =  bi.getMean().get(0)+mqLearned*qmin*kwhPerUnit;
			}

			if(log)
			System.out.println("qmin "+ qmin);

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
//			Main.printMatrix(predMeanPrices);
//			Main.printMatrix(Main.subVector(predictStart, predictStart+predictSize, priceSamples));
			// setup mdp
			if(PROFILING){
				time = System.nanoTime();
			}
			if(log){
				System.out.println("tdep: " + (tdepLearnedMean  + endOfDayOffset) );
				System.out.println("tstart: " + (tstart  + endOfDayOffset) );
				System.out.println("vmin: " + vmin  + " " + vminLearned );
				System.out.println("mq: " + mq  + " " + mqLearned );
				System.out.println("tcrit: " + (tcrit  + endOfDayOffset) );

			}

//			mqLearned = 20;
//			vminLearned = 400;
			EVMDP mdp = new EVMDP(predMeanPrices, predVarPrices, deltaPrice, numSteps, qmax, qmin, mqLearned, tstart + endOfDayOffset, tcrit + endOfDayOffset, vminLearned,tdepLearnedMean  + endOfDayOffset,tdepLearnedSD,currentLoadMDP,kwhPerUnit);
			mdp.setup();
//			System.out.println("Base");
//			mdp.printOptPolicy();
			// charge the vehicle according to policy and update total utility
			if(PROFILING){
				System.out.println("mdp calculation - Time exceeded: "+(System.nanoTime()-time)/Math.pow(10,9)+" s");
			}
			// sample tdep and load till tdep, calculate utility until tdep
			if(PROFILING){
				time = System.nanoTime();
			}
			// Check stopping criterion
			if(!notLearning){
				int stoppingRunsCounter = 0;
				double expectedUtility = mdp.getqValues()[0][mdp.priceToState(realPriceMatrix.get(0)+priceOffset)][mdp.loadToState(currentLoadMDP)][0];
//				System.out.println("base " + expectedUtility);
				double cumulatedDifference = 0;
				double[][] covarianceMatrix = new double[2][2];
//				Main.printMatrix(bi.getAInv());
				covarianceMatrix[0] = new double[] {bi.getAInv().get(0,0),bi.getAInv().get(0,1)};
				covarianceMatrix[1] = new double[] {bi.getAInv().get(1,0),bi.getAInv().get(1,1)};
				MultivariateNormalDistribution mvnd = new MultivariateNormalDistribution(new double[] {bi.getMean().get(0),bi.getMean().get(1)},covarianceMatrix);

				for(int o = 0; o < numberOfSamplesForStoppingCriterion/numberOfConcurrentThreads; o++){
//					if(log)
//					if(o%10 == 0)
//					System.out.println("stoppinval run: "+ o + " " + numberOfSamplesForStoppingCriterion/numberOfConcurrentThreads + "    " + runId);
					SimulationThread[] sts = new SimulationThread[numberOfConcurrentThreads];
					// start threads
					for(int i = 0; i < numberOfConcurrentThreads; i++){
						double[] sample = mvnd.sample();
						double sampledVmin = -1;
						double sampledMQ = -1;
						while(sampledVmin < 0 || sampledMQ < 0){
							sample = mvnd.sample();
							sampledVmin = sample[0] + qmin*kwhPerUnit * sample[1];
							sampledMQ = sample[1];
						}
//						sampledMQ = 15;
//						sampledVmin = 400;
//						if(log)
//							System.out.println(i +  " " + mqLearned + " " + vminLearned + " " + sampledVmin + " vmin mq " + sampledMQ + " eu " + expectedUtility + " sample 0 " + sample[0] + " sample 1 " + sample[1] + " bi 0 " + bi.getMean().get(0)+ " bi 1 " + bi.getMean().get(1));
//						System.out.println(sampledVmin + " " + sampledMQ);
						EVMDP samplemdp = new EVMDP(sampledMQ, sampledVmin, kwhPerUnit, mdp.getPriceProb(), mdp.getLoadProb(), mdp.getEndStateProb(), mdp.getPrices(), mdp.getLoads(), numSteps,tstart + endOfDayOffset, tcrit + endOfDayOffset,tdepLearnedMean  + endOfDayOffset, qmin, qmax,deltaPrice);


						int[] indexes = new int[2];
						indexes[0] = mdp.priceToState(realPriceMatrix.get(0)+priceOffset);
						indexes[1] = mdp.loadToState(currentLoadMDP);
						double sampleTdep = sampleFromNormal(tdepTrueMean, tdepTrueSD);

						SimulationThread thread = new SimulationThread(samplemdp,stoppingRunsCounter,mdp.getOptPolicy(),sampleTdep+endOfDayOffset,predMeanPrices,currentLoadMDP);
						thread.start();
						sts[i] = thread;

					}
					
					// collect threads and sum cumulateddifferences
					for(int i = 0; i < numberOfConcurrentThreads; i++){
						SimulationThread b = sts[i];
						try {
							b.thread.join();
//							if(log)
//							System.out.println("cd " + cumulatedDifference);
							cumulatedDifference += b.value;

						} catch (InterruptedException e) {
							// TODO Auto-generated catch block
							System.out.println(e.getMessage());
							e.printStackTrace();
						}

					}
				}
				double avgRegret = cumulatedDifference/numberOfSamplesForStoppingCriterion;
//				if(log)
					System.out.println("Rugrat " + avgRegret + " " +vminLearned + " " + mqLearned);
//				if(avgRegret < stoppingCriterionThreshold){
//					notLearning = true;
//				}
				regret.add(avgRegret);
				mqDiffs.add(mq-mqLearned);
				vminDiffs.add(vminTrue-vminLearned);
			}
			if(PROFILING){
				System.out.println("stopping criterion - Time exceeded: "+(System.nanoTime()-time)/Math.pow(10,9)+" s");
			}
			if(PROFILING){
				time = System.nanoTime();
			}
			// intialise the sorted min loader
			SortedMinLoader sml = new SortedMinLoader();
			sml.setup(Main.subVector(0, (int)(tdep + endOfDayOffset+1), predMeanPrices), qmax, currentLoadSML/kwhPerUnit);
			// load according to the different agents
			for(int o = 0; o < tdep + endOfDayOffset; o++){
				if(mdp.priceToState(realPriceMatrix.get(o)+priceOffset) < 0){
					System.out.println("Bad Price" + realPriceMatrix.get(o)+priceOffset);
				}
				if(mdp.loadToState(currentLoadMDP) < 0){
					System.out.println("Bad Load" + currentLoadMDP);
				}

				int actionMDP = mdp.getOptPolicy()[o][mdp.priceToState(realPriceMatrix.get(o)+priceOffset)][mdp.loadToState(currentLoadMDP)][0];
				// MDP: cost summation, save for every day			
				totalUtilityMDP += mdp.rewards(currentLoadMDP, actionMDP, realPriceMatrix.get(o)+priceOffset,o,0);
				costsPerDeltaTMDP.add(mdp.rewards(currentLoadMDP, actionMDP, realPriceMatrix.get(o)+priceOffset,o,0));
				currentLoadMDP = mdp.updateLoad(currentLoadMDP, actionMDP);
				loadsMDP.add(currentLoadMDP);
				actionsMDP.add((double)actionMDP);
				// Low price loader
				int actionLPL = LPL.actionQuery(currentLoadLPL/kwhPerUnit, realPriceMatrix.get(o)+priceOffset);
				totalUtilityLPL += mdp.rewards(currentLoadLPL, actionLPL, realPriceMatrix.get(o)+priceOffset,o,0);
				costsPerDeltaTLPL.add(mdp.rewards(currentLoadLPL, actionLPL, realPriceMatrix.get(o)+priceOffset,o,0));
				currentLoadLPL = mdp.updateLoad(currentLoadLPL, actionLPL);
				actionsLPL.add((double)actionLPL);
				loadsLPL.add(currentLoadLPL);
				// plug and forget loader
				int actionPAFL = PAFL.actionQuery(currentLoadPAFL/kwhPerUnit);
				totalUtilityPAFL += mdp.rewards(currentLoadPAFL, actionPAFL, realPriceMatrix.get(o)+priceOffset,o,0);
				costsPerDeltaTPAFL.add(mdp.rewards(currentLoadPAFL, actionPAFL, realPriceMatrix.get(o)+priceOffset,o,0));
				currentLoadPAFL = mdp.updateLoad(currentLoadPAFL, actionPAFL);
				actionsPAFL.add((double)actionPAFL);
				loadsPAFL.add(currentLoadPAFL);
				// SortedMinLoader
				int actionSML = sml.policy[o];
				totalUtilitySML += mdp.rewards(currentLoadSML, actionSML, realPriceMatrix.get(o)+priceOffset,o,0);
				costsPerDeltaTSML.add(mdp.rewards(currentLoadSML, actionSML, realPriceMatrix.get(o)+priceOffset,o,0));
				currentLoadSML = mdp.updateLoad(currentLoadSML, actionSML);
				actionsSML.add((double)actionSML);
				loadsSML.add(currentLoadSML);
				//record prices
				realPrices.add(realPriceMatrix.get(o));
				predictedPrices.add(predMeanPrices.get(o));
				timeStepsCounter.add((double)o+predictStart);
			}
			if(PROFILING){
				System.out.println("loading - Time exceeded: "+(System.nanoTime()-time)/Math.pow(10,9)+" s");
			}
			if(PROFILING){
				time = System.nanoTime();
			}
			if(log){
				System.out.println("endload MDP: " + currentLoadMDP);
				System.out.println("endload LPL: " + currentLoadLPL);
				System.out.println("endload PAFL: " + currentLoadPAFL);
				System.out.println("endload SML: " + currentLoadSML);

				System.out.println("costs MDP: " + totalUtilityMDP);
				System.out.println("costs LPL: " + totalUtilityLPL);
				System.out.println("costs PAFL: " + totalUtilityPAFL);
				System.out.println("costs SML: " + totalUtilitySML);
			}
			loadsDailyLPL.add(currentLoadLPL);
			loadsDailyMDP.add(currentLoadMDP);
			loadsDailyPAFL.add(currentLoadPAFL);
			loadsDailySML.add(currentLoadSML);
			costsDailyLPL.add(totalUtilityLPL);
			costsDailyMDP.add(totalUtilityMDP);
			costsDailyPAFL.add(totalUtilityPAFL);
			costsDailySML.add(totalUtilitySML);
			totalUtilityMDP = totalUtilityMDP + mdp.valueWithCustomVFunction(currentLoadMDP, (int)tdep, vminTrue, mqTrue);
			totalUtilityLPL = totalUtilityLPL + mdp.valueWithCustomVFunction(currentLoadLPL, (int)tdep, vminTrue, mqTrue);
			totalUtilityPAFL = totalUtilityPAFL + mdp.valueWithCustomVFunction(currentLoadPAFL, (int)tdep, vminTrue, mqTrue);
			totalUtilitySML = totalUtilitySML + mdp.valueWithCustomVFunction(currentLoadSML, (int)tdep, vminTrue, mqTrue);
			if(log){
				System.out.println("total utility MDP: " + totalUtilityMDP);
				System.out.println("total utility LPL: " + totalUtilityLPL);
				System.out.println("total utility PAFL: " + totalUtilityPAFL);
				System.out.println("total utility SML: " + totalUtilitySML);
			}
			counter++;
			if(PROFILING){
				System.out.println("update - Time exceeded: "+(System.nanoTime()-time)/Math.pow(10,9)+" s");
			}
			if(PROFILING){
				time = System.nanoTime();
			}
			// find out how much load was used during the day
			double usedLoad = currentLoadMDP;
			if(currentLoadMDP > 0){
				double tempload = qmin*kwhPerUnit;
				usedLoad = (int)sampleFromNormal(tempload, returnLoadSD);
				int c = 0;
				if(!(Math.abs(tempload-currentLoadMDP) < 0.001)){
					while((usedLoad > currentLoadMDP || usedLoad < tempload)&&c<100){
						usedLoad = (int)sampleFromNormal(tempload, returnLoadSD);
						c++;
						if(c == 100){
							usedLoad = currentLoadMDP;
						}
					}
				}
				if(log)
					System.out.println("used Load " + usedLoad);
				currentLoadMDP = Math.max(currentLoadMDP - usedLoad,0);
				if(log)
					System.out.println("used Load " + usedLoad);
			}
			if(PROFILING){
				System.out.println("return load sampling - Time exceeded: "+(System.nanoTime()-time)/Math.pow(10,9)+" s");
			}
			if(PROFILING){
				time = System.nanoTime();
			}
			currentLoadLPL = Math.max(0,currentLoadLPL - usedLoad);
			currentLoadPAFL = Math.max(0,currentLoadPAFL - usedLoad);
			currentLoadSML = Math.max(0,currentLoadSML - usedLoad);

			if(log){
				System.out.println("returnload MDP: " + currentLoadMDP);
				System.out.println("returnload LPL: " + currentLoadLPL);
				System.out.println("returnload PAFL: " + currentLoadPAFL);
				System.out.println("returnload SML: " + currentLoadSML);

				// TODO:
				System.out.println("predicstart: " + predictStart);
			}
			treturnMean = sampleFromNormal(predictStart + tdep + endOfDayOffset + unpluggedDuration, treturnSD);
			learnStart = (int)(treturnMean - learnSize);
			predictStart = (int)treturnMean;
			endOfDayOffset = numSteps-treturnMean%numSteps;
			currentLoadMDP = 0;
			if(log){
				System.out.println("tret: " + treturnMean);
				System.out.println("endofday offset: " + endOfDayOffset);
				System.out.println();
			}
			totalUtilitiesLPL.add(totalUtilityLPL);
			totalUtilitiesMDP.add(totalUtilityMDP);
			totalUtilitiesSML.add(totalUtilitySML);
			totalUtilitiesPAFL.add(totalUtilityPAFL);

			totalUtilityMDP = 0;
			totalUtilityLPL = 0;
			totalUtilityPAFL = 0;
			totalUtilitySML = 0;
			if(PROFILING){
				System.out.println("rest - Time exceeded: "+(System.nanoTime()-time)/Math.pow(10,9)+" s");
			}


		}

		DoubleMatrix dailyReport = DoubleMatrix.concatHorizontally(new DoubleMatrix(totalUtilitiesMDP), new DoubleMatrix(totalUtilitiesLPL));
		dailyReport = DoubleMatrix.concatHorizontally(dailyReport, new DoubleMatrix(totalUtilitiesPAFL));
		dailyReport = DoubleMatrix.concatHorizontally(dailyReport, new DoubleMatrix(totalUtilitiesSML));
		dailyReport = DoubleMatrix.concatHorizontally(dailyReport, new DoubleMatrix(costsDailyMDP));
		dailyReport = DoubleMatrix.concatHorizontally(dailyReport, new DoubleMatrix(costsDailyLPL));
		dailyReport = DoubleMatrix.concatHorizontally(dailyReport, new DoubleMatrix(costsDailyPAFL));
		dailyReport = DoubleMatrix.concatHorizontally(dailyReport, new DoubleMatrix(costsDailySML));
		dailyReport = DoubleMatrix.concatHorizontally(dailyReport, new DoubleMatrix(loadsDailyMDP));
		dailyReport = DoubleMatrix.concatHorizontally(dailyReport, new DoubleMatrix(loadsDailyLPL));
		dailyReport = DoubleMatrix.concatHorizontally(dailyReport, new DoubleMatrix(loadsDailyPAFL));
		dailyReport = DoubleMatrix.concatHorizontally(dailyReport, new DoubleMatrix(loadsDailySML));

		DoubleMatrix timeStepReport = DoubleMatrix.concatHorizontally(new DoubleMatrix(actionsMDP), new DoubleMatrix(actionsLPL));
		timeStepReport = DoubleMatrix.concatHorizontally(timeStepReport, new DoubleMatrix(actionsPAFL));
		timeStepReport = DoubleMatrix.concatHorizontally(timeStepReport, new DoubleMatrix(actionsSML));
		timeStepReport = DoubleMatrix.concatHorizontally(timeStepReport, new DoubleMatrix(loadsMDP));
		timeStepReport = DoubleMatrix.concatHorizontally(timeStepReport, new DoubleMatrix(loadsLPL));
		timeStepReport = DoubleMatrix.concatHorizontally(timeStepReport, new DoubleMatrix(loadsPAFL));
		timeStepReport = DoubleMatrix.concatHorizontally(timeStepReport, new DoubleMatrix(loadsSML));
		timeStepReport = DoubleMatrix.concatHorizontally(timeStepReport, new DoubleMatrix(costsPerDeltaTMDP));
		timeStepReport = DoubleMatrix.concatHorizontally(timeStepReport, new DoubleMatrix(costsPerDeltaTLPL));
		timeStepReport = DoubleMatrix.concatHorizontally(timeStepReport, new DoubleMatrix(costsPerDeltaTPAFL));
		timeStepReport = DoubleMatrix.concatHorizontally(timeStepReport, new DoubleMatrix(costsPerDeltaTSML));
		timeStepReport = DoubleMatrix.concatHorizontally(timeStepReport, new DoubleMatrix(predictedPrices));
		timeStepReport = DoubleMatrix.concatHorizontally(timeStepReport, new DoubleMatrix(realPrices));
		timeStepReport = DoubleMatrix.concatHorizontally(timeStepReport, new DoubleMatrix(timeStepsCounter));

		DoubleMatrix stoppingReport = DoubleMatrix.concatHorizontally(DoubleMatrix.concatHorizontally(new DoubleMatrix(regret), new DoubleMatrix(mqDiffs)), new DoubleMatrix(vminDiffs));

		String fileName = "vmin" + (int)vminvarInput + "mq" + (int)mqInput + "kwh" + (int)kwhPerUnit + "tstart" + (int)tstartTrue + "tcrit" + (int)tcritInput + "qmax" +(int)qmaxInput + "qmin" + (int)(qminInput) + "bivar" +(int)(bayesInfVar) ;
		new File(fileHandleOut + fileName).mkdir();
		FileHandler.safeDailyReport(dailyReport, fileHandleOut  + fileName + "/daily.csv");
		FileHandler.safeTimeStepReport(timeStepReport, fileHandleOut + fileName + "/timestep.csv");
		FileHandler.safeStoppingReport(stoppingReport, fileHandleOut + fileName + "/stopping.csv");
//		if(!PROFILING){
//			System.out.println("Total - Time exceeded: "+(System.nanoTime()-time)/Math.pow(10,9)+" s " + runId);
//		}		
		String append = vminvarInput + "," + mqInput + "," + kwhPerUnit + "," + tstartTrue + "," + tcritTrue + "," +qmaxInput + "," + (int)(qminInput) + "," + bayesInfVar+",";
		String totalCosts = sum(costsDailyMDP)+ ","+ sum(costsDailyLPL)+","+ sum(costsDailyPAFL) +","+ sum(costsDailySML)+ ",";
		String totalLoads = sum(loadsDailyMDP) + "," +sum(loadsDailyLPL) + "," +sum(loadsDailyPAFL) + "," +sum(loadsDailySML) + "," ;
		String totalUtils = sum(totalUtilitiesMDP) + "," +sum(totalUtilitiesLPL) + "," +sum(totalUtilitiesPAFL) + "," +sum(totalUtilitiesSML) + ",";
		String appendend = loadsMDP.size() + "," + regret.size() + "," + sum(regret)+ "," + sum(mqDiffs) + "," +sum(vminDiffs) + ",";
		append += totalCosts +totalLoads + totalUtils + appendend + "\n";
		if(append.split(",").length != 26){
			try {
			    Files.write(Paths.get(fhout+ "/err.txt"),("grave Error: "+ append).getBytes(), StandardOpenOption.APPEND);
			}catch (IOException e) {
			}
		}
		try {
		    Files.write(Paths.get(fhcomp), append.getBytes(), StandardOpenOption.APPEND);
		}catch (IOException e) {
		}
		//		FileHandler.matrixToCsv(printOut, fileHandleOut);

	}

	
	public double avg(ArrayList<Double> input){
		double result = 0;
		for(Double d: input){
			result += d;
		}
		return result/input.size();
	}
	public double sum(ArrayList<Double> input){
		double result = 0;
		for(Double d: input){
			result += d;
		}
		return result;
	}
	
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

	public double averageOverQvalues(double[][][][] qvals, int j){
		double result = 0;
		for(int i = 0; i < qvals[0].length; i++){
//			for(int j = 0; j < qvals[0][i].length; j++){
				result += qvals[0][i][j][0];
//			}
		}
		return result/(qvals[0].length);//*qvals[0][0].length);
	}
	
	public int getNextPriceIndex(double[][][] priceProbs, int j, int k){
		double rnd = Random.nextDouble();
		double summedProb = 0;
		for(int i = 0; i < priceProbs.length; i++){
//			System.out.print(summedProb + " ");
			summedProb += priceProbs[i][j][k];
			if(rnd <= summedProb ){
				return i;
			} 
		}
		System.out.println("error " + rnd + " " + summedProb);
		for(int i = 0; i < priceProbs.length; i++){
			System.out.print( priceProbs[i][j][k]+ " ");
			
		}
		return -1;
	}
	
	public class SimulationThread implements Runnable{
		public EVMDP mdp;
		public int counter;
		public double value;
		public Thread thread;
		public int[][][][] optPolicy;
		public double tdep;
		public DoubleMatrix prices;
		public double load;
		private final Object lock = new Object();

		public SimulationThread(EVMDP emdp, int counter,int[][][][] optPolicy, double tdep, DoubleMatrix prices, double initialLoad){
			this.mdp = emdp;
			this.counter = counter;
			this.optPolicy = optPolicy;
			this.tdep = tdep;
			this.prices = prices;
			this.load = initialLoad;
		}
		@Override
		public void run() {
			// TODO Auto-generated method stub
			mdp.fastSetup();
			int numruns = 100;
			double expectedSampleUtility = 0;//averageOverQvalues(mdp.getqValues(),mdp.loadToState(load));
			double expectedBaseUtility = 0;
			double initialLoad = load;
			double basePLoad = load;
			double otherLoad = load;
//			mdp.printOptPolicy();
//			System.out.println("tdep " + tdep);
			for(int j = 0; j < numruns; j++){
				double cost = 0;
				String actionstring = "";
				int rnd = Random.nextInt(mdp.getPrices().length-1); //verify
				double utilityWithBasePolicy = 0;
				int action = optPolicy[0][rnd][mdp.loadToState(basePLoad)][0];
				basePLoad = mdp.updateLoad(basePLoad, action);
				utilityWithBasePolicy += mdp.rewards(basePLoad, action,mdp.getPrices()[rnd] ,0,0);
				
				double utilityWithSampledPolicy = 0;
				int otherAction = mdp.getOptPolicy()[0][rnd][mdp.loadToState(otherLoad)][0];
				otherLoad = mdp.updateLoad(otherLoad, otherAction);
				utilityWithSampledPolicy += mdp.rewards(otherLoad, otherAction,mdp.getPrices()[rnd] ,0,0);
				
				
//				cost += mdp.getPrices()[rnd]*action;

				int nextPriceIndex = getNextPriceIndex(mdp.getPriceProb(),rnd,0);
//				System.out.println(nextPriceIndex + " " + action + " " + otherAction);
				for(int i = 1; i < tdep; i++){
					action = optPolicy[i][nextPriceIndex][mdp.loadToState(basePLoad)][0];
					utilityWithBasePolicy += mdp.rewards(basePLoad, action,mdp.getPrices()[nextPriceIndex] ,i,0);
					basePLoad = mdp.updateLoad(basePLoad, action);

//					System.out.println("cost base " + mdp.rewards(basePLoad, action,mdp.getPrices()[nextPrice] ,i,0) + " " +mdp.getPrices()[nextPrice] + " " + action + " " + basePLoad);
					
					otherAction = mdp.getOptPolicy()[i][nextPriceIndex][mdp.loadToState(otherLoad)][0];
					utilityWithSampledPolicy += mdp.rewards(otherLoad, otherAction,mdp.getPrices()[nextPriceIndex] ,i,0);
//					System.out.println(mdp.getPrices()[nextPriceIndex] + " " + action + " " + otherAction + " " + mdp.rewards(otherLoad, otherAction,mdp.getPrices()[nextPriceIndex] ,i,0));

					otherLoad = mdp.updateLoad(otherLoad, otherAction);

//					System.out.println("cost sampled " + mdp.rewards(otherLoad, otherAction,mdp.getPrices()[nextPrice] ,i,0) + " " +mdp.getPrices()[nextPrice]+ " " + action + " " + otherLoad);
//					if(action != otherAction)
//					actionstring += action + " " + otherAction + " " + mdp.getPrices()[nextPriceIndex] + ", ";
//					cost += mdp.getPrices()[nextPriceIndex]*action;

					nextPriceIndex = getNextPriceIndex(mdp.getPriceProb(),nextPriceIndex,i);


//					if(nextPriceIndex == -1){
//						mdp.printPriceProbs();
//					}
				}

//				if(utilityWithBasePolicy != utilityWithSampledPolicy){
//					System.out.println("ub,us,c " + utilityWithBasePolicy + " " + utilityWithSampledPolicy + " " + cost);
//				}
//				System.out.println("base " + basePLoad + " " + otherLoad + " " + utilityWithBasePolicy + " " + utilityWithSampledPolicy);

				utilityWithSampledPolicy +=  mdp.rewards(otherLoad, action,0 ,(int)tdep,1);

				utilityWithBasePolicy += mdp.rewards(basePLoad, action,0 ,(int)tdep,1);
				
//				if(utilityWithBasePolicy > utilityWithSampledPolicy){
//				} else if(utilityWithBasePolicy < utilityWithSampledPolicy){
//					System.out.println("other " + basePLoad + " " + otherLoad );
//
//				}
				expectedBaseUtility += utilityWithBasePolicy;
				expectedSampleUtility += utilityWithSampledPolicy;
//				System.out.println("base " + basePLoad + " " + otherLoad + " " + utilityWithBasePolicy + " " + utilityWithSampledPolicy);

				basePLoad = initialLoad;
				otherLoad = initialLoad;

//				System.out.println();

			}
			expectedBaseUtility /= numruns;
			expectedSampleUtility /= numruns;
			//			
			
//			int action = optPolicy[0][mdp.priceToState(prices.get(0,0))][mdp.loadToState(load)][0];
//			load = mdp.updateLoad(load, action);
//			utilityWithBasePolicy += mdp.rewards(load, action,mdp.getPrices()[rnd] ,0,0);
//			for(int i = 1; i < tdep; i++){
////				System.out.println(action + " " + load);
//				action = optPolicy[i][mdp.priceToState(prices.get(i,0))][mdp.loadToState(load)][0];
//				load = mdp.updateLoad(load, action);
//				utilityWithBasePolicy += mdp.rewards(load, action,prices.get(i,0) ,i,0);
//			}
			
			value = expectedSampleUtility - expectedBaseUtility;
//			if(expectedBaseUtility > expectedSampleUtility){
////				System.out.println("diff");
////				mdp.printOptPolicy();
//			System.out.println("diff " + mdp.getqSlope() + " " + mdp.getvMin());
//			} else if(expectedBaseUtility < expectedSampleUtility) {
//				System.out.println("other " + mdp.getqSlope() + " " + mdp.getvMin());
//
//			}

		}
		public void start(){
			this.thread = new Thread(this);
			this.thread.start();
		}


	}


}
